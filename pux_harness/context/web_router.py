"""``WebRouterMiddleware`` — the auto-firing websearch router.

Inspired by the "Supra-Router" idea: a cheap scan of the latest USER turn; if it
needs fresh external info, the router fires a WORKER-model web round (reusing
the org's already-armed ``web_research`` MCP tools) and injects a compact,
URL-cited brief, so the big CTO model never spends its OWN turn calling a web
tool — fresh context is already in the message list when it reads the turn.

Two router flavors:
- **Heuristic** (default, free): a conservative deterministic classifier — only
  turns that clearly reach for info likely OUTSIDE training trip it
  (recency words, explicit "look up", version numbers, future/event refs).
  Under-fire > spam: the common case (code, files, known facts) never fires.
- **Model router** (opt-in via ``profile.yaml`` ``web_router: {model_router:
  true}``): one cheap worker call returning ``NEEDS_WEB: <query>`` or ``NONE``
  — the literal "cheap classifier" spirit, for orgs that want higher recall.

Why the router runs its OWN round instead of delegating to the ``web-search``
subagent (Part B1): deepagents exposes subagents ONLY via the supervisor's
``task(...)`` tool — there is no callable handle a middleware can invoke. So the
router runs the SAME focused ``web_research`` tools + worker digest directly,
producing the identical outcome the original ask described. The ``web-search``
subagent still exists for EXPLICIT delegation when the CTO wants a deeper,
multi-source pass. Two complementary paths, not one replacing the other.

The router is an ENHANCEMENT, not a gate: a web-round failure (server down,
tool error) is LOGGED + skipped — the model still answers, and can still call
its own ``web-search`` subagent / web tools. It never fakes context: if NO
``web_research`` tool is armed, ``_build_web_router`` returns ``None`` and the
middleware never mounts (the round isn't synthesized from nothing)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

_log = logging.getLogger(__name__)

# The marker every injected brief carries — also the de-dupe key. A round for
# the SAME query already in the message list is skipped (no re-fetch spam).
AUTO_MARKER = "[Auto web context for"

# --- heuristic router -------------------------------------------------------
# Conservative: only turns that clearly reach for info likely OUTSIDE training.
_RECENCY = re.compile(
    r"\b(latest|current|today|tonight|this week|this month|this year|"
    r"right now|recently|just (released|announced|dropped|came out|shipped|posted)|"
    r"news|headline|press release|changelog|release notes|"
    r"as of|up to date|updated to|now\b)\b",
    re.IGNORECASE,
)
_LOOKUP = re.compile(
    r"\b(search the web|search web|google it|look up|look this up|look it up|"
    r"find out|find the latest|find me|can you (check|search|look)|websearch|web search)\b",
    re.IGNORECASE,
)
_VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
_FUTURE = re.compile(
    r"\b(will be|upcoming|coming (soon|out)|scheduled|expected|"
    r"next (week|month|year|version|release)|roadmap|ETA|release date)\b",
    re.IGNORECASE,
)


def _strip_content(content: Any) -> str:
    """Flatten a LangChain message ``content`` (str OR a list of content
    blocks — text/image/tool dicts) to plain text for scanning / digest."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return " ".join(parts)
    return str(content or "")


def _latest_human_text(messages: Sequence[Any]) -> str | None:
    """The latest ``HumanMessage``'s text, or ``None`` (no human turn / only
    tool results). The router acts on USER turns — a turn that's pure
    tool-result is the model mid-loop and must not trigger a round."""
    for msg in reversed(list(messages)):
        if isinstance(msg, HumanMessage):
            return _strip_content(msg.content)
    return None


def heuristic_needs_web(text: str) -> str | None:
    """Deterministic, CONSERVATIVE router. Return a search query if the turn
    clearly reaches for fresh external info, else ``None`` (under-fire > spam).

    The query is the whole turn (whitespace-collapsed + capped) — the worker
    digest + the ``research`` tool narrow it. Returns ``None`` for the common
    case (code, files, known facts, opinions) so the router never fires on it."""
    if not text or not text.strip():
        return None
    if not (
        _RECENCY.search(text) or _LOOKUP.search(text)
        or _VERSION.search(text) or _FUTURE.search(text)
    ):
        return None
    query = " ".join(text.split())
    if len(query) > 240:
        query = query[:240].rsplit(" ", 1)[0]  # cut on a word boundary
    return query


# --- model router (opt-in) --------------------------------------------------
_MODEL_ROUTER_PROMPT = (
    "You are a cheap web-router classifier. Decide if the user's latest turn "
    "needs FRESH information from the live web — recent news, current versions, "
    "live prices, events after your training, anything likely stale or external.\n"
    "Reply with EXACTLY one line, nothing else:\n"
    "  `NEEDS_WEB: <a concise web search query>`  if it does, or\n"
    "  `NONE`  if it does not (the turn is about code, files, known facts, "
    "opinions, or anything already in your knowledge).\n"
    "Default to NONE when unsure — under-fire is far better than spamming web calls."
)


def _parse_model_router(content: Any) -> str | None:
    """Parse the worker classifier's one-line reply. ``NEEDS_WEB: <q>`` -> q;
    anything else (incl. ``NONE``) -> ``None``."""
    for line in _strip_content(content).splitlines():
        line = line.strip()
        if not line:
            continue
        if line[:10].upper().replace(" ", "") == "NEEDS_WEB:":
            query = line.split(":", 1)[1].strip() if ":" in line else ""
            return query or None
        return None  # first content line wasn't NEEDS_WEB -> NONE
    return None


async def model_router_needs_web(worker: Any, text: str) -> str | None:
    """The opt-in worker-classifier: one cheap call returning a query or None.
    Defaults to NONE (the prompt enforces it) so a weak worker can't spam."""
    try:
        resp = await worker.ainvoke(
            [SystemMessage(_MODEL_ROUTER_PROMPT), HumanMessage(text)]
        )
    except Exception as exc:  # classifier failure -> fall back to no-round
        _log.error("web-router model classifier failed: %r", exc)
        return None
    return _parse_model_router(resp.content)


# --- the web round ----------------------------------------------------------
_DIGEST_PROMPT = (
    "You are digesting raw web research into a compact brief for an orchestrator "
    "model that will act on it. Rules:\n"
    "- <=160 words. Hard cap.\n"
    "- Only claims SUPPORTED by the research below; cite the source URL inline "
    "as `(source)` or list up to 3 URLs at the end.\n"
    "- If the research is empty or has nothing relevant, reply EXACTLY: "
    "`No relevant fresh results.` (do not invent facts).\n"
    "- Lead with the single most decision-relevant fact.\n"
    "- Plain text, no preamble."
)


def _stringify(raw: Any, limit: int = 6000) -> str:
    """Flatten a tool result (str / dict / list) to a bounded string for the
    digest prompt. MCP tools return JSON-y payloads; truncation keeps the
    digest prompt cheap."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = raw
    else:
        try:
            text = json.dumps(raw, ensure_ascii=False, default=str)
        except Exception:
            text = str(raw)
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


def _pick(web_tools: Sequence[BaseTool], name: str) -> BaseTool | None:
    for t in web_tools:
        if t.name == name:
            return t
    return None


async def arun_web_round(
    query: str, web_tools: Sequence[BaseTool], worker: Any,
) -> str:
    """Fire one web round (worker model) + digest to a compact cited brief.

    Prefers the one-shot ``mcp__web_research__research`` tool (search + read +
    rerank in one call); falls back to ``search`` results (titles + snippets +
    URLs) when ``research`` isn't armed. Adapting to the armed toolset is NOT a
    silent fallback — it uses what the org DECLARED; if NO web_research tool is
    armed the middleware never mounts (``_build_web_router`` returns None)."""
    research = _pick(web_tools, "mcp__web_research__research")
    if research is not None:
        raw = await research.ainvoke({"query": query, "max_results": 3, "depth": "quick"})
    else:
        search = _pick(web_tools, "mcp__web_research__search")
        if search is None:
            return "No relevant fresh results."  # no research/search tool -> empty
        raw = await search.ainvoke({"query": query, "max_results": 3})
    digest = await worker.ainvoke(
        [SystemMessage(_DIGEST_PROMPT),
         HumanMessage(f"Query: {query}\n\nResearch:\n{_stringify(raw)}")],
    )
    brief = _strip_content(digest.content).strip()
    return brief or "No relevant fresh results."


def run_web_round(
    query: str, web_tools: Sequence[BaseTool], worker: Any,
) -> str:
    """Sync parity for ``arun_web_round`` (sync agent builds). Same logic via
    ``.invoke``. The async path is the runtime-proven one (langgraph/deepagents
    drive the agent async); this exists so a sync build doesn't silently skip."""
    research = _pick(web_tools, "mcp__web_research__research")
    if research is not None:
        raw = research.invoke({"query": query, "max_results": 3, "depth": "quick"})
    else:
        search = _pick(web_tools, "mcp__web_research__search")
        if search is None:
            return "No relevant fresh results."
        raw = search.invoke({"query": query, "max_results": 3})
    digest = worker.invoke(
        [SystemMessage(_DIGEST_PROMPT),
         HumanMessage(f"Query: {query}\n\nResearch:\n{_stringify(raw)}")],
    )
    brief = _strip_content(digest.content).strip()
    return brief or "No relevant fresh results."


# --- the middleware ---------------------------------------------------------
class WebRouterMiddleware(AgentMiddleware):
    """Auto-fires a worker-model web round when the latest user turn needs fresh
    external info, injecting a compact cited brief so the big CTO model never
    spends its own turn on websearch.

    Constructed by ``stack._build_web_router`` ONLY when the org arms at least
    one ``mcp__web_research__*`` tool; mounted via ``profile.yaml``
    ``middleware.supervisor.add: [web-router]`` (NOT default-on). Innermost wrap
    (registry order LAST) -> runs right before the model, after the context +
    routing layers assembled the message list — the correct spot for an
    injector."""

    def __init__(
        self, *, web_tools: Sequence[BaseTool], worker: Any,
        use_model_router: bool = False, org: str = "",
    ) -> None:
        self.web_tools = list(web_tools)
        self.worker = worker
        self.use_model_router = use_model_router
        self.org = org

    def _already_injected(self, messages: Sequence[Any], query: str) -> bool:
        """De-dupe: skip if a brief for the SAME query is already present (the
        turn may re-enter the model across tool loops; don't re-fetch)."""
        marker = f'{AUTO_MARKER} "{query}"]'
        for msg in messages:
            if marker in _strip_content(getattr(msg, "content", "")):
                return True
        return False

    async def awrap_model_call(self, request, handler):  # type: ignore[no-untyped-def]
        text = _latest_human_text(getattr(request, "messages", []))
        if text is None:
            return await handler(request)  # tool-result turn / no human msg
        if self.use_model_router:
            query = await model_router_needs_web(self.worker, text)
        else:
            query = heuristic_needs_web(text)
        if not query:
            return await handler(request)
        if self._already_injected(getattr(request, "messages", []), query):
            return await handler(request)
        try:
            brief = await arun_web_round(query, self.web_tools, self.worker)
        except Exception as exc:
            # Enhancement, not a gate: log + proceed without context. The model
            # can still call its own web-search subagent / web tools.
            _log.error("web-router round for %r failed (skipped): %r", query, exc)
            return await handler(request)
        messages = getattr(request, "messages", None)
        if messages is not None and brief:
            messages.insert(0, HumanMessage(f'{AUTO_MARKER} "{query}"]:\n{brief}'))
            _log.info("web-router fired round for %r; injected %d-char brief", query, len(brief))
        else:
            _log.info("web-router round for %r yielded no brief (skipped)", query)
        return await handler(request)

    def wrap_model_call(self, request, handler):  # type: ignore[no-untyped-def]
        text = _latest_human_text(getattr(request, "messages", []))
        if text is None:
            return handler(request)
        # The model-router classifier is async-only; sync builds use the free
        # heuristic (sync agent paths are test/offline — the runtime is async).
        query = heuristic_needs_web(text)
        if not query:
            return handler(request)
        if self._already_injected(getattr(request, "messages", []), query):
            return handler(request)
        try:
            brief = run_web_round(query, self.web_tools, self.worker)
        except Exception as exc:
            _log.error("web-router round for %r failed (skipped): %r", query, exc)
            return handler(request)
        messages = getattr(request, "messages", None)
        if messages is not None and brief:
            messages.insert(0, HumanMessage(f'{AUTO_MARKER} "{query}"]:\n{brief}'))
            _log.info("web-router fired round for %r; injected %d-char brief", query, len(brief))
        else:
            _log.info("web-router round for %r yielded no brief (skipped)", query)
        return handler(request)
