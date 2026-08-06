"""The unified context-saving middleware.

One ``AgentMiddleware`` that does BOTH jobs in a single ``wrap_tool_call``
pass — so every tool result is read exactly once, timed once, and written
through one store:

1. **Capture** — record a structured ``tool_call`` (or ``error``) event:
   tool name, truncated args, success, elapsed seconds, and a 300-char
   output preview. These events feed the snapshot builder (cross-session
   rehydration) and the ``ctx_search`` retrieval surface.
2. **Offload** — when a tool returns more than ``threshold`` chars of plain
   text, park the FULL bytes in the store as a blob and replace the
   ``ToolMessage`` the model sees with a short preview + a ``ctx:<id>``
   retrieval handle. This is *proactive*: it keeps large results out of the
   context window BEFORE they accumulate, complementing deepagents' reactive
   ``SummarizationMiddleware`` (which only evicts on overflow).

This merges the old ``ContextOffloadMiddleware`` + ``EventCaptureMiddleware``
(two passes, two stores) into one. It rides the
MAIN agent AND every subagent — ``build_context_layer()`` is the single seam
both ``graph.py`` and ``orgs._build_sub`` import, so the whole agent tree
gets capture + offload + retrieval with zero duplication.

Retrieval tools (``ctx_recall`` / ``ctx_search``) are **exempt** from both
jobs: their purpose is to inject content, so re-stashing their output would
trap the agent the instant it retrieves a large stash (proven in E2E
before the exemption), and their output needs no event preview.
"""
from __future__ import annotations

import json
import time
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents_context.store import EventStore, StashResult, shared_event_store

# Defaults: ~2k tokens (chars/4). Tuned to catch directory dumps, big greps,
# log tails, and JSON blobs without trimming ordinary tool replies.
DEFAULT_THRESHOLD = 8000
DEFAULT_PREVIEW = 1500

# Browser specialist tools all share this prefix (mirrors browser_vision.py).
# Used by the browser-specific structural offload below.
_BROWSER_PREFIX = "pux_sandbox_browser_"

# The retrieval + meta surface — ctx_recall/ctx_search exist to bring (a slice
# of) stashed content back INTO context. Offloading OR preview-logging their
# output would defeat that (and re-stash a big recall instantly). So they are
# exempt from BOTH capture-detail and offload. The meta tools (ctx_stats,
# ctx_doctor, ctx_purge) and the indexing tool (ctx_index) are exempt too:
# their outputs are small by design (a one-line ack, a JSON blob, a check
# list), and ctx_index's PURPOSE is to park content in the store — offloading
# its tiny "Indexed N chars" ack would be noise. None of these names carry
# large agent-facing payloads, so the exemption is safe.
#
# The exec tools (ctx_execute / ctx_execute_file / ctx_batch_execute /
# ctx_fetch_and_index) are ALSO exempt: their entire purpose is to keep large
# outputs OUT of context (the agent asked for stdout-only, or the fetch already
# indexed into the store). Re-offloading their result would double-handle the
# content. ctx_batch_execute + ctx_fetch_and_index explicitly stash to the
# store themselves and return a short summary — offloading that summary is noise.
_RETRIEVAL_TOOLS = frozenset({
    "ctx_recall",
    "ctx_search",
    "ctx_index",
    "ctx_stats",
    "ctx_doctor",
    "ctx_purge",
    "ctx_execute",
    "ctx_execute_file",
    "ctx_batch_execute",
    "ctx_fetch_and_index",
})


def _is_text_tm(result: Any) -> bool:
    """Only offload a real ToolMessage with plain-string content. Multimodal
    content (lists of blocks), Command returns, and non-ToolMessage results pass
    through untouched — offloading structured/image output would lose it."""
    return isinstance(result, ToolMessage) and isinstance(result.content, str)


def _stub(stash: StashResult, content: str, tool: str, *, preview: int) -> str:
    """The replacement message the model sees instead of the full blob.

    Phrased as a plain TRUNCATED TOOL RESULT with a one-line continuation
    pointer — deliberately NOT a system notice: no ``[bracketed]`` banner, no
    "offload/stashed/context" jargon, no rationale. A skeptical live model
    reads ``[ctx-offload] ... stashed so it doesn't crowd the working context``
    as a management artifact and second-guesses whether later content is
    "real" (observed: MiMo-2.5 transcribed a unique tail correctly but then
    argued it was an injected context-management marker). Phrased as a normal
    truncation, the model just calls ``ctx_recall`` for the rest — the right
    action is the path of least resistance, no reasoning required.

    The ``ctx:<id>`` handle in the pointer IS the structural offload signal
    (a non-truncated tool result never carries one); tests key on it rather
    than on any phrasing."""
    head = content[:preview]
    ellipsis = "…" if len(content) > preview else ""
    return (
        f"{tool} returned {stash.chars} chars (first {preview} shown). "
        f"For the complete output, call ctx_recall({stash.handle!r}).\n\n"
        f"{head}{ellipsis}"
    )


def _trim_browser_payload(
    content: str, store: EventStore, tool_name: str, *,
    threshold: int, thread_id: str = "",
) -> str | None:
    """Stash the heavy fields of a browser page-result and return a slim JSON
    keeping only the skeleton the agent + vision middleware need inline.

    Browser tools (navigate, click, …) return ``page_data`` with body text, all
    links, and all image URLs — easily 4K+ chars that the model re-reads EVERY
    turn (the full message history is resent each turn, so N observations cost
    O(N²) tokens). This stashes the full payload and returns a skeleton keeping
    ``ok``, ``page_data.{url,title}``, a 200-char text preview, ``element_map``
    (the agent clicks by index), and ``screenshot_path`` (the vision middleware
    reads it). The agent recalls the full content via ``ctx_recall`` only when
    it actually needs the body text / links / images.

    Returns the slim JSON string, or ``None`` if the content isn't a browser
    page-result with heavy fields worth trimming (below ``threshold``)."""
    if len(content) <= threshold:
        return None  # small enough to keep inline — no stash friction
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    pd = payload.get("page_data")
    if not isinstance(pd, dict):
        return None  # not a browser page-result shape (e.g. /cookies, /evaluate)
    heavy_keys = [k for k in ("text", "links", "images") if pd.get(k)]
    if not heavy_keys:
        return None  # nothing heavy to stash
    stash = store.stash_blob(content, tool=tool_name, thread_id=thread_id)
    slim = dict(payload)
    slim_pd = {k: v for k, v in pd.items() if k not in ("text", "links", "images")}
    # keep a 200-char text preview so the agent can orient without a recall
    full_text = pd.get("text") or ""
    slim_pd["text"] = full_text[:200] + ("…" if len(full_text) > 200 else "")
    slim["page_data"] = slim_pd
    heavy_desc = ", ".join(
        f"{k} ({len(pd[k])} {'chars' if k == 'text' else 'items'})" for k in heavy_keys
    )
    slim["context_note"] = (
        f"Page {heavy_desc} stashed to keep context lean — "
        f"call ctx_recall({stash.handle!r}) for the full page content."
    )
    return json.dumps(slim, ensure_ascii=False, default=str)


def _offload_toolmessage(
    result: ToolMessage, store: EventStore, tool_name: str, *,
    threshold: int, preview: int, thread_id: str = "",
) -> Any:
    """Offload a single plain-text ToolMessage. Handles BOTH the browser
    structural trim (stash page_data heavy fields, keep skeleton) and the
    generic char-threshold offload (big greps, directory dumps). Returns the
    original object unchanged if nothing to trim."""
    if not _is_text_tm(result):
        return result
    # Browser structural trim fires FIRST: even a payload that's "only" 9K chars
    # carries 4K of body text the model doesn't need every turn. Stash the heavy
    # fields, keep the interaction skeleton inline.
    if tool_name.startswith(_BROWSER_PREFIX):
        slim = _trim_browser_payload(
            result.content, store, tool_name,
            threshold=threshold, thread_id=thread_id,
        )
        if slim is not None:
            return ToolMessage(
                content=slim, tool_call_id=result.tool_call_id, name=result.name,
            )
    # Generic char-threshold offload.
    if len(result.content) <= threshold:
        return result
    stash = store.stash_blob(result.content, tool=tool_name, thread_id=thread_id)
    return ToolMessage(
        content=_stub(stash, result.content, tool_name, preview=preview),
        tool_call_id=result.tool_call_id,
        name=result.name,
    )


def _offload_command(
    command: Command, store: EventStore, tool_name: str, *,
    threshold: int, preview: int, thread_id: str = "",
) -> Command:
    """Reach inside a browser-vision ``Command`` and trim the text ToolMessage,
    preserving the companion image HumanMessage(s) untouched.

    ``BrowserVisionMiddleware`` (innermost) wraps every browser result as
    ``Command(update={"messages": [text_tm, image_human]})``. Without this
    helper the outer ``ContextMiddleware`` sees a Command (not a ToolMessage),
    ``_is_text_tm`` returns False, and the ENTIRE text payload — 4K of body
    text + 30 links + 50 images — escapes offload every turn. This trims the
    text TM in place and rebuilds the Command with the image(s) intact."""
    msgs = command.update.get("messages", [])
    if not msgs:
        return command
    new_msgs: list = []
    changed = False
    for m in msgs:
        if _is_text_tm(m):
            trimmed = _offload_toolmessage(
                m, store, tool_name,
                threshold=threshold, preview=preview, thread_id=thread_id,
            )
            if trimmed is not m:
                new_msgs.append(trimmed)
                changed = True
                continue
        new_msgs.append(m)
    if not changed:
        return command
    return Command(update={"messages": new_msgs})


def _offload(
    result: Any, store: EventStore, tool_name: str, *,
    threshold: int, preview: int, thread_id: str = "",
) -> Any:
    """If ``result`` is an oversized text ToolMessage (or a browser-vision
    Command wrapping one), stash + replace. Else return unchanged. Pure (no I/O
    beyond the store); called by both the sync and async wrap hooks so behavior
    is identical either way.

    ``threshold <= 0`` is a kill-switch: offload nothing (handy for tests + a
    future env-flag to disable the feature without unwiring the middleware).

    Retrieval tools (``ctx_recall``/``ctx_search``) are exempt — see
    ``_RETRIEVAL_TOOLS``."""
    if threshold <= 0 or tool_name in _RETRIEVAL_TOOLS:
        return result
    # Browser-vision middleware (innermost) wraps results as a Command before
    # we see them — reach inside and trim the text TM, keep the image.
    if isinstance(result, Command):
        return _offload_command(
            result, store, tool_name,
            threshold=threshold, preview=preview, thread_id=thread_id,
        )
    return _offload_toolmessage(
        result, store, tool_name,
        threshold=threshold, preview=preview, thread_id=thread_id,
    )


class ContextMiddleware(AgentMiddleware):
    """Capture every tool call as an event AND offload oversized results to a
    blob — one ``wrap_tool_call`` pass, one store.

    Knobs:
    - ``threshold`` (default 8000 chars): results this size or larger are
      stashed behind a ``ctx:<id>`` handle. Set ``<= 0`` to disable offload.
    - ``preview`` (default 1500 chars): how much of a stashed result to keep
      inline in the stub message.
    - ``enabled`` (default True): master switch — when False, capture AND
      offload are both skipped (the handler runs bare).
    """

    def __init__(
        self,
        store: EventStore | None = None,
        *,
        threshold: int = DEFAULT_THRESHOLD,
        preview: int = DEFAULT_PREVIEW,
        enabled: bool = True,
    ) -> None:
        self.store = store or shared_event_store()
        self.threshold = threshold
        self.preview = preview
        self.enabled = enabled

    # -- request introspection (works for langchain's ToolCallRequest or a
    # SimpleNamespace stand-in in tests) -----------------------------------

    def _tool_name(self, request: Any) -> str:
        tc = getattr(request, "tool_call", None) or {}
        if isinstance(tc, dict):
            name = tc.get("name")
            return str(name) if name is not None else "tool"
        return "tool"

    def _tool_args_summary(self, request: Any) -> str:
        tc = getattr(request, "tool_call", None) or {}
        args = tc.get("args", {}) if isinstance(tc, dict) else {}
        raw = json.dumps(args, ensure_ascii=False, default=str)
        return raw[:500] if len(raw) > 500 else raw

    def _thread_id(self, request: Any) -> str:
        state = getattr(request, "state", None) or {}
        if isinstance(state, dict):
            return state.get("configurable", {}).get("thread_id", "")
        return ""

    def _capture(
        self, *, tool: str, args_summary: str, thread_id: str,
        result: Any, elapsed: float,
    ) -> None:
        """Record the tool_call event with a 300-char output preview (skipped
        for retrieval tools + error-status results)."""
        success = True
        output_preview = ""
        if isinstance(result, ToolMessage):
            content = result.content if isinstance(result.content, str) else ""
            success = result.status != "error"
            if tool not in _RETRIEVAL_TOOLS:
                output_preview = content[:300] if content else ""
        self.store.capture(
            "tool_call",
            {
                "tool": tool,
                "args": args_summary,
                "success": success,
                "elapsed_s": round(elapsed, 3),
                "output_preview": output_preview,
            },
            thread_id=thread_id,
        )

    # -- sync ------------------------------------------------------------------

    def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled:
            return handler(request)

        tool = self._tool_name(request)
        args_summary = self._tool_args_summary(request)
        thread_id = self._thread_id(request)
        t0 = time.time()

        try:
            result = handler(request)
        except Exception as exc:
            elapsed = time.time() - t0
            self.store.capture(
                "error",
                {
                    "tool": tool,
                    "args": args_summary,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "elapsed_s": round(elapsed, 3),
                },
                thread_id=thread_id,
            )
            self.store.flush()
            raise

        elapsed = time.time() - t0
        # Capture the event from the ORIGINAL result (what the tool actually
        # returned) BEFORE offload swaps the message the model will see — the
        # event is a record of activity, the offload is a separate concern.
        self._capture(
            tool=tool, args_summary=args_summary, thread_id=thread_id,
            result=result, elapsed=elapsed,
        )
        result = _offload(
            result, self.store, tool,
            threshold=self.threshold, preview=self.preview, thread_id=thread_id,
        )
        self.store.flush()
        return result

    # -- async (the production path — the server/runner use ainvoke) ----------

    async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled:
            return await handler(request)

        tool = self._tool_name(request)
        args_summary = self._tool_args_summary(request)
        thread_id = self._thread_id(request)
        t0 = time.time()

        try:
            result = await handler(request)
        except Exception as exc:
            elapsed = time.time() - t0
            self.store.capture(
                "error",
                {
                    "tool": tool,
                    "args": args_summary,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "elapsed_s": round(elapsed, 3),
                },
                thread_id=thread_id,
            )
            self.store.flush()
            raise

        elapsed = time.time() - t0
        self._capture(
            tool=tool, args_summary=args_summary, thread_id=thread_id,
            result=result, elapsed=elapsed,
        )
        result = _offload(
            result, self.store, tool,
            threshold=self.threshold, preview=self.preview, thread_id=thread_id,
        )
        self.store.flush()
        return result
