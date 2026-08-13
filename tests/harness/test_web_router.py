"""``WebRouterMiddleware`` — the auto-firing websearch router.

When the latest USER turn clearly needs fresh external info, a cheap WORKER-model
web round fires (reusing the org's already-armed ``web_research`` MCP tools) and
injects a compact URL-cited brief, so the big CTO model never spends its OWN turn
calling a web tool. Inspired by "Supra-Router" (a pre-routing enrichment hop).

This test proves the load-bearing properties (verify-or-die, not "should work"):

0. **Registry / defaults** — ``web-router`` is registered, SUPERVISOR-only, LAST
   in the registry (innermost wrap), and NOT in ``DEFAULT_SUPERVISOR`` (opt-in).
1. **Heuristic hits** — recency / "look up" / version-number / future-event turns
   return a query; the common case (code, files, known facts, empty) returns None.
2. **Factory safety gate** — ``_build_web_router`` returns ``None`` when NO
   ``mcp__web_research__*`` tool is armed (the round is NEVER synthesized from
   nothing); builds the middleware when one IS armed.
3. **Injection on hit** — ``awrap_model_call`` prepends a ``HumanMessage`` carrying
   the ``AUTO_MARKER`` + query, and calls the handler exactly once.
4. **De-dupe + tool-result-turn skip** — a brief already present for the SAME
   query is not re-fetched; a turn with no HumanMessage (pure tool result) is
   passed through untouched.
5. **Enhancement, not gate** — a round failure is swallowed (logged + skipped);
   the handler still runs, the model still answers.

The MCP tools + worker model are stubbed (no live web / model calls here); the
live ``[Auto web context …]`` path is proven separately via ``pux direct``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from pux_harness.agent.profile import WebRouterConfig, load_web_router_config
from pux_harness.agent.stack import (
    DEFAULT_SUPERVISOR,
    MIDDLEWARE_REGISTRY,
    Scope,
    StackCtx,
    _specs_by_name,
    middleware_names,
)
from pux_harness.context import web_router
from pux_harness.context.web_router import (
    AUTO_MARKER,
    WebRouterMiddleware,
    heuristic_needs_web,
)


# ------------------------------------------------------------------------------------
# fixtures / helpers
# ------------------------------------------------------------------------------------

def _tool(name: str) -> SimpleNamespace:
    """A minimal stand-in for a BaseTool — the router only reads ``.name`` and
    stores the list; the round itself is patched in the wrap tests."""
    return SimpleNamespace(name=name)


def _ctx(mcp_tools, web_router_cfg=None) -> StackCtx:
    return StackCtx(
        org="orchestrator",
        facts=SimpleNamespace(),
        rubric_gate=None,
        sandbox=None,
        model_retry_cfg=None,
        tool_retry_cfg=None,
        emitted_tools_supervisor=[],
        mcp_tools=list(mcp_tools),
        web_router_cfg=web_router_cfg,
    )


@pytest.fixture
def patch_worker(monkeypatch):
    """``_build_web_router`` resolves a real worker model via ``get_model``;
    stub it so the factory test stays offline."""
    import pux_harness.agent.stack as stack_mod

    monkeypatch.setattr(stack_mod, "get_model", lambda *, role, org=None, model=None: object())
    return stack_mod


# ------------------------------------------------------------------------------------
# 0. registry / defaults contract
# ------------------------------------------------------------------------------------

def test_web_router_registered_supervisor_only_last_not_default():
    assert "web-router" in middleware_names()
    assert "web-router" not in DEFAULT_SUPERVISOR, (
        "web-router spends a worker round per firing turn — it must be opt-in"
    )
    spec = _specs_by_name()["web-router"]
    assert spec.scope == frozenset({Scope.SUPERVISOR})
    # Innermost wrap — emitted AFTER every wrapping spec, so it runs right before
    # the model. (prepare/interpreter after it are non-wrap: a before_agent hook
    # and a tool-injector.)
    names = [s.name for s in MIDDLEWARE_REGISTRY]
    assert names[-1] == "web-router"


def test_load_web_router_config_absent_block_is_default():
    # An org with no profile.yaml / no web_router block gets the free heuristic.
    cfg = load_web_router_config("coder")
    assert cfg == WebRouterConfig()
    assert cfg.model_router is False


def test_load_web_router_config_mapping_parses(monkeypatch):
    import pux_harness.agent.profile as profile_mod
    monkeypatch.setattr(
        profile_mod, "_resolved_profile_yaml",
        lambda org: {"web_router": {"model_router": True}},
    )
    cfg = load_web_router_config("orchestrator")
    assert cfg.model_router is True


def test_load_web_router_config_non_mapping_raises(monkeypatch):
    import pux_harness.agent.profile as profile_mod
    monkeypatch.setattr(profile_mod, "_resolved_profile_yaml", lambda org: {"web_router": True})
    with pytest.raises(TypeError):
        load_web_router_config("orchestrator")


# ------------------------------------------------------------------------------------
# 1. heuristic router
# ------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "What's the latest version of pytorch?",
    "Search the web for the latest news on the SEC filing.",
    "Current price of NVDA right now.",
    "When will they announce v3.2.1?",
    "Find the release date for the upcoming roadmap.",
])
def test_heuristic_hits_on_fresh_info_turns(text):
    assert heuristic_needs_web(text) is not None


@pytest.mark.parametrize("text", [
    "Fix the bug in parser.py.",          # code, no recency/lookup/version/future
    "Refactor the auth module.",          # code task
    "Explain how TCP three-way handshake works.",  # known fact
    "Explain the CAP theorem.",           # known fact (no version / recency signal)
    "",                                   # empty
    "   \n  ",                            # whitespace only
])
def test_heuristic_misses_on_common_case(text):
    assert heuristic_needs_web(text) is None


def test_heuristic_query_is_collapsed_and_capped():
    q = heuristic_needs_web("what is the   latest   version\nof  langgraph?")
    assert q == "what is the latest version of langgraph?"
    long_turn = "look up " + ("word " * 60)
    q2 = heuristic_needs_web(long_turn)
    assert q2 is not None and len(q2) <= 240


# ------------------------------------------------------------------------------------
# 2. factory safety gate (the critical "never synthesize from nothing" contract)
# ------------------------------------------------------------------------------------

def test_build_web_router_none_when_no_web_research_armed(patch_worker):
    # github + equibles armed, but NO web_research -> never mount.
    ctx = _ctx([_tool("mcp__github__create_issue"), _tool("mcp__equibles__quote")])
    assert _specs_by_name()["web-router"].build(ctx, Scope.SUPERVISOR) is None


def test_build_web_router_none_when_no_mcp_tools_at_all(patch_worker):
    # The offline test path (build_stack called with mcp_tools=()) -> skip.
    ctx = _ctx([])
    assert _specs_by_name()["web-router"].build(ctx, Scope.SUPERVISOR) is None


def test_build_web_router_mounts_filtering_only_web_research(patch_worker):
    ctx = _ctx([
        _tool("mcp__github__create_issue"),
        _tool("mcp__web_research__research"),
        _tool("mcp__web_research__search"),
        _tool("mcp__equibles__quote"),
    ])
    mw = _specs_by_name()["web-router"].build(ctx, Scope.SUPERVISOR)
    assert isinstance(mw, WebRouterMiddleware)
    assert [t.name for t in mw.web_tools] == [
        "mcp__web_research__research", "mcp__web_research__search",
    ]
    assert mw.use_model_router is False
    assert mw.org == "orchestrator"


def test_build_web_router_threads_model_router_flag(patch_worker):
    ctx = _ctx(
        [_tool("mcp__web_research__research")],
        web_router_cfg=WebRouterConfig(model_router=True),
    )
    mw = _specs_by_name()["web-router"].build(ctx, Scope.SUPERVISOR)
    assert isinstance(mw, WebRouterMiddleware)
    assert mw.use_model_router is True


# ------------------------------------------------------------------------------------
# 3. injection on hit (async prod path)
# ------------------------------------------------------------------------------------

@pytest.fixture
def recording_round(monkeypatch):
    """Patch the web round + worker so the wrap test is offline + observable."""
    calls: list[str] = []

    async def _fake_arun(query, web_tools, worker):
        calls.append(query)
        return f"BRIEF for {query}"

    monkeypatch.setattr(web_router, "arun_web_round", _fake_arun)
    return calls


async def _run_wrap(mw, messages):
    handler_calls = 0
    sentinel = object()

    async def handler(req):
        nonlocal handler_calls
        handler_calls += 1
        return sentinel

    request = SimpleNamespace(messages=messages)
    ret = await mw.awrap_model_call(request, handler)
    return ret, handler_calls, request


async def test_awrap_injects_brief_and_calls_handler_once(recording_round):
    mw = WebRouterMiddleware(web_tools=[_tool("mcp__web_research__research")], worker=object())
    human = HumanMessage("What's the latest version of langgraph?")
    ret, handler_calls, request = await _run_wrap(mw, [human])

    assert handler_calls == 1
    assert ret is not None
    assert len(recording_round) == 1, "exactly one web round fired"
    # The brief is prepended at position 0 as a HumanMessage carrying the marker.
    injected = request.messages[0]
    assert isinstance(injected, HumanMessage)
    assert AUTO_MARKER in injected.content
    assert "latest version of langgraph" in injected.content
    assert "BRIEF for" in injected.content


async def test_awrap_no_inject_on_miss(recording_round):
    mw = WebRouterMiddleware(web_tools=[_tool("mcp__web_research__research")], worker=object())
    ret, handler_calls, request = await _run_wrap(mw, [HumanMessage("Fix the parser bug.")])

    assert handler_calls == 1
    assert recording_round == [], "no round fired for a code/known-fact turn"
    # messages unchanged (still just the one human turn).
    assert len(request.messages) == 1


# ------------------------------------------------------------------------------------
# 4. de-dupe + tool-result-turn skip
# ------------------------------------------------------------------------------------

async def test_awrap_dedupes_same_query(recording_round):
    """A turn may re-enter the model across tool loops; a brief for the SAME
    query already in the list is not re-fetched."""
    query = "What's the latest version of langgraph?"
    mw = WebRouterMiddleware(web_tools=[_tool("mcp__web_research__research")], worker=object())
    pre_injected = HumanMessage(f'{AUTO_MARKER} "{query}"]:\n(prior brief)')
    messages = [pre_injected, HumanMessage(query)]
    ret, handler_calls, request = await _run_wrap(mw, messages)

    assert handler_calls == 1
    assert recording_round == [], "the round must NOT re-fire for a deduped query"
    assert request.messages[0] is pre_injected, "the existing brief was not replaced"


async def test_awrap_fires_for_human_turn_with_intervening_tool_message(recording_round):
    """``_latest_human_text`` looks past a trailing ToolMessage to the latest
    HumanMessage — so a genuinely-needs-web user turn still fires its round even
    when a tool result follows it in the list. (The re-entry guard is the de-dupe
    in the previous test, not a tool-message skip.)"""
    mw = WebRouterMiddleware(web_tools=[_tool("mcp__web_research__research")], worker=object())
    messages = [HumanMessage("latest news?"), ToolMessage("tool result blob", tool_call_id="x")]
    ret, handler_calls, request = await _run_wrap(mw, messages)

    assert handler_calls == 1
    assert len(recording_round) == 1, "the human turn still fires despite the trailing tool msg"


async def test_awrap_no_human_message_passes_through(recording_round):
    """No HumanMessage at all (e.g. an all-system seed) -> no round, passthrough."""
    mw = WebRouterMiddleware(web_tools=[_tool("mcp__web_research__research")], worker=object())
    ret, handler_calls, request = await _run_wrap(mw, [ToolMessage("blob", tool_call_id="x")])
    assert handler_calls == 1
    assert recording_round == []


# ------------------------------------------------------------------------------------
# 5. enhancement, not gate (round failure -> swallowed)
# ------------------------------------------------------------------------------------

async def test_awrap_round_failure_is_swallowed(monkeypatch):
    async def _boom(query, web_tools, worker):
        raise RuntimeError("web_research server down")

    monkeypatch.setattr(web_router, "arun_web_round", _boom)
    mw = WebRouterMiddleware(web_tools=[_tool("mcp__web_research__research")], worker=object())

    ret, handler_calls, request = await _run_wrap(mw, [HumanMessage("latest news?")])
    assert handler_calls == 1, "the model still answers when the round fails"
    assert len(request.messages) == 1, "no brief injected on failure"
