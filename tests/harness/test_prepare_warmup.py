"""PrepareWarmupMiddleware — the serve-lane owner of the ``prepare()`` warmup seam.

Aegra (pux's prod Agent Protocol runtime — a langgraph-api/LangGraph-Platform
drop-in) owns the run loop itself: there is no pux entry point between "receive
run" and "invoke the graph", so the ``prepare()`` call that ``pux direct``
(main.py) and ``server.py`` make at their entry point is never reached under
Aegra — ``warmup_browser`` / ``warmup_webhook`` silently stop firing (the known
Aegra cutover delta). This middleware runs ``prepare()`` from the graph's own
``before_agent`` hook, which Aegra DOES drive.

This test proves the FOUR load-bearing properties (verify-or-die, not "should
work"):

1. **Fires once** — ``abefore_agent`` (the prod/async path) calls
   ``prepare(org, universal_warmup=...)`` exactly once, offloaded to a worker
   thread (``asyncio.to_thread``), so the event loop is not stalled by Docker
   I/O. The sync ``before_agent`` fires once too.
2. **Universal-warmup gating** — serve-class transports probe the run-completion
   endpoint (``universal_warmup=True``); ``direct`` does not (no serve up — the
   probe would retry ~15s before failing).
3. **Never breaks the run** — a ``prepare()`` exception is swallowed
   (warn-and-continue), matching ``prepare()``'s own contract; the hook returns
   ``None`` (no state mutation).
4. **Single owner / no test regression** — ``_build_prepare`` returns ``None``
   unless ``facts.prepare_warmup``, so the ``direct``/``server.py`` lanes (which
   call ``prepare()`` themselves) and tests (default ``RuntimeFacts()``) never
   double-fire or touch Docker. ``prepare`` is NOT in ``DEFAULT_SUPERVISOR`` —
   it is armed SOLELY by its gate (``_gate_prepare_warmup`` reads
   ``facts.prepare_warmup``), which is the single source of truth for when it
   mounts. Scoped SUPERVISOR-only.

No docker: ``prepare`` is monkeypatched to a recording stub.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from pux_harness.agent.stack import (
    DEFAULT_SUPERVISOR,
    RuntimeFacts,
    Scope,
    StackCtx,
    _specs_by_name,
    middleware_names,
)
from pux_harness.context.prepare_warmup import PrepareWarmupMiddleware


# ------------------------------------------------------------------------------------
# fixtures / helpers
# ------------------------------------------------------------------------------------

def _ctx(facts: RuntimeFacts) -> StackCtx:
    return StackCtx(
        org="coder",
        facts=facts,
        rubric_gate=None,
        exec_client=None,
        model_retry_cfg=None,
        tool_retry_cfg=None,
        emitted_tools_supervisor=[],
    )


@pytest.fixture
def record_prepare():
    """Return ``(calls, prepare_fn)`` — the middleware receives ``prepare_fn``
    via constructor injection (no lazy import, upstream-portable), so tests
    pass it explicitly instead of monkeypatching the source module.
    """
    calls: list[dict[str, Any]] = []

    def _fake_prepare(org, project_path=None, exec_client=None, universal_warmup=False):
        calls.append({"org": org, "universal_warmup": universal_warmup})
        return [{"name": "warmup_browser", "status": "ok", "error": None, "duration": 0.1}]

    return calls, _fake_prepare


# ------------------------------------------------------------------------------------
# 0. registry / defaults contract
# ------------------------------------------------------------------------------------

def test_prepare_registered_gate_driven_supervisor_only():
    assert "prepare" in middleware_names()
    # prepare is GATE-DRIVEN (not in DEFAULT_SUPERVISOR) — armed solely by its
    # gate (``_gate_prepare_warmup`` reads ``facts.prepare_warmup``). The gate
    # is the single source of truth for when this middleware mounts.
    assert "prepare" not in DEFAULT_SUPERVISOR, (
        "prepare is gate-driven (facts.prepare_warmup), not a default — "
        "a spec with a gate does not also appear in the default list"
    )
    spec = _specs_by_name()["prepare"]
    assert spec.gate is not None, "prepare must declare its gate"
    assert spec.scope == frozenset({Scope.SUPERVISOR}), (
        "prepare is a supervisor agent-start hook, not a per-subagent concern"
    )


# ------------------------------------------------------------------------------------
# 1. fires once
# ------------------------------------------------------------------------------------

async def test_abefore_agent_fires_prepare_once(record_prepare):
    calls, prepare_fn = record_prepare
    mw = PrepareWarmupMiddleware(org="coder", universal_warmup=True, prepare_fn=prepare_fn)
    ret = await mw.abefore_agent(SimpleNamespace(), SimpleNamespace())
    assert ret is None, "before_agent must not mutate state"
    assert len(calls) == 1, f"prepare called {len(calls)}x, want 1"
    assert calls[0]["org"] == "coder"
    assert calls[0]["universal_warmup"] is True


def test_before_agent_sync_fires_prepare_once(record_prepare):
    calls, prepare_fn = record_prepare
    mw = PrepareWarmupMiddleware(org="dre", universal_warmup=True, prepare_fn=prepare_fn)
    ret = mw.before_agent(SimpleNamespace(), SimpleNamespace())
    assert ret is None
    assert len(calls) == 1
    assert calls[0]["org"] == "dre"


async def test_abefore_agent_offloads_to_thread(monkeypatch, record_prepare):
    """The prod path must NOT run prepare on the event loop (would stall /events)."""
    calls, prepare_fn = record_prepare
    import pux_harness.context.prepare_warmup as pwm

    seen: list[bool] = []
    real_to_thread = pwm.asyncio.to_thread

    async def _capture_to_thread(fn, /, *a, **kw):
        seen.append(True)
        return await real_to_thread(fn, *a, **kw)

    monkeypatch.setattr(pwm.asyncio, "to_thread", _capture_to_thread)
    mw = PrepareWarmupMiddleware(org="coder", universal_warmup=True, prepare_fn=prepare_fn)
    await mw.abefore_agent(SimpleNamespace(), SimpleNamespace())
    assert seen == [True], "abefore_agent must offload via asyncio.to_thread"
    assert len(calls) == 1


# ------------------------------------------------------------------------------------
# 1b. fires through the REAL langgraph RunnableCallable wrap (the prod path)
# ------------------------------------------------------------------------------------
# langchain's agent factory wraps every before_agent hook in a langgraph
# RunnableCallable (factory.py:1526). That wrap passes ``state`` as the single
# positional arg and injects ``runtime`` as a KEYWORD arg detected by EXACT
# parameter name (``func_accepts`` in ``_internal/_runnable.py``). A direct
# positional call (the tests in §1) CANNOT catch a misnamed param; only this real
# wrap can. This is the test that would have caught ``_runtime`` → ``TypeError:
# missing 1 required positional argument`` (which surfaced live under Aegra). It
# is the verify-or-die contract test for the invocation arity.

async def test_abefore_agent_fires_via_real_runnable_callable_wrap(record_prepare):
    calls, prepare_fn = record_prepare
    from langgraph._internal._runnable import (  # mirror factory.py's import
        CONF,
        CONFIG_KEY_RUNTIME,
        RunnableCallable,
    )

    mw = PrepareWarmupMiddleware(org="coder", universal_warmup=True, prepare_fn=prepare_fn)
    # EXACTLY what factory.py does: RunnableCallable(sync_before_agent, async_before_agent, trace=False)
    node = RunnableCallable(mw.before_agent, mw.abefore_agent, trace=False)
    runtime = object()  # opaque — the hook ignores it; the framework injects it by name
    config = {CONF: {CONFIG_KEY_RUNTIME: runtime}}
    ret = await node.ainvoke({"messages": []}, config)
    assert ret is None, "before_agent must not mutate state"
    assert len(calls) == 1, "prepare must fire through the real wrap"


# ------------------------------------------------------------------------------------
# 2. universal_warmup gating by transport
# ------------------------------------------------------------------------------------

def test_build_prepare_serve_transport_universal_warmup_true():
    mw = _specs_by_name()["prepare"].build(
        _ctx(RuntimeFacts(prepare_warmup=True)), Scope.SUPERVISOR
    )
    assert isinstance(mw, PrepareWarmupMiddleware)
    assert mw.universal_warmup is True  # transport defaults to "serve"


def test_build_prepare_direct_transport_universal_warmup_false():
    mw = _specs_by_name()["prepare"].build(
        _ctx(RuntimeFacts(prepare_warmup=True, transport="direct")), Scope.SUPERVISOR
    )
    assert isinstance(mw, PrepareWarmupMiddleware)
    assert mw.universal_warmup is False, (
        "direct has no serve up — probing /events/health would retry ~15s"
    )


# ------------------------------------------------------------------------------------
# 3. never breaks the run (prepare raises -> swallowed)
# ------------------------------------------------------------------------------------

async def test_prepare_exception_is_swallowed():
    def _boom(*a, **kw):
        raise RuntimeError("docker daemon down")

    mw = PrepareWarmupMiddleware(org="coder", universal_warmup=True, prepare_fn=_boom)
    # Must NOT raise — prep failures are warn-and-continue, never block the run.
    ret = await mw.abefore_agent(SimpleNamespace(), SimpleNamespace())
    assert ret is None


# ------------------------------------------------------------------------------------
# 4. single owner / no test regression: prepare_warmup=False -> skip (None)
# ------------------------------------------------------------------------------------

def test_build_prepare_skipped_when_fact_false():
    """direct / server.py / tests leave prepare_warmup=False -> no middleware,
    so they never double-fire (they call prepare() themselves) and tests never
    touch Docker from a graph invoke."""
    got = _specs_by_name()["prepare"].build(_ctx(RuntimeFacts()), Scope.SUPERVISOR)
    assert got is None


def test_build_prepare_skipped_when_fact_false_even_for_serve_transport():
    # The gate is prepare_warmup, NOT transport — serve transport with the fact
    # off still skips (e.g. a test that sets transport=serve but not the flag).
    got = _specs_by_name()["prepare"].build(
        _ctx(RuntimeFacts(transport="serve")), Scope.SUPERVISOR
    )
    assert got is None
