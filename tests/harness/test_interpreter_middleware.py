"""``CodeInterpreterMiddleware`` (langchain-quickjs) — the dynamic-subagent happy path.

The CTO was doing every task itself because the static ``task`` tool can only
pick ONE subagent per call, so multi-unit work defaulted to solo execution. The
interpreter injects an ``eval`` JS-REPL tool + a ``task(...)`` global, letting a
strong orchestrator fan out (explorer recon -> ``Promise.all`` workers ->
synthesize) from a short dispatch script. The orchestrator thread stays LEAN —
a major token saver for a smart base model.

This test proves the FOUR load-bearing properties (verify-or-die, not "should
work"):

1. **Registry contract** — ``interpreter`` is registered, SUPERVISOR-scoped, and
   NOT in ``DEFAULT_SUPERVISOR`` (the strength gate arms it, not a default).
2. **Builds + exposes ``eval``** — ``_build_interpreter`` returns a real
   ``CodeInterpreterMiddleware`` whose ``.tools`` carries one ``eval`` tool.
3. **Strength gate** — armed in ``_resolve_toggles`` iff the resolved base model
   is ``strength: pro`` (driver_strong_orchestrator). Weak/unknown -> OFF (no
   token waste); SUBAGENT scope -> never (workers don't dispatch).
4. **Override** — ``middleware.supervisor.add/remove: [interpreter]`` wins either
   way (add beats the gate, remove beats a pro base).

No docker, no network: ``driver_strong_orchestrator`` is monkeypatched, and
``_resolve_toggles`` is driven with ``default_names=[]`` so only the interpreter
build fires (context/routing/browser_vision never construct).
"""
from __future__ import annotations

from langchain_quickjs import CodeInterpreterMiddleware

from pux_harness.agent import stack
from pux_harness.agent.stack import (
    DEFAULT_SUPERVISOR,
    RuntimeFacts,
    Scope,
    StackCtx,
    _resolve_toggles,
    _specs_by_name,
    middleware_names,
)


# ------------------------------------------------------------------------------------
# fixtures / helpers
# ------------------------------------------------------------------------------------

def _ctx() -> StackCtx:
    return StackCtx(
        org="general",
        facts=RuntimeFacts(),
        rubric_gate=None,
        exec_client=None,
        model_retry_cfg=None,
        tool_retry_cfg=None,
        emitted_tools_supervisor=[],
    )


def _strong(monkeypatch, value: bool) -> None:
    """Control the strength seam without touching the model registry."""
    monkeypatch.setattr(
        stack, "driver_strong_orchestrator",
        lambda *, role="base", org=None: value,
    )


def _interpreter_built(mw_list: list) -> bool:
    return any(isinstance(m, CodeInterpreterMiddleware) for m in mw_list)


# ------------------------------------------------------------------------------------
# 0. registry / defaults contract
# ------------------------------------------------------------------------------------

def test_interpreter_registered_supervisor_only_not_default():
    assert "interpreter" in middleware_names()
    assert "interpreter" not in DEFAULT_SUPERVISOR, (
        "interpreter is strength-gated (armed in _resolve_toggles), not a default"
    )
    spec = _specs_by_name()["interpreter"]
    assert spec.scope == frozenset({Scope.SUPERVISOR}), (
        "the interpreter drives the ORCHESTRATOR's dynamic dispatch — workers never"
    )


# ------------------------------------------------------------------------------------
# 1. builds + exposes the eval tool
# ------------------------------------------------------------------------------------

def test_build_interpreter_returns_eval_exposing_middleware():
    mw = _specs_by_name()["interpreter"].build(_ctx(), Scope.SUPERVISOR)
    assert isinstance(mw, CodeInterpreterMiddleware)
    assert [t.name for t in mw.tools] == ["eval"], (
        "the interpreter surfaces exactly one tool — the `eval` JS-REPL — which the "
        "orchestrator detects at runtime to know it is interpreter-enabled"
    )


# ------------------------------------------------------------------------------------
# 2. strength gate
# ------------------------------------------------------------------------------------

def test_gate_mounts_for_strong_base(monkeypatch):
    """A pro base (glm-5.2) auto-gets the happy path — no per-org config."""
    _strong(monkeypatch, True)
    out = _resolve_toggles(_ctx(), Scope.SUPERVISOR, [], [], set())
    assert _interpreter_built(out), "pro base -> interpreter auto-mounted"


def test_gate_skips_for_weak_base(monkeypatch):
    """A flash/unknown base (fast tier, mimo) gets NO interpreter — it would
    waste tokens on a tool the model can't drive reliably."""
    _strong(monkeypatch, False)
    out = _resolve_toggles(_ctx(), Scope.SUPERVISOR, [], [], set())
    assert out == [], "weak base -> no interpreter, byte-identical to today"


def test_gate_never_mounts_on_subagent_scope(monkeypatch):
    """Workers never drive dynamic dispatch — the gate is SUPERVISOR-scoped."""
    _strong(monkeypatch, True)
    out = _resolve_toggles(_ctx(), Scope.SUBAGENT, [], [], set())
    assert not _interpreter_built(out)


# ------------------------------------------------------------------------------------
# 3. override (add/remove win over the gate)
# ------------------------------------------------------------------------------------

def test_remove_overrides_a_strong_base(monkeypatch):
    """``middleware.supervisor.remove: [interpreter]`` forces it OFF even on a
    pro base (the exception escape hatch)."""
    _strong(monkeypatch, True)
    out = _resolve_toggles(_ctx(), Scope.SUPERVISOR, [], [], {"interpreter"})
    assert not _interpreter_built(out), "remove must beat the strength gate"


def test_add_overrides_a_weak_base(monkeypatch):
    """``middleware.supervisor.add: [interpreter]`` forces it ON even on a flash
    base — the explicit opt-in for a strong model the registry under-classes."""
    _strong(monkeypatch, False)
    out = _resolve_toggles(_ctx(), Scope.SUPERVISOR, [], ["interpreter"], set())
    assert _interpreter_built(out), "add must beat the strength gate"


def test_add_wins_over_remove(monkeypatch):
    """Same-named add + remove -> add wins (the documented toggle semantics)."""
    _strong(monkeypatch, False)
    out = _resolve_toggles(
        _ctx(), Scope.SUPERVISOR, [], ["interpreter"], {"interpreter"},
    )
    assert _interpreter_built(out)

# Section 4 (dynamic-prompt assembly — ``_interpreter_mounted`` /
# ``_DYNAMIC_DISPATCH_SUFFIX`` / ``_append_dynamic_suffix``) RELOCATED to
# ``tests/harness/test_prompt_parts.py``: those symbols moved from ``stack.py``
# into ``prompt_parts`` (prompt content), so their tests live with the assembler.
