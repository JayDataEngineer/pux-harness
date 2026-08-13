"""``CodeInterpreterMiddleware`` (langchain-quickjs) — the dynamic-subagent happy path.

The CTO was doing every task itself because the static ``task`` tool can only
pick ONE subagent per call, so multi-unit work defaulted to solo execution. The
interpreter injects an ``eval`` JS-REPL tool + a ``task(...)`` global, letting a
strong orchestrator fan out (explorer recon -> ``Promise.all`` workers ->
synthesize) from a short dispatch script. The orchestrator thread stays LEAN —
a major token saver for a smart base model.

This test proves the load-bearing properties (verify-or-die, not "should work")
AFTER the strength-gate was deleted (org opt-in only):

1. **Registry contract** — ``interpreter`` is registered, SUPERVISOR-scoped, NOT
   in ``DEFAULT_SUPERVISOR``, and has ``gate=None`` (no auto-mount condition).
2. **Builds + exposes ``eval``** — ``_build_interpreter`` returns a real
   ``CodeInterpreterMiddleware`` whose ``.tools`` carries one ``eval`` tool.
3. **Opt-in only** — never auto-armed regardless of base-model strength; armed
   ONLY via explicit ``add: [interpreter]``. SUBAGENT scope -> never (workers
   don't dispatch).
4. **Toggle semantics** — ``add: [interpreter]`` mounts it; ``remove:
   [interpreter]`` drops it; add wins over remove. ``interpreter_hints`` is
   PAIRED (its gate arms iff ``interpreter`` is armed), so an ``add:
   [interpreter]`` also pulls hints in.

The strength-auto-arm that used to live in ``_resolve_toggles`` was deleted
because every default-tier org inherited it silently (shipped base is
strength:pro) while no org prompt references the eval tool — see
``docs/org-declarative-surface.md``. Fan-out orgs declare ``add: [interpreter]``
explicitly.
"""
from __future__ import annotations

from langchain_quickjs import CodeInterpreterMiddleware

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
        sandbox=None,
        model_retry_cfg=None,
        tool_retry_cfg=None,
        emitted_tools_supervisor=[],
    )


def _interpreter_built(mw_list: list) -> bool:
    return any(isinstance(m, CodeInterpreterMiddleware) for m in mw_list)


# ------------------------------------------------------------------------------------
# 0. registry / defaults contract
# ------------------------------------------------------------------------------------

def test_interpreter_registered_supervisor_only_opt_in_only():
    assert "interpreter" in middleware_names()
    assert "interpreter" not in DEFAULT_SUPERVISOR, (
        "interpreter is opt-in (add: [interpreter]); it has no gate and no default"
    )
    spec = _specs_by_name()["interpreter"]
    assert spec.scope == frozenset({Scope.SUPERVISOR}), (
        "the interpreter drives the ORCHESTRATOR's dynamic dispatch — workers never"
    )
    assert spec.gate is None, (
        "interpreter has NO auto-mount gate — org opt-in via `add: [interpreter]` "
        "is the ONLY path. The strength-auto-arm was deleted (see module docstring)."
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
# 2. opt-in only (no auto-arm regardless of base strength)
# ------------------------------------------------------------------------------------

def test_never_auto_mounts_without_explicit_add():
    """No add → no interpreter, no matter the base model. The strength gate was
    deleted; opt-in is the only path."""
    out = _resolve_toggles(_ctx(), Scope.SUPERVISOR, [], [], set())
    assert not _interpreter_built(out), (
        "interpreter requires explicit `add: [interpreter]` — never auto-mounted"
    )


def test_never_mounts_on_subagent_scope():
    """Workers never drive dynamic dispatch — the spec is SUPERVISOR-scoped, so
    the resolver's validator rejects a subagent ``add: [interpreter]`` loudly."""
    import pytest
    with pytest.raises(ValueError, match="not allowed in the subagent scope"):
        _resolve_toggles(_ctx(), Scope.SUBAGENT, [], ["interpreter"], set())


# ------------------------------------------------------------------------------------
# 3. toggle semantics (add/remove/add-wins-over-remove)
# ------------------------------------------------------------------------------------

def test_add_mounts_interpreter():
    """``middleware.supervisor.add: [interpreter]`` mounts it (the opt-in path
    for fan-out orgs: coder / orchestrator / deep-research-engine / game-studio)."""
    out = _resolve_toggles(_ctx(), Scope.SUPERVISOR, [], ["interpreter"], set())
    assert _interpreter_built(out), "explicit add -> interpreter mounted"


def test_remove_drops_explicit_add():
    """``middleware.supervisor.remove: [interpreter]`` drops it even when added
    (the exception escape hatch)."""
    out = _resolve_toggles(
        _ctx(), Scope.SUPERVISOR, [], ["interpreter"], {"interpreter"},
    )
    # add wins over remove (documented toggle semantics) — remove only drops
    # default/gated mounts, not explicit adds. So the interpreter IS built.
    assert _interpreter_built(out), (
        "add wins over remove — same-named add+remove -> add wins"
    )


def test_remove_drops_when_only_in_defaults():
    """``remove: [interpreter]`` drops it when armed only via default/gate (there
    is no default and no gate, so remove is a no-op here — but the semantics hold
    for any spec: remove drops everything except an explicit add)."""
    out = _resolve_toggles(_ctx(), Scope.SUPERVISOR, [], [], {"interpreter"})
    assert not _interpreter_built(out)


# ------------------------------------------------------------------------------------
# 4. pairing — interpreter_hints arms iff interpreter arms
# ------------------------------------------------------------------------------------

def test_add_interpreter_also_mounts_hints():
    """``interpreter_hints`` is PAIRED with ``interpreter``: its gate reads the
    partial on-set and arms iff ``interpreter`` is armed. An explicit ``add:
    [interpreter]`` pulls hints in too — no separate add needed."""
    from pux_harness.context.interpreter_hints import InterpreterHintsMiddleware
    out = _resolve_toggles(_ctx(), Scope.SUPERVISOR, [], ["interpreter"], set())
    assert _interpreter_built(out)
    assert any(isinstance(m, InterpreterHintsMiddleware) for m in out), (
        "interpreter_hints gates on 'interpreter in on-set' — adding interpreter "
        "must also mount hints"
    )


def test_no_interpreter_means_no_hints():
    """Without interpreter, hints never mount (its gate returns False)."""
    from pux_harness.context.interpreter_hints import InterpreterHintsMiddleware
    out = _resolve_toggles(_ctx(), Scope.SUPERVISOR, [], [], set())
    assert not any(isinstance(m, InterpreterHintsMiddleware) for m in out)


# Section 5 (dynamic-prompt assembly — ``_interpreter_mounted`` /
# ``_DYNAMIC_DISPATCH_SUFFIX`` / ``_append_dynamic_suffix``) RELOCATED to
# ``tests/harness/test_prompt_parts.py``: those symbols moved from ``stack.py``
# into ``prompt_parts`` (prompt content), so their tests live with the assembler.
