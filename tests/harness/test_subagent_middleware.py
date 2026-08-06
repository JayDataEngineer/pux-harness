"""Per-subagent middleware — the frontmatter ``middleware: [name, ...]`` seam.

Owns the mechanism added so an agent's OWN frontmatter can mount registered
middleware on THAT subagent only (e.g. ``audit`` on a read-only investigator),
restoring the per-subagent auditor the pre-factory stack supported. Two layers:

* the BUILDER (``stack.make_subagent_middleware_builder``) — resolves the org's
  subagent baseline PLUS the per-agent names, validated + registry-ordered;
* the SEAM (``load_subagents`` reads ``spec["middleware"]`` + calls the builder).

Proven here against REAL middleware instances (``PUX_BROWSER_VISION=0`` drops
``browser_vision`` so the SUBAGENT baseline is just ``context`` — the build is
genuine, not mocked) and against a stubbed ``load_subagents`` for the wiring.
"""
from __future__ import annotations

import pytest

from pux_harness.agent import orgs, stack
from pux_harness.context.audit import AuditMiddleware
from pux_harness.context.middleware import ContextMiddleware


def _ctx() -> stack.StackCtx:
    """A minimal StackCtx — enough for the SUBAGENT specs (``context`` +
    ``audit``) to build. ``rubric_gate``/``exec_client`` are unused in this
    scope; ``context`` reads nothing off ctx, ``audit`` reads only ``org``."""
    return stack.StackCtx(
        org="test-org",
        facts=stack.RuntimeFacts(),
        rubric_gate=None,
        exec_client=None,
    )


@pytest.fixture(autouse=True)
def _no_browser_vision(monkeypatch):
    """``browser_vision`` builds ``None`` (skipped) when disabled — so the
    SUBAGENT baseline is the single ``context`` instance and the driver's
    multimodal path is never entered. Keeps the build real + hermetic."""
    monkeypatch.setenv("PUX_BROWSER_VISION", "0")


# --- the builder: baseline vs per-agent adds ------------------------------


def test_builder_baseline_is_context_only():
    """``builder([])`` (no per-agent extras) = the org's subagent baseline —
    context + read_file_vision + full_prefix_caching (browser_vision skipped
    via env). Byte-identical to the pre-per-agent shared list, so a subagent
    that declares no ``middleware:`` is unchanged."""
    builder = stack.make_subagent_middleware_builder(_ctx(), [], set())
    baseline = builder([])
    names = [type(m).__name__ for m in baseline]
    assert names == [
        "ContextMiddleware",
        "ReadFileVisionMiddleware",
        "FullPrefixCachingMiddleware",
    ], names


def test_builder_adds_audit_outermost():
    """``builder(["audit"])`` mounts ``AuditMiddleware`` on the subagent AND it
    lands FIRST (outermost) — registry order puts ``audit`` before ``context``,
    so the auditor wraps the context layer + the real tool (it observes the
    actual call, not a slice). The ``audit`` instance carries the SUBAGENT
    scope (so rows are attributable to the worker tier)."""
    builder = stack.make_subagent_middleware_builder(_ctx(), [], set())
    mw = builder(["audit"])
    names = [type(m).__name__ for m in mw]
    assert names == [
        "AuditMiddleware",
        "ContextMiddleware",
        "ReadFileVisionMiddleware",
        "FullPrefixCachingMiddleware",
    ], names
    assert isinstance(mw[0], AuditMiddleware)
    assert mw[0].scope == "subagent"


def test_builder_rubric_name_resolves_on_subagent_without_gate():
    """The scope-flip proof. ``rubric`` used to be supervisor-only, so
    ``builder(["rubric"])`` would RAISE here. Now the name passes the subagent
    scope check — and with NO gate armed ``_build_rubric`` returns None, so the
    subagent baseline is untouched (read-only-by-default). The scope only lifts
    the opt-in; the gate decides whether anything is built. A future re-lock of
    the scope (regression) makes this raise again."""
    builder = stack.make_subagent_middleware_builder(_ctx(), [], set())
    mw = builder(["rubric"])                # _ctx().rubric_gate is None
    assert [type(m).__name__ for m in mw] == [
        "ContextMiddleware",
        "ReadFileVisionMiddleware",
        "FullPrefixCachingMiddleware",
    ]


def test_builder_adds_rubric_on_subagent_when_gate_armed(monkeypatch):
    """The whole point: a subagent CAN mount the non-skippable
    ``RubricMiddleware`` verify-gate — no longer scope-locked to the supervisor.
    An armed gate + ``builder(["rubric"])`` builds a REAL ``RubricMiddleware``
    carrying the gate's ``max_iterations``; the grader runs INSIDE the worker's
    own graph, gathering evidence with the ``pux_grader_*`` tools. ``get_model``
    + ``build_grader_tools`` are stubbed (the grader client is lazy anyway) — we
    assert the wiring + the carried cap, not a live grading round."""
    from deepagents import RubricMiddleware
    from pux_harness.agent import profile

    ctx = stack.StackCtx(
        org="test-org",
        facts=stack.RuntimeFacts(),
        rubric_gate=profile.RubricGate(enabled=True, max_iterations=3),
        exec_client="EXEC",
    )
    monkeypatch.setattr(stack, "get_model", lambda **kw: "GRADER-MODEL")
    monkeypatch.setattr(stack, "build_grader_tools", lambda _ec: ["t0", "t1", "t2"])
    builder = stack.make_subagent_middleware_builder(ctx, [], set())

    mw = builder(["rubric"])
    # Registry order: ``context`` → ``rubric`` → ``read_file_vision`` →
    # ``full_prefix_caching``.
    names = [type(m).__name__ for m in mw]
    assert names == [
        "ContextMiddleware",
        "RubricMiddleware",
        "ReadFileVisionMiddleware",
        "FullPrefixCachingMiddleware",
    ], names
    assert isinstance(mw[1], RubricMiddleware)
    assert mw[1].max_iterations == 3


def test_builder_rejects_supervisor_only_name():
    """A supervisor-only middleware (``routing``) on a subagent fails LOUD —
    the resolver validates scope. This is the guard that keeps a user from
    arming a supervisor concern on a worker by typo."""
    builder = stack.make_subagent_middleware_builder(_ctx(), [], set())
    with pytest.raises(ValueError, match="routing"):
        builder(["routing"])


def test_builder_rejects_unknown_name():
    """An unregistered name fails loud (typo guard)."""
    builder = stack.make_subagent_middleware_builder(_ctx(), [], set())
    with pytest.raises(ValueError, match="bogus"):
        builder(["bogus"])


def test_builder_per_agent_add_stacks_on_org_baseline():
    """If the org already armed a subagent add (``middleware.subagent.add``),
    a per-agent frontmatter add is ADDITIONAL — both apply (add wins). The
    builder closes over the org's add list; the per-agent names extend it."""
    builder = stack.make_subagent_middleware_builder(_ctx(), ["audit"], set())
    mw = builder([])           # org baseline already arms audit
    assert any(isinstance(m, AuditMiddleware) for m in mw)
    mw2 = builder(["audit"])   # per-agent re-declares it — no double-mount
    assert sum(isinstance(m, AuditMiddleware) for m in mw2) == 1


# --- the seam: load_subagents applies per-agent middleware ----------------


def _spec(name: str, *, middleware=None) -> dict:
    spec = {"name": name, "description": name, "system_prompt": f"{name} body"}
    if middleware is not None:
        spec["middleware"] = middleware
    return spec


def _stub_build_sub(slug, spec, _tool_map, system_prompt, _org, *, middleware,
                    mcp_tools=(), declared_servers=frozenset()):
    """A minimal ``_build_sub`` stand-in: sets the SHARED ``middleware`` list
    (what the seam then overwrites per-agent) + the keys load_subagents reads."""
    return {
        "name": spec.get("name", slug),
        "description": spec.get("description", slug),
        "system_prompt": system_prompt,
        "middleware": list(middleware),
    }


def test_load_subagents_applies_per_agent_middleware(monkeypatch):
    """The seam: a subagent whose frontmatter carries ``middleware:`` gets the
    builder's output; one without keeps the shared list. ``load_subagents`` is
    driven with stubbed discovery/spec/_build_sub + a recording fake builder so
    the WIRING (read ``spec["middleware"]`` → call builder → set ``sub``) is
    isolated from the heavy prompt/model resolution."""
    monkeypatch.setattr(orgs, "discover_orgs", lambda: ["fake-org"])
    monkeypatch.setattr(orgs, "org_agent_slugs", lambda _org: ["auditor", "plain"])
    monkeypatch.setattr(orgs, "_org_declared_mcp_servers", lambda _org: frozenset())
    monkeypatch.setattr(orgs, "_load_agent_spec", lambda slug, _org: {
        "auditor": _spec("auditor", middleware=["audit"]),
        "plain": _spec("plain"),
    }[slug])
    monkeypatch.setattr(orgs, "_build_sub", _stub_build_sub)

    calls: list[list[str]] = []

    def fake_builder(names: list[str], *, rubric_text=None):
        calls.append(list(names))
        return [f"MW:{','.join(names)}"]

    shared = ["SHARED-BASELINE"]
    subs = orgs.load_subagents(
        "fake-org", [],
        subagent_middleware=shared,
        retrieval_tools=[],
        build_subagent_middleware=fake_builder,
    )
    by_name = {s["name"]: s for s in subs}
    # The auditor (declared ``middleware: [audit]``) got the builder's output;
    # the plain subagent kept the shared baseline unchanged.
    assert by_name["auditor"]["middleware"] == ["MW:audit"], by_name["auditor"]
    assert by_name["plain"]["middleware"] == ["SHARED-BASELINE"], by_name["plain"]
    # The builder fired ONCE — only for the auditor.
    assert calls == [["audit"]], calls


def test_load_subagents_accepts_scalar_middleware(monkeypatch):
    """A bare string (``middleware: audit``) is accepted as a one-name list —
    lenient, matching how ``tools``/``mcp`` accept a scalar."""
    monkeypatch.setattr(orgs, "discover_orgs", lambda: ["fake-org"])
    monkeypatch.setattr(orgs, "org_agent_slugs", lambda _org: ["auditor"])
    monkeypatch.setattr(orgs, "_org_declared_mcp_servers", lambda _org: frozenset())
    monkeypatch.setattr(orgs, "_load_agent_spec",
                        lambda _slug, _org: _spec("auditor", middleware="audit"))
    monkeypatch.setattr(orgs, "_build_sub", _stub_build_sub)

    seen: list[list[str]] = []
    orgs.load_subagents(
        "fake-org", [],
        subagent_middleware=["shared"],
        retrieval_tools=[],
        build_subagent_middleware=lambda names, *, rubric_text=None: (seen.append(list(names)), ["MW"])[1],
    )
    assert seen == [["audit"]], seen


def test_load_subagents_per_agent_middleware_requires_builder(monkeypatch):
    """A direct/test caller that arms per-agent middleware WITHOUT supplying a
    builder fails loud (the runtime factory always supplies one) — never a
    silent drop of the declared middleware."""
    monkeypatch.setattr(orgs, "discover_orgs", lambda: ["fake-org"])
    monkeypatch.setattr(orgs, "org_agent_slugs", lambda _org: ["auditor"])
    monkeypatch.setattr(orgs, "_org_declared_mcp_servers", lambda _org: frozenset())
    monkeypatch.setattr(orgs, "_load_agent_spec",
                        lambda _slug, _org: _spec("auditor", middleware=["audit"]))
    monkeypatch.setattr(orgs, "_build_sub", _stub_build_sub)

    with pytest.raises(ValueError, match="build_subagent_middleware"):
        orgs.load_subagents(
            "fake-org", [],
            subagent_middleware=["shared"],
            retrieval_tools=[],
        )  # no builder supplied
