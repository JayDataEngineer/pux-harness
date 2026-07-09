"""Constructed-from-parts system-prompt assembly — the no-gap registry.

This is the prompt analogue of ``stack.MIDDLEWARE_REGISTRY``: an ORDERED tuple of
named ``PromptPartSpec`` parts joined by ``assemble_prompt``. It proves the load-
bearing properties (verify-or-die, not "should work"):

1. **The no-gap property** — a part whose ``build`` returns ``None`` is OMITTED
   (never an error) and leaves NO stray separator; registry position == output
   position; the scope filter keeps the two registries apart.
2. **The supervisor/subagent boundary** (the user's hard rule, made structural) —
   an assembled SUBAGENT prompt carries NONE of the root AGENTS.md, the harness
   addendum, the dynamic-dispatch notice, or the ask-user instruction, EVEN WHEN
   the same ``PromptCtx`` carries those supervisor fields. Subagents are
   SPECIALIZED for independent tasks.
3. **Conditional parts toggle byte-cleanly** — each gated suffix (ask_user /
   dynamic interpreter / org suffix) is the ONLY delta between its ON and OFF
   variants; OFF has no marker leak and no trailing blank.
4. **The relocated detector + suffix** (moved here from ``stack.py``) —
   ``_interpreter_mounted`` keys off the QUALIFIED class name (no quickjs import
   to ask the question); ``_DYNAMIC_DISPATCH_SUFFIX`` carries the dispatch
   strategy.
5. **The nuclear-replace is a PERMANENT failure** — ``base_system_prompt`` (the
   global-REPLACE that wiped the assembled prompt) is rejected at the offline
   tripwire (``validate_profile``) AND at per-agent frontmatter
   (``load_subagents``). (The runtime ``build_stack`` guard is exercised in the
   consumer repo's ``tests/agent/test_stack.py`` alongside the other
   ``build_stack`` integration tests.)

No docker, no network, no real org tree: every org-resolving seam is monkey-
patched, so this runs as part of the harness's own (org-agnostic) suite.
"""
from __future__ import annotations

import pytest
from langchain_quickjs import CodeInterpreterMiddleware

from deepagents import HarnessProfileConfig
from pux_harness.agent import orgs, profile
from pux_harness.agent.hitl import ASK_USER_PROMPT_SUFFIX
from pux_harness.agent.prompt_parts import (
    _DYNAMIC_DISPATCH_SUFFIX,
    _interpreter_mounted,
    PromptCtx,
    PromptPartSpec,
    PromptScope,
    SUBAGENT_PROMPT_PARTS,
    SUPERVISOR_PROMPT_PARTS,
    assemble_prompt,
)

# Markers that appear ONLY in the supervisor prompt. A subagent prompt must
# contain NONE of them — the boundary test asserts each is absent.
_ADDENDUM_HEAD = "## Harness addendum"
_DYNAMIC_HEAD = "## Dynamic dispatch"
_ASK_USER_HEAD = "END your turn immediately"  # inside ASK_USER_PROMPT_SUFFIX
BASE = "ROOT+OVERLAY BODY"  # stands in for orgs.build_system_prompt(org)


# ------------------------------------------------------------------------------------
# 1. assemble_prompt — the no-gap property
# ------------------------------------------------------------------------------------

def test_none_build_is_skipped_no_stray_separator():
    """A part whose build returns None is OMITTED — never an error — and leaves
    no stray ``\\n\\n``. The no-gap property: every part is registered with an
    explicit condition (None == condition off == excluded)."""
    parts = (
        PromptPartSpec("a", frozenset({PromptScope.SUPERVISOR}), lambda c: "A"),
        PromptPartSpec("b", frozenset({PromptScope.SUPERVISOR}), lambda c: None),
        PromptPartSpec("c", frozenset({PromptScope.SUPERVISOR}), lambda c: "C"),
    )
    assert assemble_prompt(parts, PromptCtx(), PromptScope.SUPERVISOR) == "A\n\nC"


def test_registry_position_is_output_position():
    """Position in the registry == position in the joined output (deterministic)."""
    parts = (
        PromptPartSpec("z", frozenset({PromptScope.SUPERVISOR}), lambda c: "Z"),
        PromptPartSpec("a", frozenset({PromptScope.SUPERVISOR}), lambda c: "A"),
    )
    assert assemble_prompt(parts, PromptCtx(), PromptScope.SUPERVISOR) == "Z\n\nA"


def test_scope_filter_excludes_the_other_scope():
    """Asking for one scope emits only that scope's parts — a part scoped to the
    other registry is invisible (the registries are disjoint by construction)."""
    parts = (
        PromptPartSpec("sup", frozenset({PromptScope.SUPERVISOR}), lambda c: "SUP"),
        PromptPartSpec("sub", frozenset({PromptScope.SUBAGENT}), lambda c: "SUB"),
    )
    assert assemble_prompt(parts, PromptCtx(), PromptScope.SUPERVISOR) == "SUP"
    assert assemble_prompt(parts, PromptCtx(), PromptScope.SUBAGENT) == "SUB"


def test_empty_when_nothing_matches():
    """No matching non-None parts -> empty string (no crash)."""
    parts = (PromptPartSpec("x", frozenset({PromptScope.SUBAGENT}), lambda c: "X"),)
    assert assemble_prompt(parts, PromptCtx(), PromptScope.SUPERVISOR) == ""


# ------------------------------------------------------------------------------------
# 2. the registries are well-formed + disjoint
# ------------------------------------------------------------------------------------

def test_supervisor_registry_order():
    assert [p.name for p in SUPERVISOR_PROMPT_PARTS] == [
        "agents_md_core",
        "org_system_prompt_suffix",
        "ask_user_suffix",
        "dynamic_dispatch_suffix",
    ]


def test_subagent_registry_order():
    assert [p.name for p in SUBAGENT_PROMPT_PARTS] == [
        "agent_body",
        "org_system_prompt_suffix",
        "agent_system_prompt_suffix",
    ]


def test_no_supervisor_only_part_is_in_the_subagent_registry():
    """The boundary, encoded in the registries: the supervisor-only parts
    (agents_md_core / ask_user_suffix / dynamic_dispatch_suffix) are NOWHERE in
    the subagent registry. The ONLY name the two may share is the org-wide
    suffix (deliberately applied to both)."""
    sub_names = {p.name for p in SUBAGENT_PROMPT_PARTS}
    for forbidden in ("agents_md_core", "ask_user_suffix", "dynamic_dispatch_suffix"):
        assert forbidden not in sub_names
    shared = {p.name for p in SUPERVISOR_PROMPT_PARTS} & sub_names
    assert shared <= {"org_system_prompt_suffix"}


# ------------------------------------------------------------------------------------
# 3. the boundary — a subagent prompt has ZERO supervisor content
# ------------------------------------------------------------------------------------

def test_subagent_prompt_lacks_every_supervisor_marker():
    """The user's hard rule, made structural: assembling a SUBAGENT prompt never
    emits root AGENTS.md, the harness addendum, the dynamic-dispatch notice, or
    the ask-user instruction — EVEN WHEN the same ctx carries those supervisor
    fields (they must be ignored for a subagent). The output is exactly the
    agent's own body + the org suffix + its per-agent suffix."""
    ctx = PromptCtx(
        agent_body="You are a specialist. Do the delegated task.",
        system_prompt_suffix="Org-wide suffix line.",
        agent_system_prompt_suffix="Per-agent suffix line.",
        # supervisor-only fields populated — must be IGNORED under SUBAGENT scope:
        agents_md_base=BASE,
        ask_user_active=True,
        interpreter_mounted=True,
    )
    prompt = assemble_prompt(SUBAGENT_PROMPT_PARTS, ctx, PromptScope.SUBAGENT)
    assert prompt == (
        "You are a specialist. Do the delegated task.\n\n"
        "Org-wide suffix line.\n\nPer-agent suffix line."
    )
    for marker in (BASE, _ADDENDUM_HEAD, _DYNAMIC_HEAD, _ASK_USER_HEAD):
        assert marker not in prompt, f"subagent leaked supervisor marker: {marker!r}"


def test_subagent_prompt_with_no_suffixes_is_exactly_its_body():
    """No suffixes -> the subagent prompt is EXACTLY its body (no stray seam)."""
    ctx = PromptCtx(agent_body="BODY")
    assert assemble_prompt(SUBAGENT_PROMPT_PARTS, ctx, PromptScope.SUBAGENT) == "BODY"


# ------------------------------------------------------------------------------------
# 4. conditional parts toggle byte-cleanly (the gap-free remainder)
# ------------------------------------------------------------------------------------

def test_dynamic_suffix_toggles_cleanly():
    """interpreter_mounted is the ONLY difference: ON == OFF + '\\n\\n' + suffix.
    OFF leaks no dynamic marker and has no trailing blank."""
    on = assemble_prompt(
        SUPERVISOR_PROMPT_PARTS,
        PromptCtx(agents_md_base=BASE, interpreter_mounted=True),
        PromptScope.SUPERVISOR,
    )
    off = assemble_prompt(
        SUPERVISOR_PROMPT_PARTS,
        PromptCtx(agents_md_base=BASE, interpreter_mounted=False),
        PromptScope.SUPERVISOR,
    )
    assert on == off + "\n\n" + _DYNAMIC_DISPATCH_SUFFIX
    assert _DYNAMIC_HEAD in on
    assert _DYNAMIC_HEAD not in off
    assert not off.endswith("\n\n")


def test_ask_user_suffix_toggles_cleanly():
    """ask_user_active is the ONLY difference."""
    on = assemble_prompt(
        SUPERVISOR_PROMPT_PARTS,
        PromptCtx(agents_md_base=BASE, ask_user_active=True),
        PromptScope.SUPERVISOR,
    )
    off = assemble_prompt(
        SUPERVISOR_PROMPT_PARTS,
        PromptCtx(agents_md_base=BASE, ask_user_active=False),
        PromptScope.SUPERVISOR,
    )
    assert on == off + "\n\n" + ASK_USER_PROMPT_SUFFIX
    assert _ASK_USER_HEAD in on
    assert _ASK_USER_HEAD not in off
    assert not off.endswith("\n\n")


def test_org_suffix_toggles_cleanly():
    on = assemble_prompt(
        SUPERVISOR_PROMPT_PARTS,
        PromptCtx(agents_md_base=BASE, system_prompt_suffix="SUFFIX"),
        PromptScope.SUPERVISOR,
    )
    off = assemble_prompt(
        SUPERVISOR_PROMPT_PARTS,
        PromptCtx(agents_md_base=BASE),
        PromptScope.SUPERVISOR,
    )
    assert on == off + "\n\nSUFFIX"


def test_all_conditions_off_is_exactly_the_core():
    """Every conditional OFF -> the supervisor prompt is agents_md_core only
    (root + overlay + the folded addendum). No seam drift, no stray marker."""
    from pux_harness.agent.prompt_parts import _ADDENDUM

    prompt = assemble_prompt(
        SUPERVISOR_PROMPT_PARTS,
        PromptCtx(agents_md_base=BASE),
        PromptScope.SUPERVISOR,
    )
    assert prompt == f"{BASE}{_ADDENDUM}"


# ------------------------------------------------------------------------------------
# 5. relocated — the detector + the dynamic-dispatch suffix content
# ------------------------------------------------------------------------------------

def _interp() -> CodeInterpreterMiddleware:
    # The same shape ``stack._build_interpreter`` constructs; the detector keys
    # off the QUALIFIED type name, so any real instance trips it.
    return CodeInterpreterMiddleware(
        subagents=True, ptc=["glob", "grep", "ls", "read_file"],
    )


def test_interpreter_mounted_detects_built_middleware():
    """``_interpreter_mounted`` sees a real built interpreter; misses a non-
    interpreter object and an empty list. Detected by QUALIFIED class name so the
    check needs NO langchain_quickjs import at call sites that mount nothing (a
    perf gate, not just a mount gate)."""
    interp = _interp()
    other = object()
    assert _interpreter_mounted([interp]) is True
    assert _interpreter_mounted([other]) is False
    assert _interpreter_mounted([]) is False
    assert _interpreter_mounted([other, interp]) is True


def test_dynamic_dispatch_suffix_carries_the_strategy():
    """The notice tells a strong orchestrator to PREFER one dispatch script over
    the static task flow, fan out with Promise.all, keep its thread lean, and
    defer the JS API to the eval tool's own description."""
    assert _DYNAMIC_DISPATCH_SUFFIX.strip()
    assert "eval" in _DYNAMIC_DISPATCH_SUFFIX
    assert "task({subagentType" in _DYNAMIC_DISPATCH_SUFFIX
    assert "Promise.all" in _DYNAMIC_DISPATCH_SUFFIX
    assert "PREFER" in _DYNAMIC_DISPATCH_SUFFIX


# ------------------------------------------------------------------------------------
# 6. base_system_prompt nuclear-replace is a PERMANENT failure (offline sites)
# ------------------------------------------------------------------------------------
# The runtime ``build_stack`` guard is exercised in tests/agent/test_stack.py.

def test_validate_profile_rejects_base_system_prompt(monkeypatch):
    """The offline tripwire fires for every org in ``--check-contract``: a
    ``base_system_prompt`` on the profile is a permanent contract failure."""
    monkeypatch.setattr(
        profile, "load_profile",
        lambda org: HarnessProfileConfig(base_system_prompt="NUKE"),
    )
    with pytest.raises(ValueError, match="base_system_prompt"):
        profile.validate_profile("general")


def test_load_subagents_rejects_per_agent_base_system_prompt(monkeypatch):
    """A stray ``base_system_prompt:`` in an agent's frontmatter must FAIL, not
    silently drop (a silent drop is a gap). Every org-resolving seam is stubbed
    so this is org-agnostic (no real org tree read)."""
    monkeypatch.setattr(orgs, "discover_orgs", lambda: ["general"])
    monkeypatch.setattr(orgs, "org_agent_slugs", lambda org: ["probe"])
    monkeypatch.setattr(
        orgs, "_load_agent_spec",
        lambda slug, org: {
            "name": "probe",
            "description": "d",
            "system_prompt": "BODY",
            "base_system_prompt": "NUKE",
        },
    )
    monkeypatch.setattr(orgs, "_org_declared_mcp_servers", lambda org: frozenset())
    monkeypatch.setattr(
        orgs, "_build_sub",
        lambda slug, spec, tool_map, system_prompt, org, **k: {"name": "probe"},
    )
    with pytest.raises(ValueError, match="base_system_prompt"):
        orgs.load_subagents(
            "general", [], profile=None,
            subagent_middleware=[], retrieval_tools=[],
        )
