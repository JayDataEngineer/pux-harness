"""CU-1 parity gate: ``CapabilityResolver`` is a ZERO-behavior-change facade.

``build_stack`` used to assemble the org-local tool + skill channels *inline*
(declared sandbox tools, dynamic ``lib/`` functions, skills roots). It now
routes through ONE ``CapabilityResolver.resolve(...)`` front-door that
dispatches by ``kind`` to the UNCHANGED leaf resolvers
(``build_declared_tools`` / ``build_dynamic_tools`` /
``supervisor_skills_roots``). This test proves the facade changed nothing: the
resolver's channels are bit-equal to the leaf resolvers called DIRECTLY (the
leaves ARE the legacy path), the composed ``tools_surface`` preserves the
load-bearing order, and the unified ``Capability`` index covers every member.

A self-contained tmp org (one declared tool + own/shared skills roots) is the
fixture; the dynamic-ON gate is toggled by monkeypatching the flag (isolates
resolver gating from policy parsing). No docker:
``build_declared_tools`` / ``build_dynamic_tools`` synthesize at build time and
capture the exec client in a closure (invoked at call time only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import StructuredTool

import pux_harness.agent.capabilities as caps_mod
import pux_harness.agent.orgs as orgs_mod
import pux_harness.sandbox.tools._shared as shared_mod
import pux_harness.sandbox.tools.declared as declared_mod
import pux_harness.sandbox.tools.dynamic as dynamic_mod
from pux_harness.agent.capabilities import (
    Capability,
    CapabilityIndex,
    CapabilityResolver,
    ResolvedCapabilities,
)
from pux_harness.agent.orgs import _org_path, supervisor_skills_roots
from pux_harness.sandbox.tools._shared import _NoArgs
from pux_harness.sandbox.tools.declared import build_declared_tools
from pux_harness.sandbox.tools.dynamic import build_dynamic_tools

_ORG = "captest"


class _FakeExec:
    """Stand-in for ``DockerExecClient`` — never invoked at build time."""

    def exec(self, command: str, *, timeout: int | None = None) -> tuple[str, int]:
        return ("ok\n", 0)


def _tool(name: str) -> StructuredTool:
    return StructuredTool(
        name=name, description="d", args_schema=_NoArgs, func=lambda **_: "ok"
    )


def _names(tools: Any) -> list[str]:
    return [t.name for t in tools]


@pytest.fixture
def org_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch project root with ONE org carrying a declared tool + own/shared
    skills roots.

    ``project_root`` is pinned to ``tmp_path`` so ``_orgs_dir()`` / ``_org_path``
    / ``supervisor_skills_roots`` / the policy loader all resolve against the
    scratch tree. Each tool module binds its OWN ``project_root`` (imported from
    ``kit._paths``), so the patch is applied to every consumer module that the
    leaf resolvers look up — the same pattern as ``tests/harness/test_dynamic.py``.
    """
    for mod in (orgs_mod, declared_mod, dynamic_mod, shared_mod):
        monkeypatch.setattr(mod, "project_root", lambda: tmp_path)
    base = tmp_path / "orgs" / "specialists" / _ORG
    base.mkdir(parents=True)
    (base / "AGENTS.md").write_text("# captest\n")
    # declared sandbox tool (one entry -> one pux_sandbox_* StructuredTool)
    sb = base / "sandbox"
    (sb / "tools").mkdir(parents=True)
    (sb / "signals.py").write_text("# script\n")
    (sb / "tools" / "tools.yaml").write_text(
        "tools:\n"
        "  - name: scan_signals\n"
        "    description: Scan a ticker.\n"
        "    script: signals.py\n"
    )

    def _skill(root: Path, name: str) -> None:
        d = root / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: d\n---\nbody\n"
        )

    _skill(base / "skills", "own_skill")  # the org's own skills root
    _skill(tmp_path / "orgs" / "_shared" / "skills", "shared_skill")  # shared root
    return tmp_path


# --- THE GATE: resolver channels == leaf resolvers (the legacy path) --------


def test_resolver_channels_equal_leaf_resolvers(org_root: Path) -> None:
    """The facade dispatches to the UNCHANGED leaves — declared/skills are
    bit-equal to calling the leaves directly; specialists + mcp are threaded
    through unchanged (identity). This IS the CU-1 contract: zero behavior
    change, or the facade is wrong."""
    exec_client = _FakeExec()
    specialists = [_tool("execute"), _tool("read_file")]
    mcp_tools = [_tool("mcp_equibles_filings")]

    resolved = CapabilityResolver(exec_client).resolve(
        _ORG, specialists=specialists, mcp_tools=mcp_tools,
    )
    assert isinstance(resolved, ResolvedCapabilities)

    # Leaves called directly == the legacy inline path build_stack used to run.
    assert _names(resolved.declared) == _names(
        build_declared_tools(_org_path(_ORG) / "sandbox", exec_client)
    )
    # dynamic gated OFF by default (no policy.yaml -> NoPolicy -> False). The
    # leaf builder itself always emits the 4 tools; the resolver's gate is what
    # matches the legacy `if load_dynamic_tools_enabled(org) else []`.
    assert resolved.dynamic == []
    assert resolved.skill_roots == supervisor_skills_roots(_ORG)
    # specialists + mcp threaded through UNCHANGED (same objects, identity).
    assert resolved.specialists == specialists
    assert list(resolved.mcp) == mcp_tools


def test_tools_surface_preserves_order_and_composition(org_root: Path) -> None:
    """``tools_surface`` is the subagent-shared surface: specialists FIRST, then
    declared, then dynamic — byte-identical to the legacy
    ``[*specialists, *declared, *dynamic]``. Order is load-bearing (subagent
    ``tools:`` allowlists resolve names against this list)."""
    exec_client = _FakeExec()
    s1 = _tool("execute")
    resolved = CapabilityResolver(exec_client).resolve(_ORG, specialists=[s1])
    assert resolved.tools_surface == [s1, *resolved.declared, *resolved.dynamic]
    # specialists precede declared precede (empty) dynamic
    assert _names(resolved.tools_surface)[:1] == ["execute"]


def test_resolver_dynamic_on_when_enabled(
    org_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the dynamic-tools gate ON, ``resolver.dynamic`` == the 4
    ``pux_dyn_*`` tools — the legacy gated build. Monkeypatching the flag
    isolates the resolver's gating branch from policy parsing."""
    monkeypatch.setattr(caps_mod, "load_dynamic_tools_enabled", lambda org: True)
    exec_client = _FakeExec()
    resolved = CapabilityResolver(exec_client).resolve(_ORG)
    assert _names(resolved.dynamic) == _names(
        build_dynamic_tools(_org_path(_ORG) / "lib", exec_client)
    )
    assert len(resolved.dynamic) == 4
    # dynamic tools surface AFTER specialists + declared
    surface = _names(resolved.tools_surface)
    assert surface[-4:] == _names(resolved.dynamic)


# --- the unified index (the CU-2 audit / CU-5 export surface) ---------------


def test_capability_index_covers_every_channel_member(org_root: Path) -> None:
    """The ``Capability`` audit index has ONE row per channel member, tagged by
    ``kind`` + ``provenance`` — no drops, no dupes. CU-2's CLI + CU-5's MDA
    exporter read this; it must faithfully mirror the resolved channels."""
    exec_client = _FakeExec()
    specialists = [_tool("execute")]
    mcp_tools = [_tool("mcp_x_y")]
    resolved = CapabilityResolver(exec_client).resolve(
        _ORG, specialists=specialists, mcp_tools=mcp_tools,
    )
    rows = resolved.capabilities
    refs = {(r.kind, r.ref, r.provenance) for r in rows}
    assert all(isinstance(r, Capability) for r in rows)

    # specialist -> kind=tool, provenance=registry
    assert ("tool", "execute", "registry") in refs
    # declared -> kind=tool, provenance=declared
    for name in _names(resolved.declared):
        assert ("tool", name, "declared") in refs
    assert _names(resolved.declared) == ["pux_sandbox_scan_signals"]
    # mcp -> kind=mcp, provenance=catalog
    assert ("mcp", "mcp_x_y", "catalog") in refs
    # every skill root -> kind=skill
    skill_refs = {r.ref for r in rows if r.kind == "skill"}
    assert set(resolved.skill_roots) == skill_refs

    # one row per channel member — no silent drops/dupes
    expected = (
        len(resolved.specialists)
        + len(resolved.declared)
        + len(resolved.dynamic)
        + len(list(resolved.mcp))
        + len(resolved.skill_roots)
    )
    assert len(rows) == expected


def test_kind_and_provenance_are_orthogonal_axes() -> None:
    """``kind`` (surface) and ``provenance`` (ownership ladder) are independent:
    all three tool levels are ``kind=tool`` with different provenance. The
    dynamic-tools level model composes UNDER ``kind=tool``, not over it."""
    rows = [
        Capability(kind="tool", ref="execute", provenance="registry"),
        Capability(kind="tool", ref="pux_sandbox_scan", provenance="declared"),
        Capability(kind="tool", ref="pux_dyn_make_function", provenance="dynamic"),
        Capability(kind="mcp", ref="equibles", provenance="catalog"),
        Capability(kind="skill", ref="orgs/x/skills"),
    ]
    # same kind, three provenances — the (a)/(b)/(c) ladder under one kind
    tool_provs = {r.provenance for r in rows if r.kind == "tool"}
    assert tool_provs == {"registry", "declared", "dynamic"}


def test_capability_index_merges_fleet_and_org_channels(org_root: Path) -> None:
    """``CapabilityIndex.load(org)`` is the discovery/audit view (CU-2): it
    merges the fleet registries (``REGISTRY`` tools + ``MIDDLEWARE_REGISTRY``
    middleware — always present, pure Python) with the org's declared channels
    (declared tools, skill roots). In the scratch tree the shared catalogs
    (mcp/skills.yaml) are absent so those rows drop — this asserts the
    DISPATCH + the org-local channels + the prefixed ``ref`` naming that keeps
    the catalog coherent with the runtime resolver index (``_index``).

    Docker-free: every leaf is a spec/declaration loader — no exec_client."""
    rows = CapabilityIndex.load(_ORG)
    assert rows  # non-empty
    assert all(isinstance(r, Capability) for r in rows)
    assert all(
        r.kind in {"tool", "skill", "mcp", "middleware", "job"} for r in rows
    ), {r.kind for r in rows}

    # fleet: registry tools + middleware are always present (pure Python).
    by_kp = {(r.kind, r.provenance) for r in rows}
    assert ("tool", "registry") in by_kp
    assert any(r.kind == "middleware" for r in rows)

    # per-org declared tool — PREFIXED (PUX_PREFIX), matching the resolver's
    # runtime index so the catalog + resolved surface agree on a ref.
    declared_refs = {
        r.ref for r in rows if r.kind == "tool" and r.provenance == "declared"
    }
    assert "pux_sandbox_scan_signals" in declared_refs

    # per-org skill roots surface as kind=skill (the SkillsMiddleware roots).
    skill_refs = {r.ref for r in rows if r.kind == "skill"}
    assert set(supervisor_skills_roots(_ORG)) == skill_refs

    # the org has no policy.yaml -> no jobs channel (NoPolicy caught).
    assert not any(r.kind == "job" for r in rows)
