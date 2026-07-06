"""Phase 5 — org-level inheritance (``org.yaml extends:``) at the KIT layer.

The portable, project_root-parameterized core (``pux_harness.kit.loaders``): NO
Docker, NO model registry, NO pux shim. These build a real temp project root
and exercise org-extends resolution end to end — the single-hop reader, the
root→child chain (with cycle / unresolvable failures), the cycle-safe runtime
fallback, the chain-inherited roster union, the chain-aware agent search dirs
(an inherited slug resolves through a parent's ``agents/``; a child specializes
it via its own), and the AGENTS.md overlay concat.

The sibling orchestrator file (``tests/test_org_extends.py``) owns the pux-shim
delegates, the contract rules, and the profile deep-merge; this file owns the
kit-level mechanism both share. Mirrors ``test_agent_extends.py``'s split.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pux_harness.kit.loaders import (
    _agent_search_dirs,
    _chain_overlay,
    _load_agent_spec,
    _own_org_agent_slugs,
    _resolved_org_chain,
    build_system_prompt,
    org_agent_slugs,
    org_extends,
    org_extends_chain,
)


# --- tree helpers ----------------------------------------------------------

def _make_org(
    root: Path, name: str, *, body: str,
    agents: list[str] | None = None, extends: str | None = None,
) -> Path:
    """Write ``orgs/<name>/AGENTS.md`` + (when there's a roster or an extends)
    ``org.yaml``. The roster + extends lines both live in ``org.yaml``."""
    d = root / "orgs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "AGENTS.md").write_text(body)
    lines: list[str] = []
    if agents is not None:
        lines.append(f"agents: [{', '.join(agents)}]")
    if extends is not None:
        lines.append(f"extends: {extends}")
    if lines:
        (d / "org.yaml").write_text("\n".join(lines) + "\n")
    return d


def _make_agent(
    root: Path, slug: str, org: str, *, body: str = "BODY",
    tools: list[str] | None = None,
) -> Path:
    """Write ``orgs/<org>/agents/<slug>.md`` (frontmatter + body)."""
    agents_dir = root / "orgs" / org / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---", f'name: "{slug}"', f'description: "{slug} agent"']
    if tools:
        lines.append(f"tools: [{', '.join(tools)}]")
    lines += ["---", "", body]
    path = agents_dir / f"{slug}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# --- org_extends (single-hop RAW reader) -----------------------------------

def test_org_extends_none_when_no_extends(tmp_path: Path) -> None:
    _make_org(tmp_path, "solo", body="# Solo\n", agents=["a"])
    assert org_extends("solo", tmp_path) is None


def test_org_extends_reads_parent(tmp_path: Path) -> None:
    _make_org(tmp_path, "base", body="# Base\n")
    _make_org(tmp_path, "child", body="# Child\n", extends="base")
    assert org_extends("child", tmp_path) == "base"


def test_org_extends_none_when_no_org_yaml(tmp_path: Path) -> None:
    # A CTO-only org ships AGENTS.md but no org.yaml.
    _make_org(tmp_path, "ctoonly", body="# CTO\n")
    assert org_extends("ctoonly", tmp_path) is None


def test_org_extends_none_for_non_string(tmp_path: Path) -> None:
    # A garbage ``extends: []`` is treated as no parent, not crashed on.
    d = tmp_path / "orgs" / "bad"
    d.mkdir(parents=True)
    (d / "AGENTS.md").write_text("# Bad\n")
    (d / "org.yaml").write_text("extends: []\n")
    assert org_extends("bad", tmp_path) is None


# --- org_extends_chain (root->child; raises on cycle / unresolvable) -------

def test_chain_simple_root_to_child(tmp_path: Path) -> None:
    _make_org(tmp_path, "base", body="# Base\n")
    _make_org(tmp_path, "child", body="# Child\n", extends="base")
    assert org_extends_chain("child", tmp_path) == ["base", "child"]


def test_chain_multilevel(tmp_path: Path) -> None:
    _make_org(tmp_path, "gp", body="# GP\n")
    _make_org(tmp_path, "mid", body="# Mid\n", extends="gp")
    _make_org(tmp_path, "kid", body="# Kid\n", extends="mid")
    assert org_extends_chain("kid", tmp_path) == ["gp", "mid", "kid"]


def test_chain_no_extends_is_self(tmp_path: Path) -> None:
    _make_org(tmp_path, "solo", body="# Solo\n")
    assert org_extends_chain("solo", tmp_path) == ["solo"]


def test_chain_cycle_raises(tmp_path: Path) -> None:
    _make_org(tmp_path, "a", body="# A\n", extends="b")
    _make_org(tmp_path, "b", body="# B\n", extends="a")
    with pytest.raises(ValueError, match="extends cycle"):
        org_extends_chain("a", tmp_path)


def test_chain_self_cycle_raises(tmp_path: Path) -> None:
    _make_org(tmp_path, "loopy", body="# L\n", extends="loopy")
    with pytest.raises(ValueError, match="extends cycle"):
        org_extends_chain("loopy", tmp_path)


def test_chain_unresolvable_parent_raises(tmp_path: Path) -> None:
    _make_org(tmp_path, "orphan", body="# O\n", extends="ghost")
    with pytest.raises(FileNotFoundError, match="no such org"):
        org_extends_chain("orphan", tmp_path)


def test_chain_parent_without_agents_md_raises(tmp_path: Path) -> None:
    # A dir that is not a valid base org (no AGENTS.md) is rejected — the chain
    # walker requires every ancestor to be a real, AGENTS.md-bearing org.
    (tmp_path / "orgs" / "naked").mkdir(parents=True)
    _make_org(tmp_path, "child", body="# C\n", extends="naked")
    with pytest.raises(FileNotFoundError, match="no AGENTS.md"):
        org_extends_chain("child", tmp_path)


# --- _resolved_org_chain (cycle-safe runtime fallback) ---------------------

def test_resolved_chain_falls_back_on_cycle(tmp_path: Path) -> None:
    _make_org(tmp_path, "a", body="# A\n", extends="b")
    _make_org(tmp_path, "b", body="# B\n", extends="a")
    # Does NOT raise — falls back to [name] so runtime loaders never crash.
    assert _resolved_org_chain("a", tmp_path) == ["a"]


def test_resolved_chain_falls_back_on_unresolvable(tmp_path: Path) -> None:
    _make_org(tmp_path, "orphan", body="# O\n", extends="ghost")
    assert _resolved_org_chain("orphan", tmp_path) == ["orphan"]


def test_resolved_chain_clean_when_no_extends(tmp_path: Path) -> None:
    _make_org(tmp_path, "solo", body="# S\n")
    assert _resolved_org_chain("solo", tmp_path) == ["solo"]


# --- org_agent_slugs (chain-inherited roster union, root->child) -----------

def test_roster_inherits_parent_slugs(tmp_path: Path) -> None:
    _make_org(tmp_path, "base", body="# Base\n", agents=["alpha", "beta"])
    _make_org(tmp_path, "child", body="# Child\n", agents=["gamma"], extends="base")
    # parent slugs first (root→child order), own appended.
    assert org_agent_slugs("child", tmp_path) == ["alpha", "beta", "gamma"]


def test_roster_specialization_does_not_dup(tmp_path: Path) -> None:
    # child redeclares an inherited slug -> specialization (one entry, parent pos).
    _make_org(tmp_path, "base", body="# Base\n", agents=["alpha"])
    _make_org(tmp_path, "child", body="# Child\n", agents=["alpha"], extends="base")
    assert org_agent_slugs("child", tmp_path) == ["alpha"]


def test_roster_multilevel_accumulates(tmp_path: Path) -> None:
    _make_org(tmp_path, "gp", body="# GP\n", agents=["a"])
    _make_org(tmp_path, "mid", body="# Mid\n", agents=["b"], extends="gp")
    _make_org(tmp_path, "kid", body="# Kid\n", agents=["c"], extends="mid")
    assert org_agent_slugs("kid", tmp_path) == ["a", "b", "c"]


def test_own_roster_no_inheritance(tmp_path: Path) -> None:
    _make_org(tmp_path, "solo", body="# S\n", agents=["alpha", "beta"])
    assert _own_org_agent_slugs("solo", tmp_path) == ["alpha", "beta"]


def test_roster_empty_when_no_agents_key(tmp_path: Path) -> None:
    _make_org(tmp_path, "ctoonly", body="# C\n")  # no org.yaml
    assert org_agent_slugs("ctoonly", tmp_path) == []


# --- _agent_search_dirs (chain-aware: child-local -> ancestors -> _shared) --

def test_search_dirs_child_first_then_ancestors_then_shared(tmp_path: Path) -> None:
    _make_org(tmp_path, "base", body="# Base\n", agents=["alpha"])
    _make_org(tmp_path, "child", body="# Child\n", extends="base")
    # ``_agent_search_dirs`` only yields dirs that EXIST on disk (first-hit
    # resolution), so materialize each link in the chain + ``_shared``.
    _make_agent(tmp_path, "alpha", "base", body="x")        # base/agents
    _make_agent(tmp_path, "local", "child", body="x")       # child/agents
    (tmp_path / "orgs" / "_shared" / "agents").mkdir(parents=True)
    dirs = [str(d) for d in _agent_search_dirs("child", tmp_path)]
    assert dirs[0].endswith("orgs/child/agents")
    assert dirs[1].endswith("orgs/base/agents")
    assert dirs[-1].endswith("orgs/_shared/agents")


def test_inherited_slug_resolves_through_parent_dir(tmp_path: Path) -> None:
    # alpha lives in BASE's agents/ dir; child inherits it via the roster.
    _make_org(tmp_path, "base", body="# Base\n", agents=["alpha"])
    _make_agent(tmp_path, "alpha", "base", body="ALPHA BODY")
    _make_org(tmp_path, "child", body="# Child\n", extends="base")
    spec = _load_agent_spec("alpha", "child", tmp_path)
    assert spec is not None
    assert "ALPHA BODY" in spec["system_prompt"]


def test_child_specializes_inherited_agent(tmp_path: Path) -> None:
    # child drops its own alpha.md -> child's wins (specialization), base shadowed.
    _make_org(tmp_path, "base", body="# Base\n", agents=["alpha"])
    _make_agent(tmp_path, "alpha", "base", body="BASE ALPHA")
    _make_org(tmp_path, "child", body="# Child\n", agents=["alpha"], extends="base")
    _make_agent(tmp_path, "alpha", "child", body="CHILD ALPHA")
    spec = _load_agent_spec("alpha", "child", tmp_path)
    assert spec is not None
    assert "CHILD ALPHA" in spec["system_prompt"]
    assert "BASE ALPHA" not in spec["system_prompt"]


# --- AGENTS.md overlay concat (root->child, own last) ----------------------

def test_chain_overlay_concats_root_to_child(tmp_path: Path) -> None:
    _make_org(tmp_path, "base", body="BASE OVERLAY")
    _make_org(tmp_path, "child", body="CHILD OVERLAY", extends="base")
    overlay = _chain_overlay("child", tmp_path)
    assert overlay.index("BASE OVERLAY") < overlay.index("CHILD OVERLAY")


def test_chain_overlay_multilevel_order(tmp_path: Path) -> None:
    _make_org(tmp_path, "gp", body="GP")
    _make_org(tmp_path, "mid", body="MID", extends="gp")
    _make_org(tmp_path, "kid", body="KID", extends="mid")
    overlay = _chain_overlay("kid", tmp_path)
    assert overlay.index("GP") < overlay.index("MID") < overlay.index("KID")


def test_chain_overlay_non_extending_is_own_only(tmp_path: Path) -> None:
    _make_org(tmp_path, "solo", body="SOLO OVERLAY")
    assert _chain_overlay("solo", tmp_path) == "SOLO OVERLAY"


def test_build_system_prompt_root_plus_chain_overlay_plus_addendum(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("ROOT PROMPT")
    _make_org(tmp_path, "base", body="BASE OVERLAY")
    _make_org(tmp_path, "child", body="CHILD OVERLAY", extends="base")
    prompt = build_system_prompt("child", project_root=tmp_path)
    assert "ROOT PROMPT" in prompt
    assert prompt.index("ROOT PROMPT") < prompt.index("BASE OVERLAY")
    assert prompt.index("BASE OVERLAY") < prompt.index("CHILD OVERLAY")
    # addendum appended verbatim
    with_add = build_system_prompt("child", project_root=tmp_path, addendum="\nADDENDUM\n")
    assert with_add.endswith("ADDENDUM\n")


def test_build_system_prompt_non_extending_byte_identical_shape(tmp_path: Path) -> None:
    # non-extending org: root + own overlay (no parent fragment, single seam).
    (tmp_path / "AGENTS.md").write_text("ROOT")
    _make_org(tmp_path, "solo", body="SOLO OVERLAY")
    prompt = build_system_prompt("solo", project_root=tmp_path)
    assert "ROOT" in prompt and "SOLO OVERLAY" in prompt
    assert prompt.count("SOLO OVERLAY") == 1
