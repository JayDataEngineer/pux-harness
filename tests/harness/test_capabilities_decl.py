"""CU-3 — the ``capabilities:`` declaration sugar: parity proof.

The opt-in unified key (frontmatter ``kind ∈ {tool, skill}``; ``org.yaml``
``kind == mcp``) is PURE SUGAR: it desugars into the EXISTING kind-specific
declaration the leaf resolvers already read, so an org written in the new form
resolves bit-identically to its old form. This test proves that three ways:

1. **Desugar correctness** (unit) — ``desugar_agent_capabilities`` /
   ``org_mcp_items_from_dict`` produce the legacy keys/lists and reject the
   wrong-home / bad-shape cases loud (no silent skip).
2. **Frontmatter parity** — an agent written with ``tools:`` / ``skills:``
   loads to the SAME spec as one written with ``capabilities:`` (the per-agent
   surface the operator complained about).
3. **MCP parity** — an org with ``policy.yaml tool_servers: [equibles]``
   resolves to the SAME ``ToolServerSpec`` as one with
   ``org.yaml capabilities: [{kind: mcp, ref: equibles}]`` (the org-level mcp
   channel routed through ONE ``resolve_tool_servers``).

Docker-free: every path is a declaration loader. The frontmatter parity uses
``kit.loaders._load_agent_spec`` directly (explicit ``project_root``); the mcp
parity patches ``tool_servers``' path resolvers at a scratch tree.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import pux_harness.agent.tool_servers as TS
from pux_harness.agent.tool_servers import ToolServerSpec, resolve_tool_servers
from pux_harness.kit.capabilities_decl import (
    AGENT_CAPABILITY_KINDS,
    CapabilitiesSugarError,
    desugar_agent_capabilities,
    org_mcp_items_from_dict,
)
from pux_harness.kit.loaders import _load_agent_spec

_ORG = "capdecl"


# --- Part 1: desugar correctness (unit) --------------------------------------


def test_desugar_merges_tool_and_skill_into_legacy_keys():
    fm = {
        "name": "x",
        "description": "d",
        "capabilities": [
            {"kind": "tool", "ref": "python"},
            {"kind": "skill", "ref": "./skills"},
        ],
    }
    out = desugar_agent_capabilities(dict(fm), "x")
    assert out["tools"] == ["python"]
    assert out["skills"] == ["./skills"]
    assert "capabilities" not in out  # consumed — leaves never see it
    # name/description pass through untouched
    assert out["name"] == "x" and out["description"] == "d"


def test_desugar_composes_with_legacy_keys_and_dedups():
    fm = {
        "tools": ["python", "grep"],
        "skills": ["./a"],
        "capabilities": [
            {"kind": "tool", "ref": "python"},      # dup of legacy -> dropped
            {"kind": "tool", "ref": "edit_file"},   # new -> appended
            {"kind": "skill", "ref": "./b"},
        ],
    }
    out = desugar_agent_capabilities(dict(fm), "x")
    # legacy list preserved first, sugar appended, deduped
    assert out["tools"] == ["python", "grep", "edit_file"]
    assert out["skills"] == ["./a", "./b"]


def test_desugar_accepts_comma_string_legacy_and_merges():
    fm = {"tools": "python, grep", "capabilities": [{"kind": "tool", "ref": "edit_file"}]}
    out = desugar_agent_capabilities(dict(fm), "x")
    assert out["tools"] == ["python", "grep", "edit_file"]


def test_desugar_no_block_is_a_noop_returning_same_dict_shape():
    fm = {"name": "x", "tools": ["python"]}
    out = desugar_agent_capabilities(dict(fm), "x")
    assert out == {"name": "x", "tools": ["python"]}


@pytest.mark.parametrize("kind", ["mcp", "middleware", "job"])
def test_desugar_rejects_org_level_kind_in_frontmatter(kind):
    """An org-level kind in a per-agent file is the wrong home — fails loud,
    not silently routed or dropped."""
    with pytest.raises(CapabilitiesSugarError, match="not a per-agent kind"):
        desugar_agent_capabilities({"capabilities": [{"kind": kind, "ref": "x"}]}, "x")


def test_desugar_rejects_bad_shapes():
    with pytest.raises(CapabilitiesSugarError, match="must be a list"):
        desugar_agent_capabilities({"capabilities": "oops"}, "x")
    with pytest.raises(CapabilitiesSugarError, match="must be a mapping"):
        desugar_agent_capabilities({"capabilities": ["x"]}, "x")
    with pytest.raises(CapabilitiesSugarError, match="ref must be"):
        desugar_agent_capabilities({"capabilities": [{"kind": "tool"}]}, "x")
    with pytest.raises(CapabilitiesSugarError, match="unknown kind|not a per-agent"):
        desugar_agent_capabilities(
            {"capabilities": [{"kind": "wat", "ref": "x"}]}, "x"
        )


def test_org_mcp_items_bare_ref_and_allowlist_variants():
    # bare catalog-ref string
    assert org_mcp_items_from_dict(
        {"capabilities": [{"kind": "mcp", "ref": "equibles"}]}, "o"
    ) == ["equibles"]
    # allowlist -> {ref, tools} override
    assert org_mcp_items_from_dict(
        {"capabilities": [{"kind": "mcp", "ref": "equibles", "allowlist": ["filings"]}]}, "o"
    ) == [{"ref": "equibles", "tools": ["filings"]}]
    # `tools:` is accepted as an alias for `allowlist:`
    assert org_mcp_items_from_dict(
        {"capabilities": [{"kind": "mcp", "ref": "e", "tools": ["x"]}]}, "o"
    ) == [{"ref": "e", "tools": ["x"]}]


def test_org_mcp_items_empty_cases_yield_nothing():
    assert org_mcp_items_from_dict(None, "o") == []
    assert org_mcp_items_from_dict({}, "o") == []
    assert org_mcp_items_from_dict({"agents": ["x"]}, "o") == []


@pytest.mark.parametrize("kind", ["tool", "skill"])
def test_org_mcp_items_rejects_per_agent_kind_in_org_yaml(kind):
    with pytest.raises(CapabilitiesSugarError, match="per-agent kind"):
        org_mcp_items_from_dict({"capabilities": [{"kind": kind, "ref": "x"}]}, "o")


@pytest.mark.parametrize("kind", ["middleware", "job"])
def test_org_mcp_items_rejects_unsupported_org_kind(kind):
    """middleware/job are not in the sugar surface (no ref-catalog) — a typo or
    attempted-unsupported kind fails loud rather than silently dropping."""
    with pytest.raises(CapabilitiesSugarError, match="not in the sugar surface"):
        org_mcp_items_from_dict({"capabilities": [{"kind": kind, "ref": "x"}]}, "o")


def test_sugar_kind_constants_match_design():
    """The sugar surface is exactly the three model add-on channels with a clean
    ref semantics (tool/skill per-agent; mcp org-level). Locked by design."""
    assert AGENT_CAPABILITY_KINDS == ("tool", "skill")


# --- Part 2: frontmatter parity (the per-agent surface) ----------------------


def _write_agent(agents_dir: Path, slug: str, frontmatter: str) -> None:
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / f"{slug}.md").write_text(f"---\n{frontmatter}---\nbody\n")


def test_frontmatter_capabilities_resolves_identically_to_tools_skills(tmp_path):
    """THE CU-3 gate for the per-agent surface: an agent written the new way
    (``capabilities:``) loads to the SAME spec as one written the old way
    (``tools:`` / ``skills:``). The desugar is invisible to every downstream
    consumer (``_resolve_tools`` / Rule 4 / the runtime)."""
    agents = tmp_path / "orgs" / _ORG / "agents"
    _write_agent(
        agents, "old",
        'name: old\ndescription: d\ntools: [python, grep]\nskills: [./skills]\n',
    )
    _write_agent(
        agents, "new",
        "name: new\ndescription: d\ncapabilities:\n"
        "  - {kind: tool, ref: python}\n"
        "  - {kind: tool, ref: grep}\n"
        "  - {kind: skill, ref: ./skills}\n",
    )
    old = _load_agent_spec("old", _ORG, tmp_path)
    new = _load_agent_spec("new", _ORG, tmp_path)
    assert old is not None and new is not None
    # bit-identical tools/skills — the desugar changed nothing the leaves see
    assert old["tools"] == new["tools"] == ["python", "grep"]
    assert old["skills"] == new["skills"] == ["./skills"]
    assert "capabilities" not in new
    assert old["system_prompt"] == new["system_prompt"]


# --- Part 3: mcp parity (the org-level channel) ------------------------------


def _patch_ts_to_scratch(monkeypatch, orgs_dir: Path, org: str) -> Path:
    """Redirect ``tool_servers``' path resolvers at a scratch ``orgs/`` tree."""
    org_dir = orgs_dir / org
    org_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(TS, "_orgs_dir", lambda: orgs_dir)
    monkeypatch.setattr(TS, "_org_path", lambda o: orgs_dir / o)
    # catalog + cache point at the scratch shared catalog
    catalog = orgs_dir / "_shared" / "tool_servers.yaml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "equibles:\n"
        "  kind: mcp\n"
        "  transport: sse\n"
        "  url: https://api.equibles.com/mcp\n"
    )
    monkeypatch.setattr(TS, "_catalog_path", lambda: catalog)
    return org_dir


def _reset_catalog_cache() -> None:
    TS._catalog_cache = None


def test_mcp_org_yaml_sugar_resolves_identically_to_policy_tool_servers(
    tmp_path, monkeypatch,
):
    """THE CU-3 gate for mcp: ``policy.yaml tool_servers: [equibles]`` (old) and
    ``org.yaml capabilities: [{kind: mcp, ref: equibles}]`` (new) resolve to the
    SAME ``ToolServerSpec`` — one ``resolve_tool_servers`` path, both forms."""
    orgs_dir = tmp_path / "orgs"
    org_dir = _patch_ts_to_scratch(monkeypatch, orgs_dir, _ORG)

    # OLD form: mcp declared in policy.yaml tool_servers
    (org_dir / "policy.yaml").write_text("tool_servers:\n  - equibles\n")
    (org_dir / "org.yaml").write_text("agents: []\n")
    _reset_catalog_cache()
    old = resolve_tool_servers(_ORG, env={}, permissive=True)

    # NEW form: mcp declared in org.yaml capabilities; policy.yaml is EMPTY
    # (policy.load("") -> Policy() with tool_servers=[]), so the only declaration
    # site is org.yaml — proving both homes resolve through ONE path.
    (org_dir / "policy.yaml").write_text("")
    (org_dir / "org.yaml").write_text(
        "agents: []\ncapabilities:\n  - {kind: mcp, ref: equibles}\n"
    )
    _reset_catalog_cache()
    new = resolve_tool_servers(_ORG, env={}, permissive=True)

    def _sig(s: ToolServerSpec) -> tuple:
        return (s.name, s.kind, s.transport, s.url)

    assert [_sig(s) for s in old] == [_sig(s) for s in new]
    assert len(old) == 1 and old[0].name == "equibles"
    assert old[0].transport == "sse"


def test_mcp_org_yaml_allowlist_and_union_with_policy(tmp_path, monkeypatch):
    """org.yaml mcp sugar (a) honors an allowlist override and (b) UNIONS with
    policy.yaml tool_servers, not replaces — both declaration sites compose."""
    orgs_dir = tmp_path / "orgs"
    org_dir = _patch_ts_to_scratch(monkeypatch, orgs_dir, _ORG)
    # second catalog entry so we can exercise the union
    cat = orgs_dir / "_shared" / "tool_servers.yaml"
    cat.write_text(
        "equibles:\n  kind: mcp\n  transport: sse\n  url: https://a.example/mcp\n"
        "media:\n  kind: mcp\n  transport: sse\n  url: https://b.example/mcp\n"
    )
    (org_dir / "policy.yaml").write_text("tool_servers:\n  - media\n")
    (org_dir / "org.yaml").write_text(
        "agents: []\ncapabilities:\n"
        "  - {kind: mcp, ref: equibles, allowlist: [filings]}\n"
    )
    _reset_catalog_cache()
    specs = resolve_tool_servers(_ORG, env={}, permissive=True)
    by_name = {s.name: s for s in specs}
    assert set(by_name) == {"equibles", "media"}  # union of both sites
    assert by_name["equibles"].tools == ["filings"]  # allowlist override honored


def test_mcp_org_yaml_bad_ref_surfaces_as_tool_servers_violation(
    tmp_path, monkeypatch,
):
    """An org.yaml ``capabilities: mcp`` ref that isn't in the catalog fails
    loud through the SAME path as a bad policy ref — ``resolve_tool_servers``
    raises, which ``validate_tool_servers`` surfaces as a ``tool-servers``
    contract violation (one validation path for both declaration sites)."""
    orgs_dir = tmp_path / "orgs"
    org_dir = _patch_ts_to_scratch(monkeypatch, orgs_dir, _ORG)
    (org_dir / "policy.yaml").write_text("credentials: {}\n")
    (org_dir / "org.yaml").write_text(
        "agents: []\ncapabilities:\n  - {kind: mcp, ref: no_such_server}\n"
    )
    _reset_catalog_cache()
    with pytest.raises(ValueError, match="unknown catalog ref"):
        resolve_tool_servers(_ORG, env={}, permissive=True)
