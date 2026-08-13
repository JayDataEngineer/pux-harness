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
3. **MCP resolution (CU-4 strict one-way)** — ``org.yaml capabilities:
   [{kind: mcp, ref: equibles}]`` is the ONE resolution path for foreign MCP
   servers. The pre-unification ``policy.yaml tool_servers:`` read path was
   REMOVED: it is unread by ``resolve_tool_servers`` (declares nothing) and is
   a permanent contract failure (``no-legacy-tool-servers``) — the two no
   longer compose, there is no union.

Docker-free: every path is a declaration loader. The frontmatter parity uses
``kit.loaders._load_agent_spec`` directly (explicit ``project_root``); the mcp
parity patches ``tool_servers``' path resolvers at a scratch tree.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import pux_harness.agent.tool_servers as TS
from pux_harness.agent.tool_servers import resolve_tool_servers
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


@pytest.mark.parametrize("kind", ["middleware", "job"])
def test_desugar_rejects_non_surface_kind_in_frontmatter(kind):
    """A kind not in the agent frontmatter sugar surface (``mcp`` now IS, but
    ``middleware`` / ``job`` are not — no ref-catalog) fails loud, not silently
    routed or dropped. ``mcp`` is valid in BOTH homes now (org arms, agent
    routes a focused subset), so it's no longer rejected here."""
    with pytest.raises(CapabilitiesSugarError, match="not in the agent frontmatter sugar surface"):
        desugar_agent_capabilities({"capabilities": [{"kind": kind, "ref": "x"}]}, "x")


def test_desugar_rejects_bad_shapes():
    with pytest.raises(CapabilitiesSugarError, match="must be a list"):
        desugar_agent_capabilities({"capabilities": "oops"}, "x")
    with pytest.raises(CapabilitiesSugarError, match="must be a mapping"):
        desugar_agent_capabilities({"capabilities": ["x"]}, "x")
    with pytest.raises(CapabilitiesSugarError, match="ref must be"):
        desugar_agent_capabilities({"capabilities": [{"kind": "tool"}]}, "x")
    with pytest.raises(CapabilitiesSugarError, match="unknown kind|not in the agent frontmatter sugar surface"):
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
    ref semantics. ``mcp`` is valid in BOTH homes (org arms; agent routes a
    focused subset). Locked by design."""
    assert AGENT_CAPABILITY_KINDS == ("tool", "skill", "mcp")


# --- Part 1b: agent ``kind: mcp`` desugar (focused subagent subset) ----------


def test_desugar_mcp_routes_to_mcp_key_not_tools():
    # bare catalog-ref string -> the ref alone
    out = desugar_agent_capabilities({"capabilities": [{"kind": "mcp", "ref": "web_research"}]}, "x")
    assert out["mcp"] == ["web_research"]
    # must NOT land in tools (those resolve via the mcp-excluding tool_map)
    assert "tools" not in out and "skills" not in out
    assert "capabilities" not in out


def test_desugar_mcp_allowlist_and_tools_alias():
    # allowlist -> {ref, tools} mapping (same shape org_mcp_items_from_dict emits)
    out = desugar_agent_capabilities(
        {"capabilities": [{"kind": "mcp", "ref": "e", "allowlist": ["filings", "prices"]}]}, "x"
    )
    assert out["mcp"] == [{"ref": "e", "tools": ["filings", "prices"]}]
    # `tools:` is accepted as an alias for `allowlist:` (symmetry with org.yaml)
    out2 = desugar_agent_capabilities(
        {"capabilities": [{"kind": "mcp", "ref": "e", "tools": ["x"]}]}, "x"
    )
    assert out2["mcp"] == [{"ref": "e", "tools": ["x"]}]


def test_desugar_mcp_composes_with_tool_and_skill():
    out = desugar_agent_capabilities(
        {"capabilities": [
            {"kind": "tool", "ref": "python"},
            {"kind": "mcp", "ref": "web_research"},
            {"kind": "skill", "ref": "./s"},
        ]}, "x",
    )
    assert out["tools"] == ["python"]
    assert out["skills"] == ["./s"]
    assert out["mcp"] == ["web_research"]


def test_desugar_mcp_bad_allowlist_fails_loud():
    with pytest.raises(CapabilitiesSugarError, match="allowlist must be a list"):
        desugar_agent_capabilities(
            {"capabilities": [{"kind": "mcp", "ref": "e", "allowlist": "filings"}]}, "x"
        )


# --- _resolve_mcp (harness: lenient two-level declared-gate) -----------------
# ``_resolve_mcp`` only reads ``t.name`` and returns the objects as-is, so a
# lightweight namespace stand-in is a faithful probe of the resolution logic.
#
# The gate is two-level (see ``_resolve_mcp`` docstring): level 1 = DECLARED by
# the org (``declared_servers`` kwarg) — a ref the org never declared is a
# config error and fails loud; level 2 = ARMED this run (``mcp_tools``) — a
# declared ref with zero armed tools (server offline/unreachable) is LENIENT:
# it contributes nothing and the build lives. Happy-path tests pass
# ``declared_servers`` to exercise the real contract (the default empty set
# treats every ref as undeclared -> fail loud, the misuse-guard).

from types import SimpleNamespace  # noqa: E402


def _mk_tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


_WEB = ["mcp__web_research__research", "mcp__web_research__search", "mcp__web_research__fetch"]
_EQ = ["mcp__equibles__filings", "mcp__equibles__prices"]


def test_resolve_mcp_harness_bare_ref_takes_all_server_tools():
    from pux_harness.agent.orgs import _resolve_mcp
    tools = [_mk_tool(n) for n in _WEB + _EQ]
    got = _resolve_mcp(["web_research"], tools, "ws", declared_servers={"web_research", "equibles"})
    assert {t.name for t in got} == set(_WEB)


def test_resolve_mcp_harness_allowlist_narrows_and_exact_server_match():
    from pux_harness.agent.orgs import _resolve_mcp
    tools = [_mk_tool(n) for n in _WEB + _EQ]
    # equibles ≠ equibles-extra: the trailing ``__`` is an exact-server guard,
    # and the bare name is the post-prefix segment.
    got = _resolve_mcp(
        [{"ref": "web_research", "tools": ["search"]}], tools, "ws",
        declared_servers={"web_research", "equibles"},
    )
    assert [t.name for t in got] == ["mcp__web_research__search"]


def test_resolve_mcp_harness_undeclared_ref_fails_loud_naming_declared():
    """Level-1 config error: an agent refs a server the org never DECLARED.
    This is the typo / forgotten-arm misconfig — it fails loud (naming the
    declared set) so a focused specialist can never silently inherit the
    supervisor's whole MCP surface."""
    from pux_harness.agent.orgs import _resolve_mcp
    tools = [_mk_tool(n) for n in _WEB]  # org armed + declared web_research only
    with pytest.raises(KeyError, match=r"not declared by this org"):
        _resolve_mcp(["equibles"], tools, "ws", declared_servers={"web_research"})


def test_resolve_mcp_harness_declared_but_unarmed_is_lenient():
    """Level-2 leniency: a DECLARED ref with zero armed tools this run (server
    offline/unreachable, or the org building offline) contributes NOTHING and
    does NOT raise — mirrors the org layer's own leniency
    (``resolve_tool_servers`` tolerates an unreachable declared server)."""
    from pux_harness.agent.orgs import _resolve_mcp
    tools = [_mk_tool(n) for n in _WEB]  # equibles declared but NOT armed
    got = _resolve_mcp(["equibles"], tools, "ws", declared_servers={"equibles"})
    assert got == []


def test_resolve_mcp_harness_empty_declared_default_fails_loud_on_any_ref():
    """Misuse-guard: the default empty ``declared_servers`` treats every ref as
    undeclared -> fail loud. ``load_subagents`` always passes the real set, so a
    caller hitting this default forgot to thread it."""
    from pux_harness.agent.orgs import _resolve_mcp
    tools = [_mk_tool(n) for n in _WEB]
    with pytest.raises(KeyError, match=r"not declared by this org"):
        _resolve_mcp(["web_research"], tools, "ws")  # no declared_servers -> default frozenset()


def test_resolve_mcp_harness_missing_allowlist_name_fails_loud():
    from pux_harness.agent.orgs import _resolve_mcp
    tools = [_mk_tool(n) for n in _WEB]
    with pytest.raises(KeyError, match="allowlist names"):
        _resolve_mcp(
            [{"ref": "web_research", "tools": ["nope"]}], tools, "ws",
            declared_servers={"web_research"},
        )


def test_resolve_mcp_harness_empty_is_noop():
    from pux_harness.agent.orgs import _resolve_mcp
    tools = [_mk_tool(n) for n in _WEB]
    assert _resolve_mcp(None, tools, "ws") == []
    assert _resolve_mcp([], tools, "ws") == []


def test_resolve_mcp_kit_is_lenient_on_unarmed_ref():
    """The kit sibling SKIPS a ref the consumer didn't ship (no raise) — mirrors
    the kit ``_resolve_tools`` leniency. The harness is the strict/loud path."""
    from pux_harness.kit.compile import _resolve_mcp
    tools = [_mk_tool(n) for n in _WEB]
    assert _resolve_mcp(["equibles"], tools, "ws") == []  # skipped, not raised
    assert {t.name for t in _resolve_mcp(["web_research"], tools, "ws")} == set(_WEB)


def test_focused_mcp_subagent_surface_excludes_kitchen_sink(tmp_path):
    """E2E (kit, docker-free): an agent whose ONLY capability is
    ``{kind: mcp, ref: web_research}`` compiles to a subagent whose ``tools`` are
    EXACTLY that server's tools — no inherited specialists, no other mcp server.
    This is the whole point of focused mcp: a perfectable, rapid specialist that
    keeps big-model context clean. The flip lives in ``_build_sub``
    (``tools or mcp`` -> set ``tools``, disabling deepagents inheritance)."""
    from pux_harness.kit.compile import load_subagents
    org_dir = tmp_path / "orgs" / _ORG
    (org_dir / "agents").mkdir(parents=True)
    (org_dir / "AGENTS.md").write_text(f"# {_ORG}\n")  # valid discoverable org
    (org_dir / "org.yaml").write_text("agents: [ws]\n")
    _write_agent(
        org_dir / "agents", "ws",
        "name: ws\ndescription: d\ncapabilities:\n  - {kind: mcp, ref: web_research}\n",
    )
    # consumer ships web_research + equibles + a specialist the agent did NOT ask for
    tools = [_mk_tool(n) for n in _WEB + _EQ + ["pux_sandbox_execute"]]
    subs = load_subagents(_ORG, tools, project_root=tmp_path)
    assert len(subs) == 1
    assert {t.name for t in subs[0]["tools"]} == set(_WEB)  # focused — no kitchen sink


def test_focused_subagent_combines_tool_and_mcp(tmp_path):
    """Inverse proof: an agent declaring BOTH a ``kind: tool`` and a ``kind: mcp``
    gets BOTH in its focused surface (and still excludes undeclared tools)."""
    from pux_harness.kit.compile import load_subagents
    org_dir = tmp_path / "orgs" / _ORG
    (org_dir / "agents").mkdir(parents=True)
    (org_dir / "AGENTS.md").write_text(f"# {_ORG}\n")
    (org_dir / "org.yaml").write_text("agents: [mix]\n")
    _write_agent(
        org_dir / "agents", "mix",
        "name: mix\ndescription: d\ncapabilities:\n"
        "  - {kind: tool, ref: pux_sandbox_execute}\n"
        "  - {kind: mcp, ref: equibles}\n",
    )
    tools = [_mk_tool(n) for n in _WEB + _EQ + ["pux_sandbox_execute"]]
    subs = load_subagents(_ORG, tools, project_root=tmp_path)
    assert len(subs) == 1
    assert {t.name for t in subs[0]["tools"]} == {
        "pux_sandbox_execute", *_EQ,
    }  # the tool + the equibles server; web_research excluded


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


def _patch_contract_to_scratch(monkeypatch, orgs_dir: Path) -> Path:
    """Redirect ``audit_org``'s resolution at a scratch ``orgs/`` tree.

    Mirrors the parent ``fake_orgs_tree`` fixture (``tests/conftest.py``): the
    contract-test seam is patched at BOTH module sites — ``org_validation._orgs_dir``
    AND ``orgs._orgs_dir`` — so every ``orgs.py`` delegate that reads
    ``orgs._orgs_dir()`` (``_agent_search_dirs`` / ``org_agent_slugs`` / …), not
    just ``audit_org``'s direct ``org_validation._orgs_dir()`` calls, resolves against
    the scratch tree. Patching only ``org_validation._orgs_dir`` leaves those delegates
    walking the real tree (``search_org_dir`` → ``FileNotFoundError``).
    ``_shared/agents`` is pre-created (the search-dirs root the roster walk
    touches)."""
    from pux_harness.agent import org_validation, orgs

    (orgs_dir / "_shared" / "agents").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(org_validation, "_orgs_dir", lambda: orgs_dir)
    monkeypatch.setattr(orgs, "_orgs_dir", lambda: orgs_dir)
    return orgs_dir


def test_mcp_org_yaml_capabilities_is_sole_resolution_path(tmp_path, monkeypatch):
    """CU-4 (strict one-way): ``org.yaml capabilities: [{kind: mcp, ref: ...}]``
    is the ONE resolution path for foreign MCP servers. The pre-unification
    ``policy.yaml tool_servers:`` read path was REMOVED — declaring an mcp
    server ONLY there resolves to NOTHING (the block's real gate is the
    ``no-legacy-tool-servers`` contract rule, asserted by the
    ``test_no_legacy_tool_servers_*`` cases below)."""
    orgs_dir = tmp_path / "orgs"
    org_dir = _patch_ts_to_scratch(monkeypatch, orgs_dir, _ORG)

    # Sole path: org.yaml capabilities resolves to the catalog spec.
    (org_dir / "policy.yaml").write_text("")
    (org_dir / "org.yaml").write_text(
        "agents: []\ncapabilities:\n  - {kind: mcp, ref: equibles}\n"
    )
    _reset_catalog_cache()
    specs = resolve_tool_servers(_ORG, env={}, permissive=True)
    assert len(specs) == 1
    assert specs[0].name == "equibles"
    assert specs[0].transport == "sse"

    # Legacy ``policy.yaml tool_servers:`` is NO LONGER read by the resolver:
    # an org whose only mcp declaration is the banned policy block (no org.yaml
    # capabilities) resolves to nothing — proving it is not a resolution site.
    (org_dir / "policy.yaml").write_text("tool_servers:\n  - equibles\n")
    (org_dir / "org.yaml").write_text("agents: []\n")
    _reset_catalog_cache()
    assert resolve_tool_servers(_ORG, env={}, permissive=True) == []


def test_mcp_org_yaml_capabilities_allowlist_no_legacy_union(tmp_path, monkeypatch):
    """CU-4 (strict one-way): org.yaml mcp capabilities honor an allowlist
    override, and there is NO union with ``policy.yaml tool_servers:`` — that
    legacy block is unread by the resolver (a permanent contract failure in
    ``org_validation.py``). Only the org.yaml capability resolves; ``media`` (declared in
    the banned policy block) never surfaces."""
    orgs_dir = tmp_path / "orgs"
    org_dir = _patch_ts_to_scratch(monkeypatch, orgs_dir, _ORG)
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
    assert set(by_name) == {"equibles"}  # no union: legacy policy ``media`` unread
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


def test_no_legacy_tool_servers_policy_block_is_contract_violation(
    tmp_path, monkeypatch,
):
    """CU-4 no-legacy-left-behind gate: a ``policy.yaml tool_servers:`` block is a
    PERMANENT contract failure — ``org.yaml capabilities:`` (kind: mcp) is the one
    canonical MCP site. The gate BITES: ``audit_org`` emits a
    ``no-legacy-tool-servers`` error. (The resolver half — that the block is
    unread, declares nothing — is proven by the
    ``test_mcp_org_yaml_capabilities_*`` cases above; this proves the contract
    half that FORBIDS the block outright.)"""
    from pux_harness.agent import org_validation as C

    orgs_dir = tmp_path / "orgs"
    _patch_contract_to_scratch(monkeypatch, orgs_dir)
    org_dir = orgs_dir / _ORG
    org_dir.mkdir(parents=True, exist_ok=True)
    # minimal valid bundle: prose AGENTS.md + empty roster; the banned legacy
    # MCP declaration site is the ONLY thing in policy.yaml.
    (org_dir / "AGENTS.md").write_text(f"# {_ORG}\n")
    (org_dir / "org.yaml").write_text("agents: []\n")
    (org_dir / "policy.yaml").write_text("tool_servers:\n  - equibles\n")

    rules = {v.rule for v in C.audit_org(_ORG)}
    assert "no-legacy-tool-servers" in rules


def test_no_legacy_tool_servers_clean_org_is_silent(tmp_path, monkeypatch):
    """The negative half of the gate: an org that uses the canonical
    ``org.yaml capabilities:`` (kind: mcp) site and NO policy ``tool_servers:``
    block does NOT trip ``no-legacy-tool-servers`` — the rule never false-fires
    on the post-CU-4 form. (All real orgs migrated; this pins the negative.)"""
    from pux_harness.agent import org_validation as C

    orgs_dir = tmp_path / "orgs"
    _patch_contract_to_scratch(monkeypatch, orgs_dir)
    org_dir = orgs_dir / _ORG
    org_dir.mkdir(parents=True, exist_ok=True)
    # The rule inspects ONLY policy.yaml's raw sections; org.yaml ``capabilities:``
    # is irrelevant to it, so a bare roster keeps the negative clean (no catalog
    # resolution noise). No ``tool_servers:`` block -> the gate stays silent.
    (org_dir / "AGENTS.md").write_text(f"# {_ORG}\n")
    (org_dir / "org.yaml").write_text("agents: []\n")
    (org_dir / "policy.yaml").write_text("credentials: {}\n")

    rules = {v.rule for v in C.audit_org(_ORG)}
    assert "no-legacy-tool-servers" not in rules
