"""Tests for the portable loaders — the kit's pure, project_root-parameterized
core. No Docker, no model registry: these read an org folder off the fs and
return data. They run identically here and inside the pux harness shim (which
re-imports them), so this is the one source of truth."""
from __future__ import annotations

from pathlib import Path

import pytest

from pux_harness.kit.loaders import (
    _load_agent_spec,
    _merge_extends,
    _resolve_skills,
    build_system_prompt,
    discover_orgs,
    load_org_prompt,
    load_root_prompt,
    org_agent_slugs,
)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A minimal project root: a root AGENTS.md, one org with a specialist +
    a skill, and a shared agent the org can fall back to."""
    (tmp_path / "AGENTS.md").write_text("# Base\n\nYou are an assistant.\n")
    org = tmp_path / "orgs" / "kitorg"
    (org / "agents").mkdir(parents=True)
    (org / "skills" / "wan2gp").mkdir(parents=True)
    (org / "AGENTS.md").write_text("# Kit Org\n\nOverlay instructions.\n")
    (org / "org.yaml").write_text("agents: [worker]\n")
    (org / "agents" / "worker.md").write_text(
        "---\nname: worker\ndescription: d\ntools: [generate_form]\nskills: [orgs/kitorg/skills]\n---\n\n# Worker\n\nDo work.\n"
    )
    (org / "skills" / "wan2gp" / "SKILL.md").write_text(
        "---\nname: wan2gp\ndescription: d\n---\n\n# Wan2GP\n\nparams.\n"
    )
    shared = tmp_path / "orgs" / "_shared" / "agents"
    shared.mkdir(parents=True)
    (shared / "helper.md").write_text("---\nname: helper\ndescription: d\n---\n\n# Helper\n\nshared.\n")
    return tmp_path


def test_discover_orgs_finds_org(tree: Path) -> None:
    assert discover_orgs(tree) == ["kitorg"]


def test_org_agent_slugs(tree: Path) -> None:
    assert org_agent_slugs("kitorg", tree) == ["worker"]


def test_load_root_prompt_and_overlay(tree: Path) -> None:
    assert "You are an assistant." in load_root_prompt(tree)
    assert "Overlay instructions." in load_org_prompt("kitorg", tree)


def test_build_system_prompt_combines_then_addendum(tree: Path) -> None:
    prompt = build_system_prompt("kitorg", project_root=tree)
    assert "You are an assistant." in prompt  # root
    assert "Overlay instructions." in prompt   # org overlay
    # addendum is appended verbatim, default empty
    with_add = build_system_prompt("kitorg", project_root=tree, addendum="\nEXTRA RULES\n")
    assert with_add.endswith("EXTRA RULES\n")


def test_build_system_prompt_works_without_root_agents(tmp_path: Path) -> None:
    """A standalone app may keep its base prompt entirely in the org overlay."""
    org = tmp_path / "orgs" / "solo"
    org.mkdir(parents=True)
    (org / "AGENTS.md").write_text("# Solo\n\noverlay only.\n")
    (org / "org.yaml").write_text("agents: []\n")
    prompt = build_system_prompt("solo", project_root=tmp_path)
    assert "overlay only." in prompt
    assert "solo" in prompt.lower()  # org overlay only; no root AGENTS.md present


def test_load_agent_spec_local_wins_over_shared(tree: Path) -> None:
    worker = _load_agent_spec("worker", "kitorg", tree)
    assert worker is not None
    assert worker["name"] == "worker"
    assert worker["tools"] == ["generate_form"]
    assert worker["skills"] == ["orgs/kitorg/skills"]
    assert "Do work." in worker["system_prompt"]

    helper = _load_agent_spec("helper", "kitorg", tree)
    assert helper is not None
    assert helper["name"] == "helper"
    assert "shared." in helper["system_prompt"]

    assert _load_agent_spec("ghost", "kitorg", tree) is None


def test_resolve_skills_local_vs_workspace(tree: Path) -> None:
    # kit default (workspace_root=None): ABSOLUTE local paths under project_root.
    local = _resolve_skills("orgs/kitorg/skills", "worker", project_root=tree)
    assert local == [str(tree / "orgs" / "kitorg" / "skills")]

    # harness mode (workspace_root set): container-absolute paths.
    container = _resolve_skills(
        "orgs/kitorg/skills", "worker", project_root=tree, workspace_root="/sandbox/workspace",
    )
    assert container == ["/sandbox/workspace/orgs/kitorg/skills"]


def test_resolve_skills_rejects_bad_paths(tree: Path) -> None:
    with pytest.raises(KeyError, match="no such directory"):
        _resolve_skills("orgs/kitorg/nope", "worker", project_root=tree)
    with pytest.raises(ValueError, match="project-relative"):
        _resolve_skills("/abs/path", "worker", project_root=tree)
    with pytest.raises(ValueError, match="project-relative"):
        _resolve_skills("../escape", "worker", project_root=tree)


# --- _merge_extends (the per-agent override vocabulary) -----------
#
# The merge is the universal per-agent override surface — the SAME fields that
# work org-wide via profile.yaml work per-agent via frontmatter. These unit-test
# every merge rule in isolation against a hand-built base spec dict.

def _base_spec(**over: object) -> dict:
    """A fresh, fully-populated base spec (each test merges a delta onto it)."""
    spec: dict = {
        "name": "base",
        "description": "base desc",
        "tools": ["alpha", "beta"],
        "skills": ["orgs/shared/skills"],
        "model": "mimo-v2.5",
        "tool_description_overrides": {"pux_sandbox_alpha": "orig"},
        "system_prompt": "BASE BODY",
    }
    spec.update(over)
    return spec


def test_merge_extends_tools_add_appends_in_order() -> None:
    merged = _merge_extends(_base_spec(), {"tools_add": ["gamma"]}, "")
    assert merged["tools"] == ["alpha", "beta", "gamma"]


def test_merge_extends_tools_add_dedupes_by_suffix() -> None:
    # ``alpha`` already present (by suffix) -> not re-added.
    merged = _merge_extends(_base_spec(), {"tools_add": ["alpha", "gamma"]}, "")
    assert merged["tools"] == ["alpha", "beta", "gamma"]


def test_merge_extends_tools_remove_drops_by_suffix() -> None:
    merged = _merge_extends(_base_spec(tools=["alpha", "beta", "gamma"]),
                            {"tools_remove": ["beta"]}, "")
    assert merged["tools"] == ["alpha", "gamma"]


def test_merge_extends_explicit_tools_full_replace() -> None:
    # An explicit ``tools:`` is a FULL replace — opt into a fixed whitelist,
    # not a union (matches _resolve_tools semantics).
    merged = _merge_extends(_base_spec(), {"tools": ["delta"]}, "")
    assert merged["tools"] == ["delta"]


def test_merge_extends_skills_add_union() -> None:
    merged = _merge_extends(_base_spec(), {"skills_add": ["orgs/o/skills"]}, "")
    assert merged["skills"] == ["orgs/shared/skills", "orgs/o/skills"]


def test_merge_extends_skills_explicit_full_replace() -> None:
    merged = _merge_extends(_base_spec(), {"skills": ["orgs/only/skills"]}, "")
    assert merged["skills"] == ["orgs/only/skills"]


def test_merge_extends_description_delta_wins() -> None:
    merged = _merge_extends(_base_spec(), {"description": "new"}, "")
    assert merged["description"] == "new"


def test_merge_extends_description_append_concatenates() -> None:
    merged = _merge_extends(_base_spec(), {"description_append": "EXTRA"}, "")
    assert merged["description"] == "base desc EXTRA"
    # append on top of a delta description wins the base too
    merged2 = _merge_extends(_base_spec(), {"description": "new",
                                            "description_append": "EXTRA"}, "")
    assert merged2["description"] == "new EXTRA"


def test_merge_extends_model_delta_wins() -> None:
    merged = _merge_extends(_base_spec(), {"model": "glm-5.2"}, "")
    assert merged["model"] == "glm-5.2"


def test_merge_extends_tool_description_overrides_per_key_merge() -> None:
    # new key added, existing key preserved
    merged = _merge_extends(
        _base_spec(),
        {"tool_description_overrides": {"pux_sandbox_beta": "new"}}, "",
    )
    assert merged["tool_description_overrides"] == {
        "pux_sandbox_alpha": "orig", "pux_sandbox_beta": "new",
    }
    # delta wins on conflict
    merged2 = _merge_extends(
        _base_spec(),
        {"tool_description_overrides": {"pux_sandbox_alpha": "overridden"}}, "",
    )
    assert merged2["tool_description_overrides"] == {"pux_sandbox_alpha": "overridden"}


def test_merge_extends_system_prompt_concatenated() -> None:
    merged = _merge_extends(_base_spec(), {}, "DELTA BODY")
    assert merged["system_prompt"] == "BASE BODY\n\nDELTA BODY"
    # no delta body -> base body unchanged
    assert _merge_extends(_base_spec(), {}, "")["system_prompt"] == "BASE BODY"


def test_merge_extends_no_delta_fields_is_base_plus_body() -> None:
    # A child that only adds a body (prompt_append) inherits everything else.
    merged = _merge_extends(_base_spec(), {}, "MORE")
    assert merged["name"] == "base"
    assert merged["tools"] == ["alpha", "beta"]
    assert merged["model"] == "mimo-v2.5"


# --- _load_agent_spec extends recursion (resolution + cycle) -----

def _write_agent(root: Path, slug: str, *, org: str = "_shared",
                 body: str = "BODY", fm: str = "") -> None:
    """Write ``orgs/<org>/agents/<slug>.md`` with optional extra frontmatter
    lines (used for ``extends:`` + the delta fields)."""
    d = root / "orgs" / org / "agents"
    d.mkdir(parents=True, exist_ok=True)
    head = f"---\nname: {slug}\ndescription: {slug} agent\n{fm}---\n\n"
    (d / f"{slug}.md").write_text(head + body + "\n")


def test_load_agent_spec_extends_inherits_base_tools_and_body(tmp_path: Path) -> None:
    """An org-local child with ``extends: base`` + ``tools_add`` inherits the
    base's body + tool whitelist AND adds the new tool — no fork needed."""
    _write_agent(tmp_path, "base", fm="tools: [alpha]\n", body="# Base\n\nBASE BODY.")
    _write_agent(tmp_path, "child", org="o",
                 fm="extends: base\ntools_add: [beta]\n",
                 body="# Child\n\nCHILD BODY.")
    spec = _load_agent_spec("child", "o", tmp_path)
    assert spec is not None
    assert spec["name"] == "child"
    # inherited alpha + added beta, base order preserved
    assert spec["tools"] == ["alpha", "beta"]
    assert "BASE BODY." in spec["system_prompt"]
    assert "CHILD BODY." in spec["system_prompt"]
    # base resolves independently (no extends key in its merged output)
    base = _load_agent_spec("base", "o", tmp_path)
    assert base is not None
    assert "extends" not in base
    assert base["system_prompt"] == "# Base\n\nBASE BODY."


def test_load_agent_spec_extends_chain_multilevel(tmp_path: Path) -> None:
    """c extends b extends a: tools accumulate, bodies concatenate in order."""
    _write_agent(tmp_path, "a", fm="tools: [t1]\n", body="A BODY")
    _write_agent(tmp_path, "b", fm="extends: a\ntools_add: [t2]\n", body="B BODY")
    _write_agent(tmp_path, "c", fm="extends: b\ntools_add: [t3]\n", body="C BODY")
    spec = _load_agent_spec("c", "o", tmp_path)
    assert spec is not None
    assert spec["tools"] == ["t1", "t2", "t3"]
    for fragment in ("A BODY", "B BODY", "C BODY"):
        assert fragment in spec["system_prompt"]


def test_load_agent_spec_extends_cycle_raises(tmp_path: Path) -> None:
    _write_agent(tmp_path, "x", fm="extends: y\n", body="X")
    _write_agent(tmp_path, "y", fm="extends: x\n", body="Y")
    with pytest.raises(ValueError, match="extends cycle"):
        _load_agent_spec("x", "o", tmp_path)


def test_load_agent_spec_extends_self_cycle_raises(tmp_path: Path) -> None:
    _write_agent(tmp_path, "loopy", fm="extends: loopy\n", body="L")
    with pytest.raises(ValueError, match="extends cycle"):
        _load_agent_spec("loopy", "o", tmp_path)


def test_load_agent_spec_extends_unresolvable_raises(tmp_path: Path) -> None:
    _write_agent(tmp_path, "lonely", fm="extends: ghost\n", body="L")
    with pytest.raises(FileNotFoundError, match="no such agent"):
        _load_agent_spec("lonely", "o", tmp_path)


def test_load_agent_spec_extends_non_string_raises(tmp_path: Path) -> None:
    _write_agent(tmp_path, "bad", fm="extends: []\n", body="B")
    with pytest.raises(ValueError, match="extends must be a non-empty"):
        _load_agent_spec("bad", "o", tmp_path)

