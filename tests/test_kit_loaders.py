"""Tests for the portable loaders — the kit's pure, project_root-parameterized
core. No Docker, no model registry: these read an org folder off the fs and
return data. They run identically here and inside the pux harness shim (which
re-imports them), so this is the one source of truth."""
from __future__ import annotations

from pathlib import Path

import pytest

from pux_harness.kit.loaders import (
    _load_agent_spec,
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
