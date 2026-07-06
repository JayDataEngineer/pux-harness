"""Phase 6 — ``supervisor_skills_roots`` is the focused set for the CTO's
native ``SkillsMiddleware``.

The kit exposes ONE function that picks the supervisor's skills roots:
``orgs/_shared/skills`` + THIS org's own ``skills/`` (existing dirs only),
mapped per ``workspace_root``. This is the FOCUSED set (the org's own +
shared) — distinct from the broad every-org catalog
``pux_sandbox_list_skills`` exposes (a discovery aid that COMPLEMENTS the
middleware). ``pux_sandbox_load_skill`` is GONE; bodies peek via native
``read_file`` (proven at the orchestrator layer in ``test_skills_peek.py``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pux_harness.kit.loaders import supervisor_skills_roots


def _make_skills(root: Path, rel: str) -> Path:
    """Materialize a skills root with one SKILL.md so ``is_dir()`` is True."""
    d = root / rel
    d.mkdir(parents=True)
    (d / "demo").mkdir()
    (d / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: d\n---\nbody\n"
    )
    return d


# --- the focused set + path mapping ----------------------------------------


def test_returns_shared_plus_own_host_absolute(tmp_path: Path):
    """Default ``workspace_root=None`` → ABSOLUTE host paths, in candidate
    order: shared first, then the org's own."""
    _make_skills(tmp_path, "orgs/_shared/skills")
    _make_skills(tmp_path, "orgs/foo/skills")
    out = supervisor_skills_roots("foo", tmp_path)
    assert out == [
        str(tmp_path / "orgs/_shared/skills"),
        str(tmp_path / "orgs/foo/skills"),
    ]


def test_returns_container_absolute_when_workspace_root_pinned(tmp_path: Path):
    """The harness pins ``workspace_root="/sandbox/workspace"`` → container-
    absolute paths (the project is bind-mounted 1:1 at that path)."""
    _make_skills(tmp_path, "orgs/_shared/skills")
    _make_skills(tmp_path, "orgs/foo/skills")
    out = supervisor_skills_roots("foo", tmp_path, workspace_root="/sandbox/workspace")
    assert out == [
        "/sandbox/workspace/orgs/_shared/skills",
        "/sandbox/workspace/orgs/foo/skills",
    ]


# --- existing-only filter + the specialist-org candidate -------------------


def test_drops_missing_dirs_existing_only(tmp_path: Path):
    """A candidate that doesn't exist is silently dropped (the broad
    ``_resolve_skills`` KeyError never fires — pre-filtered to existing)."""
    _make_skills(tmp_path, "orgs/_shared/skills")  # no orgs/foo/skills
    out = supervisor_skills_roots("foo", tmp_path)
    assert out == [str(tmp_path / "orgs/_shared/skills")]


def test_specialist_org_uses_specialists_path(tmp_path: Path):
    """A specialist org (e.g. ``dev-bot``) keeps its skills under
    ``orgs/specialists/<org>/skills``; the bare ``orgs/<org>/skills``
    candidate is correctly absent for it."""
    _make_skills(tmp_path, "orgs/_shared/skills")
    _make_skills(tmp_path, "orgs/specialists/dev-bot/skills")
    out = supervisor_skills_roots("dev-bot", tmp_path, workspace_root="/sandbox/workspace")
    assert out == [
        "/sandbox/workspace/orgs/_shared/skills",
        "/sandbox/workspace/orgs/specialists/dev-bot/skills",
    ]


def test_empty_when_no_skills_dirs(tmp_path: Path):
    """A no-skills org → ``[]``. The binding turns ``[]`` into ``skills=None``
    (no SkillsMiddleware mounted) — byte-identical to the pre-Phase-6 stack.
    Proven at the orchestrator layer in ``test_skills_peek.py``."""
    out = supervisor_skills_roots("foo", tmp_path)
    assert out == []


def test_candidate_order_is_shared_then_own(tmp_path: Path):
    """Order is the candidate-list order (shared, then own, then specialist),
    NOT filesystem discovery order — stable across platforms."""
    _make_skills(tmp_path, "orgs/foo/skills")
    _make_skills(tmp_path, "orgs/_shared/skills")
    out = supervisor_skills_roots("foo", tmp_path)
    assert out[0].endswith("orgs/_shared/skills")
    assert out[1].endswith("orgs/foo/skills")
