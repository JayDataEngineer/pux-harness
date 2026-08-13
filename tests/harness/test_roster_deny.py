"""Data-driven ``roster_deny:`` contract rules.

Verifies the TWO generic rules that replaced the former ``if name == "coder"``
hard-coded branches:

1. ``roster-deny-enforced`` — no slug in ``org.yaml``'s ``roster_deny:`` list
   appears in the effective (chain-inherited) roster.
2. ``roster-deny-disables-general-purpose`` — when ``general-purpose`` /
   ``general`` is in ``roster_deny:``, the profile MUST declare
   ``general_purpose_subagent: {enabled: false}`` so deepagents' auto-add is
   neutered.

These rules are DATA-DRIVEN: any org can declare the focus-CTO shape by
populating ``roster_deny:`` in its org.yaml. There are zero org-name literals
in the checker's executable branches. The former
``coder-no-general-subagent`` / ``coder-disables-general-purpose`` rule names
are gone — one intent, two generic rules, one data field.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from pux_harness.agent import profile
from pux_harness.agent import org_validation as ov


# ``audit_org`` resolves orgs through ``kit._paths.project_root()``, which is
# ``$PUX_PROJECT_ROOT`` if set, else the CWD. When pytest runs inside the
# ``pux-harness`` submodule (no ``orgs/`` dir), the CWD fallback can't find the
# org under test. ``bin/pux`` sets ``PUX_PROJECT_ROOT`` in production; mirror
# that here by deriving the parent repo from this file's location.
# (pux-harness/tests/harness/test_roster_deny.py → parents[3] = parent repo.)
_PARENT_REPO = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _seed_project_root(monkeypatch):
    monkeypatch.setenv("PUX_PROJECT_ROOT", str(_PARENT_REPO))


def _violation_rules(org: str) -> list[str]:
    return [v.rule for v in ov.audit_org(org)]


def test_coder_passes_with_data_driven_deny():
    """coder declares ``roster_deny: [general, general-purpose, researcher]``
    in org.yaml AND ``general_purpose_subagent: {enabled: false}`` in
    profile.yaml → both rules pass."""
    codes = _violation_rules("coder")
    assert "roster-deny-enforced" not in codes
    assert "roster-deny-disables-general-purpose" not in codes


def test_roster_deny_enforced_fires_when_denied_slug_in_roster(monkeypatch):
    """If the effective roster contains a denied slug, the rule fires.

    We monkeypatch ``org_agent_slugs`` (the function the checker calls to read
    the effective roster) to inject a denied slug for coder. The checker then
    sees ``general-purpose`` in coder's roster AND in ``roster_deny`` → fires.
    """

    def _fake_slugs(name):
        if name == "coder":
            # Inject a denied slug into the effective roster.
            return ["coder-explorer", "code-worker", "web-agent", "general-purpose"]
        return _orig_slugs(name)

    _orig_slugs = ov.org_agent_slugs
    monkeypatch.setattr(ov, "org_agent_slugs", _fake_slugs)
    codes = _violation_rules("coder")
    assert "roster-deny-enforced" in codes, (
        f"roster-deny-enforced must fire when a denied slug is in the roster; "
        f"got codes={codes}"
    )


def test_roster_deny_disables_gp_fires_when_profile_missing_disabled(monkeypatch):
    """If ``general-purpose`` is in roster_deny but the profile does NOT
    declare ``general_purpose_subagent: {enabled: false}``, the rule fires.

    We monkeypatch ``load_profile`` to return a real config whose GP block is
    None (mimicking a profile that dropped the disabled declaration)."""
    import dataclasses

    _orig_load = profile.load_profile

    def _fake_load(name):
        cfg = _orig_load(name)
        # Strip the GP-disabled declaration — mimic an org that dropped it.
        return dataclasses.replace(cfg, general_purpose_subagent=None)

    monkeypatch.setattr(profile, "load_profile", _fake_load)
    rules = _violation_rules("coder")
    assert "roster-deny-disables-general-purpose" in rules, (
        f"roster-deny-disables-general-purpose must fire when GP is denied but "
        f"the profile doesn't disable it; got rules={rules}"
    )


def test_no_org_name_literals_in_checker_branches():
    """The contract checker has ZERO hard-coded org-name ``if`` branches after
    the data-fication. ``if name == "coder"`` is gone from EXECUTABLE code.
    Comments/docstrings referencing the old form (for archaeology) are allowed;
    we AST-walk and reject any ``If`` node with a string-compare on an org name.

    This is a structural guard against regression — a future hard-coded branch
    would re-introduce the exact dupe pattern the user asked to eliminate."""
    src = pathlib.Path(ov.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden_orgs = {
        "coder", "orchestrator", "deep-research-engine", "game-studio",
        "twitter-agent", "telegram-agent", "invest", "media-studio",
        "video-production", "browser-agent", "social-media-pipeline",
        "web-search", "fs-explorer",
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        # Pattern: <something> == "<org-name>" (or "<org-name>" == <something>)
        for cmp in node.comparators:
            if isinstance(cmp, ast.Constant) and isinstance(cmp.value, str):
                if cmp.value in forbidden_orgs:
                    violations.append(f"line {node.lineno}: compare against {cmp.value!r}")
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            if node.left.value in forbidden_orgs:
                violations.append(f"line {node.lineno}: compare against {node.left.value!r}")
    assert not violations, (
        "hard-coded org-name compare re-introduced in org_validation.py "
        "(must be DATA on the org, not a branch): " + "; ".join(violations)
    )
