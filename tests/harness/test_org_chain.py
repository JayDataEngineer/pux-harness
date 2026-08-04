"""Tests for ``pux org chain`` — inheritance-chain introspection (Fix 2).

Verifies the chain renderer produces the correct extends-chain, per-file
inventory, merge-rules table, and effective supervisor base for:
- An extending org (coder → general)
- A standalone org (general)
- A non-extending org (_demo → general)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pux_harness.agent.org_chain import render_org_chain


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestChainRendering:
    """The extends-chain renders correctly for each org topology."""

    def test_coder_chain_has_general_parent(self) -> None:
        out = render_org_chain("coder", _PROJECT_ROOT)
        assert "Inheritance chain for 'coder'" in out
        assert "general" in out
        assert "coder" in out
        assert "extends: general" in out

    def test_general_is_standalone(self) -> None:
        out = render_org_chain("general", _PROJECT_ROOT)
        assert "no extends — standalone org" in out

    def test_demo_extends_general(self) -> None:
        out = render_org_chain("_demo", _PROJECT_ROOT)
        assert "_demo" in out
        assert "general" in out
        assert "extends: general" in out


class TestFileInventory:
    """The per-org file inventory shows which files exist."""

    def test_coder_has_profile_yaml(self) -> None:
        out = render_org_chain("coder", _PROJECT_ROOT)
        assert "profile.yaml" in out
        assert "present" in out

    def test_general_has_no_profile_yaml(self) -> None:
        out = render_org_chain("general", _PROJECT_ROOT)
        # general/ has no profile.yaml
        general_section = out.split("general/")[1].split("specialists" if "specialists" in out else "===")[0]
        assert "profile.yaml" in general_section
        assert "absent" in general_section

    def test_coder_has_agents(self) -> None:
        out = render_org_chain("coder", _PROJECT_ROOT)
        assert "agents/" in out
        assert "agent(s)" in out


class TestMergeRules:
    """The merge-rules table is present and names every file type."""

    def test_all_file_types_in_rules(self) -> None:
        out = render_org_chain("coder", _PROJECT_ROOT)
        assert "Merge rules by file type" in out
        for ftype in ("AGENTS.md", "profile.yaml", "policy.yaml", "org.yaml", "agents/*.md"):
            assert ftype in out, f"{ftype} missing from merge rules"

    def test_agents_md_rule_is_concatenation(self) -> None:
        out = render_org_chain("coder", _PROJECT_ROOT)
        assert "concatenation" in out

    def test_profile_yaml_rule_is_deep_merge(self) -> None:
        out = render_org_chain("coder", _PROJECT_ROOT)
        assert "deep-merge" in out

    def test_policy_yaml_rule_is_never_inherited(self) -> None:
        out = render_org_chain("coder", _PROJECT_ROOT)
        assert "never inherited" in out


class TestEffectiveBase:
    """The effective supervisor base composition is correct."""

    def test_coder_base_has_three_sources(self) -> None:
        out = render_org_chain("coder", _PROJECT_ROOT)
        assert "Effective supervisor base" in out
        assert "orgs/general/AGENTS.md" in out
        assert "orgs/specialists/coder/AGENTS.md" in out
        assert "orgs/_shared/harness_addendum.md" in out

    def test_general_base_has_two_sources(self) -> None:
        out = render_org_chain("general", _PROJECT_ROOT)
        assert "orgs/general/AGENTS.md" in out
        assert "orgs/_shared/harness_addendum.md" in out
        # general does NOT have a specialist overlay
        assert "specialists/general" not in out


class TestErrorHandling:
    """Bad org names produce clean errors, not crashes."""

    def test_bad_org_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            render_org_chain("does-not-exist", _PROJECT_ROOT)
