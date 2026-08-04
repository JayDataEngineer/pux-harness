"""Tests for ``pux prompt show`` — the static prompt introspection renderer (D8).

Verifies that the renderer walks the SAME part registries as ``assemble_prompt``,
produces the correct active/conditional classification, and that the raw output
matches what ``assemble_prompt`` would emit at runtime (for the static case).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pux_harness.agent.prompt_parts import (
    PromptCtx,
    PromptScope,
    assemble_prompt,
)
from pux_harness.agent.prompt_show import (
    format_prompt_raw,
    render_parts,
    show_subagent,
    show_supervisor,
)


# Repo root (the parent of the pux-harness package dir = the dir containing orgs/).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestSupervisorRendering:
    """The supervisor (CTO) prompt renders correctly."""

    def test_coder_supervisor_has_four_parts(self) -> None:
        """Coder's supervisor prompt has exactly 4 parts (2 active, 2 conditional)."""
        from pux_harness.agent.prompt_show import _build_supervisor_ctx

        ctx = _build_supervisor_ctx("coder", _PROJECT_ROOT)
        parts = render_parts(PromptScope.SUPERVISOR, ctx)
        assert len(parts) == 4
        names = [p.name for p in parts]
        assert names == [
            "agents_md_core",
            "org_system_prompt_suffix",
            "ask_user_suffix",
            "dynamic_dispatch_suffix",
        ]

    def test_coder_active_parts_have_content(self) -> None:
        """Parts 1 (base+addendum) and 2 (suffix) are active for coder."""
        from pux_harness.agent.prompt_show import _build_supervisor_ctx

        ctx = _build_supervisor_ctx("coder", _PROJECT_ROOT)
        parts = render_parts(PromptScope.SUPERVISOR, ctx)
        # Part 1: always-on base + addendum
        assert parts[0].content is not None
        assert len(parts[0].content) > 1000  # substantial prompt
        # Part 2: coder has a system_prompt_suffix in profile.yaml
        assert parts[1].content is not None
        assert "verify" in parts[1].content.lower()
        # Parts 3-4: conditional, off at static time
        assert parts[2].content is None  # ask_user
        assert parts[3].content is None  # dynamic_dispatch

    def test_raw_matches_assemble_prompt(self) -> None:
        """The raw output (active parts joined) matches assemble_prompt for the
        same ctx — proving the renderer uses the SAME assembly logic."""
        from pux_harness.agent.prompt_show import _build_supervisor_ctx

        ctx = _build_supervisor_ctx("coder", _PROJECT_ROOT)
        parts = render_parts(PromptScope.SUPERVISOR, ctx)
        raw = format_prompt_raw(parts)
        # assemble_prompt with the same ctx produces the same string
        from pux_harness.agent.prompt_parts import SUPERVISOR_PROMPT_PARTS

        assembled = assemble_prompt(SUPERVISOR_PROMPT_PARTS, ctx, PromptScope.SUPERVISOR)
        assert raw == assembled

    def test_general_org_no_suffix(self) -> None:
        """The 'general' org has no profile.yaml suffix — part 2 is inactive."""
        from pux_harness.agent.prompt_show import _build_supervisor_ctx

        ctx = _build_supervisor_ctx("general", _PROJECT_ROOT)
        parts = render_parts(PromptScope.SUPERVISOR, ctx)
        # Part 1 always active
        assert parts[0].content is not None
        # Part 2: general has no system_prompt_suffix
        assert parts[1].content is None


class TestSubagentRendering:
    """The subagent prompt renders correctly."""

    def test_coder_explorer_has_three_parts(self) -> None:
        """A subagent has exactly 3 parts."""
        parts = show_subagent("coder", "coder-explorer", _PROJECT_ROOT, raw=False)
        assert "part 1/3" in parts
        assert "part 2/3" in parts
        assert "part 3/3" in parts

    def test_coder_explorer_body_is_substantial(self) -> None:
        """The agent body (part 1) contains the .md body content."""
        from pux_harness.agent.prompt_show import _build_subagent_ctx

        ctx, error = _build_subagent_ctx("coder", "coder-explorer", _PROJECT_ROOT)
        assert error is None
        parts = render_parts(PromptScope.SUBAGENT, ctx)
        assert parts[0].content is not None
        # The coder-explorer body should mention exploration / codebase
        body_lower = parts[0].content.lower()
        assert "explorer" in body_lower or "codebase" in body_lower

    def test_bad_slug_returns_error(self) -> None:
        """An unknown slug produces a clean error, not a crash."""
        result = show_subagent("coder", "does-not-exist", _PROJECT_ROOT)
        assert "ERROR" in result
        assert "does-not-exist" in result


class TestProvenanceLabels:
    """Every part has a human-readable source and condition label."""

    def test_all_supervisor_parts_have_provenance(self) -> None:
        from pux_harness.agent.prompt_show import _build_supervisor_ctx

        ctx = _build_supervisor_ctx("coder", _PROJECT_ROOT)
        parts = render_parts(PromptScope.SUPERVISOR, ctx)
        for part in parts:
            assert part.source, f"{part.name} has no source label"
            assert part.condition, f"{part.name} has no condition label"
            assert part.content is not None or len(part.condition) > 10, (
                f"{part.name} is inactive but has no condition explanation"
            )

    def test_all_subagent_parts_have_provenance(self) -> None:
        from pux_harness.agent.prompt_show import _build_subagent_ctx

        ctx, _ = _build_subagent_ctx("coder", "coder-explorer", _PROJECT_ROOT)
        parts = render_parts(PromptScope.SUBAGENT, ctx)
        for part in parts:
            assert part.source, f"{part.name} has no source label"
            assert part.condition, f"{part.name} has no condition label"


class TestOutputFormat:
    """The formatted output is human-readable and well-structured."""

    def test_provenance_output_has_headers(self) -> None:
        out = show_supervisor("coder", _PROJECT_ROOT, raw=False)
        assert "=== SUPERVISOR" in out
        assert "=== TOTAL:" in out
        assert "part 1/4" in out
        assert "part 4/4" in out

    def test_raw_output_is_just_text(self) -> None:
        out = show_supervisor("coder", _PROJECT_ROOT, raw=True)
        # No provenance headers in raw mode
        assert "=== TOTAL" not in out
        assert "part 1/4" not in out
        # Should start with the base prompt content
        assert len(out) > 1000

    def test_conditional_parts_marked(self) -> None:
        out = show_supervisor("coder", _PROJECT_ROOT, raw=False)
        assert "CONDITIONAL" in out
        assert "not emitted at static time" in out


class TestAddendumFromFile:
    """Fix 1: the harness addendum is loaded from orgs/_shared/harness_addendum.md,
    not just from the embedded Python constant."""

    def test_addendum_file_exists(self) -> None:
        """The file exists in the orgs tree (experimenters can edit it)."""
        addendum_path = _PROJECT_ROOT / "orgs" / "_shared" / "harness_addendum.md"
        assert addendum_path.is_file(), f"harness_addendum.md missing at {addendum_path}"

    def test_file_matches_embedded_constant(self) -> None:
        """The file body + seam == the embedded _ADDENDUM (byte-identity)."""
        from pux_harness.agent.prompt_parts import _ADDENDUM, load_harness_addendum

        file_version = load_harness_addendum(_PROJECT_ROOT)
        assert file_version == _ADDENDUM, (
            "orgs/_shared/harness_addendum.md body does not match the embedded "
            "_ADDENDUM — the file was edited or the constant drifted. Run "
            "`pux prompt show` to verify the addendum renders correctly."
        )

    def test_source_label_names_the_file(self) -> None:
        """The provenance label for agents_md_core names the FILE, not just the constant."""
        out = show_supervisor("coder", _PROJECT_ROOT, raw=False)
        assert "orgs/_shared/harness_addendum.md" in out


class TestAskUserFromFile:
    """The ask-user suffix is loaded from orgs/_shared/ask_user_suffix.md,
    not just from the embedded Python constant (same lift as the addendum)."""

    def test_ask_user_file_exists(self) -> None:
        """The file exists in the orgs tree (experimenters can edit it)."""
        path = _PROJECT_ROOT / "orgs" / "_shared" / "ask_user_suffix.md"
        assert path.is_file(), f"ask_user_suffix.md missing at {path}"

    def test_file_matches_embedded_constant(self) -> None:
        """The file body == the embedded ASK_USER_PROMPT_SUFFIX (byte-identity)."""
        from pux_harness.agent.hitl import ASK_USER_PROMPT_SUFFIX
        from pux_harness.agent.prompt_parts import load_ask_user_suffix

        file_version = load_ask_user_suffix(_PROJECT_ROOT)
        assert file_version == ASK_USER_PROMPT_SUFFIX, (
            "orgs/_shared/ask_user_suffix.md body does not match the embedded "
            "ASK_USER_PROMPT_SUFFIX — the file was edited or the constant drifted."
        )

    def test_source_label_names_the_file(self) -> None:
        """The provenance label for ask_user_suffix names the FILE."""
        out = show_supervisor("coder", _PROJECT_ROOT, raw=False)
        assert "orgs/_shared/ask_user_suffix.md" in out


class TestDynamicDispatchFromFile:
    """The dynamic-dispatch suffix is loaded from
    orgs/_shared/dynamic_dispatch_suffix.md, not just from the embedded Python
    constant (same lift as the addendum)."""

    def test_dynamic_dispatch_file_exists(self) -> None:
        """The file exists in the orgs tree (experimenters can edit it)."""
        path = _PROJECT_ROOT / "orgs" / "_shared" / "dynamic_dispatch_suffix.md"
        assert path.is_file(), f"dynamic_dispatch_suffix.md missing at {path}"

    def test_file_matches_embedded_constant(self) -> None:
        """The file body == the embedded _DYNAMIC_DISPATCH_SUFFIX (byte-identity)."""
        from pux_harness.agent.prompt_parts import (
            _DYNAMIC_DISPATCH_SUFFIX,
            load_dynamic_dispatch_suffix,
        )

        file_version = load_dynamic_dispatch_suffix(_PROJECT_ROOT)
        assert file_version == _DYNAMIC_DISPATCH_SUFFIX, (
            "orgs/_shared/dynamic_dispatch_suffix.md body does not match the "
            "embedded _DYNAMIC_DISPATCH_SUFFIX — the file was edited or the "
            "constant drifted."
        )

    def test_source_label_names_the_file(self) -> None:
        """The provenance label for dynamic_dispatch_suffix names the FILE."""
        out = show_supervisor("coder", _PROJECT_ROOT, raw=False)
        assert "orgs/_shared/dynamic_dispatch_suffix.md" in out


class TestSimulateFlags:
    """Fix 3: --with-ask-user / --with-interpreter let the experimenter preview
    the conditional parts as if they were runtime-active."""

    def test_with_ask_user_makes_part_active(self) -> None:
        """The ask_user part renders ACTIVE when ask_user=True."""
        out = show_supervisor("coder", _PROJECT_ROOT, ask_user=True)
        assert "ask_user_suffix" in out
        # The part should be ACTIVE (not CONDITIONAL)
        lines = out.split("\n")
        ask_user_status = [
            l for l in lines if "ask_user active AND turn-based" in l
        ]
        assert ask_user_status, "ask_user status line not found"
        assert "ACTIVE" in ask_user_status[0], (
            f"ask_user should be ACTIVE with ask_user=True, got: {ask_user_status[0]}"
        )

    def test_with_interpreter_makes_part_active(self) -> None:
        """The dynamic_dispatch part renders ACTIVE when interpreter=True."""
        out = show_supervisor("coder", _PROJECT_ROOT, interpreter=True)
        lines = out.split("\n")
        dispatch_status = [
            l for l in lines if "CodeInterpreterMiddleware mounted" in l
        ]
        assert dispatch_status, "dynamic_dispatch status line not found"
        assert "ACTIVE" in dispatch_status[0], (
            f"dynamic_dispatch should be ACTIVE with interpreter=True, got: {dispatch_status[0]}"
        )

    def test_both_flags_make_all_active(self) -> None:
        """With both flags, all 4 parts are ACTIVE (0 conditional)."""
        out = show_supervisor(
            "coder", _PROJECT_ROOT, ask_user=True, interpreter=True
        )
        assert "4 active, 0 conditional" in out

    def test_default_keeps_them_conditional(self) -> None:
        """Without flags, the parts stay CONDITIONAL (the default, unchanged)."""
        out = show_supervisor("coder", _PROJECT_ROOT)
        assert "2 conditional" in out or "1 conditional" in out


class TestExtraPromptParts:
    """Fix 4: extra_prompt_parts in profile.yaml appends always-on file-sourced parts."""

    def test_no_extras_for_coder(self) -> None:
        """Coder has no extra_prompt_parts — the registry is just the 4 built-in parts."""
        from pux_harness.agent.prompt_show import _resolve_extra_parts

        extras = _resolve_extra_parts("coder", _PROJECT_ROOT, PromptScope.SUPERVISOR)
        assert extras == (), f"coder should have no extras, got {[p.name for p in extras]}"

    def test_build_extra_parts_with_valid_entries(self, tmp_path) -> None:
        """build_extra_parts constructs PromptPartSpec from a list of entries."""
        from pux_harness.agent.prompt_parts import build_extra_parts, PromptScope

        org_dir = tmp_path / "myorg"
        (org_dir / "extra").mkdir(parents=True)
        (org_dir / "extra" / "section.md").write_text("## Extra section\nContent here.")
        entries = [
            {"name": "my_section", "file": "extra/section.md", "scope": ["supervisor"]},
        ]
        parts = build_extra_parts(entries, "myorg", org_dir, PromptScope.SUPERVISOR)
        assert len(parts) == 1
        assert parts[0].name == "my_section"
        # The builder returns the file content
        ctx = PromptCtx()
        content = parts[0].build(ctx)
        assert "Extra section" in content

    def test_build_extra_parts_scope_filter(self, tmp_path) -> None:
        """Scope filtering: a supervisor-only entry is skipped for subagent scope."""
        from pux_harness.agent.prompt_parts import build_extra_parts, PromptScope

        org_dir = tmp_path / "myorg"
        (org_dir / "extra").mkdir(parents=True)
        (org_dir / "extra" / "sup.md").write_text("supervisor only")
        (org_dir / "extra" / "both.md").write_text("both scopes")
        entries = [
            {"name": "sup_only", "file": "extra/sup.md", "scope": ["supervisor"]},
            {"name": "both_scopes", "file": "extra/both.md", "scope": ["supervisor", "subagent"]},
        ]
        sup_parts = build_extra_parts(entries, "myorg", org_dir, PromptScope.SUPERVISOR)
        sub_parts = build_extra_parts(entries, "myorg", org_dir, PromptScope.SUBAGENT)
        assert len(sup_parts) == 2
        assert len(sub_parts) == 1
        assert sub_parts[0].name == "both_scopes"

    def test_build_extra_parts_missing_file_raises(self, tmp_path) -> None:
        """A missing file raises FileNotFoundError with a helpful message."""
        from pux_harness.agent.prompt_parts import build_extra_parts, PromptScope

        entries = [{"name": "missing", "file": "nonexistent.md"}]
        import pytest

        with pytest.raises(FileNotFoundError, match="nonexistent.md"):
            build_extra_parts(entries, "myorg", tmp_path, PromptScope.SUPERVISOR)

    def test_build_extra_parts_bad_entry_raises(self, tmp_path) -> None:
        """An entry missing name/file raises ValueError."""
        from pux_harness.agent.prompt_parts import build_extra_parts, PromptScope

        import pytest

        with pytest.raises(ValueError, match="needs .name. and .file"):
            build_extra_parts([{"name": "no_file"}], "myorg", tmp_path, PromptScope.SUPERVISOR)
