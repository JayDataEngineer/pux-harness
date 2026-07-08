"""Tests for ``pux_harness.pack_hooks`` — the PACK_HOOK_REGISTRY (dynamic-tools P4).

Hooks gate the pack BEFORE the tarball is written: a syntax-broken agent function
or a leaked secret REFUSES the pack. The gitleaks hook's subprocess is INJECTED
(``gitleaks_runner``) so the scan LOGIC is proven deterministically offline; the
real-gitleaks LIVE proof (fake key → pack_org refuses) lives in
``tests/export/test_export_hooks.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pux_harness.pack_hooks import (
    PACK_HOOK_REGISTRY,
    HookContext,
    HookResult,
    PackHookError,
    ast_check_hook,
    gitleaks_hook,
    provenance_from_results,
    run_pack_hooks,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _ctx(files: dict[str, str], tmp_path: Path, org: str = "acme") -> HookContext:
    """Build a HookContext from ``{archive_path: text}`` (no full org tree — the
    hooks only consume the collected files dict)."""
    paths: dict[str, Path] = {}
    for arc, content in files.items():
        p = tmp_path / arc
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        paths[arc] = p
    return HookContext(org=org, org_dir=tmp_path, files=paths, manifest=None)


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeGitleaks:
    """``subprocess.run``-shaped fake for the gitleaks hook.

    - ``absent``  → ``gitleaks version`` rc=1 (binary "missing") → required-gate refuse.
    - ``leak``    → ``detect`` emits a finding for the first staged file → refuse.
    - ``tool_error`` → ``detect`` rc=2 (neither clean nor leaks) → refuse.
    """

    def __init__(self, *, leak: bool = False, absent: bool = False, tool_error: bool = False):
        self.leak = leak
        self.absent = absent
        self.tool_error = tool_error
        self.detect_calls = 0

    def __call__(self, cmd, **kw):
        if cmd[:2] == ["gitleaks", "version"]:
            return _FakeProc(1 if self.absent else 0, "8.30.0")
        if cmd[:2] == ["gitleaks", "detect"]:
            self.detect_calls += 1
            if self.tool_error:
                return _FakeProc(2)
            source = cmd[cmd.index("--source") + 1]
            rpath = cmd[cmd.index("--report-path") + 1]
            findings = []
            if self.leak:
                # Emit a finding for the first real staged file so the back-map
                # (staged path → archive path) is exercised.
                for p in sorted(Path(source).rglob("*")):
                    if p.is_file() and p.name != ".gitleaks-report.json":
                        findings.append({
                            "File": str(p),
                            "RuleID": "generic-api-key",
                            "StartLine": 7,
                        })
                        break
            Path(rpath).write_text(json.dumps(findings))
            return _FakeProc(1 if findings else 0)
        return _FakeProc(1)


# ===========================================================================
# ast_check_hook
# ===========================================================================

class TestAstCheckHook:
    def test_clean_py_passes(self, tmp_path):
        ctx = _ctx({"lib/functions/good.py": "def run():\n    return 1\n"}, tmp_path)
        res = ast_check_hook(ctx)
        assert res.ok is True
        assert res.details["checked"] == 1
        assert res.findings == []

    def test_broken_py_refuses_with_line(self, tmp_path):
        ctx = _ctx({"lib/functions/bad.py": "def run(\n    pass\n"}, tmp_path)
        res = ast_check_hook(ctx)
        assert res.ok is False
        assert len(res.findings) == 1
        f = res.findings[0]
        assert "lib/functions/bad.py" in f
        assert "SyntaxError" in f

    def test_ignores_non_py(self, tmp_path):
        ctx = _ctx({"README.md": "# not python", "good.py": "x = 1\n"}, tmp_path)
        res = ast_check_hook(ctx)
        assert res.ok is True
        assert res.details["checked"] == 1

    def test_no_py_files_is_clean(self, tmp_path):
        ctx = _ctx({"a.md": "x", "b.yaml": "y: 1\n"}, tmp_path)
        res = ast_check_hook(ctx)
        assert res.ok is True
        assert res.details["checked"] == 0

    def test_vendored_kit_is_skipped(self, tmp_path):
        # The vendored runtime kit (pux_harness/*) is trusted harness source —
        # a broken kit file must NOT fail the org's pack (it is not org content).
        ctx = _ctx({"pux_harness/kit/broken.py": "def ((", "lib/ok.py": "y = 2\n"}, tmp_path)
        res = ast_check_hook(ctx)
        assert res.ok is True
        assert res.details["checked"] == 1

    def test_unreadable_py_refuses(self, tmp_path):
        p = tmp_path / "weird.py"
        p.write_bytes(b"\xff\xfe not utf8")
        ctx = HookContext(org="acme", org_dir=tmp_path,
                          files={"weird.py": p}, manifest=None)
        res = ast_check_hook(ctx)
        assert res.ok is False
        assert "unreadable" in res.findings[0]


# ===========================================================================
# gitleaks_hook (logic via injected runner)
# ===========================================================================

class TestGitleaksHook:
    def test_clean_passes(self, tmp_path):
        ctx = _ctx({"lib/functions/fn.py": "def run():\n    return 1\n"}, tmp_path)
        res = gitleaks_hook(ctx, runner=_FakeGitleaks(leak=False))
        assert res.ok is True
        assert res.details["scanned"] >= 1
        assert res.findings == []

    def test_leak_refuses_with_backmapped_path(self, tmp_path):
        ctx = _ctx({"lib/functions/fn.py": "def run():\n    return 1\n"}, tmp_path)
        res = gitleaks_hook(ctx, runner=_FakeGitleaks(leak=True))
        assert res.ok is False
        assert len(res.findings) == 1
        f = res.findings[0]
        # The finding is back-mapped to the SHIPPED archive path (not the tmp
        # staging path) + carries the rule id + line.
        assert "lib/functions/fn.py" in f
        assert "generic-api-key" in f

    def test_absent_binary_refuses_no_detect_call(self, tmp_path):
        # REQUIRED gate: an absent gitleaks REFUSES the pack (no silent skip).
        fake = _FakeGitleaks(absent=True)
        ctx = _ctx({"lib/fn.py": "x = 1\n"}, tmp_path)
        res = gitleaks_hook(ctx, runner=fake)
        assert res.ok is False
        assert any("not found" in s for s in res.findings)
        assert fake.detect_calls == 0  # availability failed → no scan attempted

    def test_tool_error_refuses(self, tmp_path):
        ctx = _ctx({"lib/fn.py": "x = 1\n"}, tmp_path)
        res = gitleaks_hook(ctx, runner=_FakeGitleaks(tool_error=True))
        assert res.ok is False
        assert any("tool error" in s for s in res.findings)

    def test_no_files_is_clean(self, tmp_path):
        ctx = HookContext(org="acme", org_dir=tmp_path, files={}, manifest=None)
        res = gitleaks_hook(ctx, runner=_FakeGitleaks())
        assert res.ok is True
        assert res.details["scanned"] == 0

    def test_vendored_kit_not_scanned(self, tmp_path):
        # Only the kit path is present → nothing staged → clean (kit is trusted).
        ctx = _ctx({"pux_harness/kit/x.py": "api_key = 'sk-leak'\n"}, tmp_path)
        res = gitleaks_hook(ctx, runner=_FakeGitleaks(leak=True))
        assert res.ok is True
        assert res.details["scanned"] == 0


# ===========================================================================
# run_pack_hooks (registry orchestration)
# ===========================================================================

class TestRunPackHooks:
    def test_registry_is_ast_then_gitleaks(self):
        # The ordered pipeline — ast is cheapest + first (fail fast on syntax).
        assert PACK_HOOK_REGISTRY == [ast_check_hook, gitleaks_hook]

    def test_runs_all_in_order_when_clean(self, tmp_path):
        ctx = _ctx({"lib/fn.py": "def run():\n    return 1\n"}, tmp_path)
        results = run_pack_hooks(ctx, gitleaks_runner=_FakeGitleaks())
        assert [r.name for r in results] == ["ast_check", "gitleaks"]
        assert all(r.ok for r in results)

    def test_ast_failure_aborts_before_gitleaks(self, tmp_path):
        # Broken AST → ast_check fails FIRST; gitleaks never runs.
        ctx = _ctx({"lib/bad.py": "def (\n"}, tmp_path)
        fake = _FakeGitleaks()
        with pytest.raises(PackHookError) as ei:
            run_pack_hooks(ctx, gitleaks_runner=fake)
        assert ei.value.result.name == "ast_check"
        assert [r.name for r in ei.value.all_results] == ["ast_check"]
        assert fake.detect_calls == 0  # gitleaks never reached

    def test_gitleaks_failure_after_clean_ast(self, tmp_path):
        # Clean AST + a leak → gitleaks fails; both results carried.
        ctx = _ctx({"lib/fn.py": "def run():\n    return 1\n"}, tmp_path)
        with pytest.raises(PackHookError) as ei:
            run_pack_hooks(ctx, gitleaks_runner=_FakeGitleaks(leak=True))
        assert ei.value.result.name == "gitleaks"
        assert [r.name for r in ei.value.all_results] == ["ast_check", "gitleaks"]

    def test_empty_registry_returns_empty(self, tmp_path):
        ctx = _ctx({"lib/fn.py": "def (\n"}, tmp_path)  # would fail ast
        assert run_pack_hooks(ctx, registry=[]) == []

    def test_custom_registry(self, tmp_path):
        # An operator-supplied registry (e.g. ast-only) is honored.
        ctx = _ctx({"lib/bad.py": "def (\n"}, tmp_path)
        with pytest.raises(PackHookError) as ei:
            run_pack_hooks(ctx, registry=[ast_check_hook])
        assert ei.value.result.name == "ast_check"

    def test_hook_raising_exception_is_caught_as_failure(self, tmp_path):
        def boom(ctx):
            raise RuntimeError("hook impl bug")
        ctx = _ctx({"lib/fn.py": "x = 1\n"}, tmp_path)
        with pytest.raises(PackHookError) as ei:
            run_pack_hooks(ctx, registry=[boom])
        # A hook bug must NOT silently pass the pack.
        assert ei.value.result.ok is False
        assert "hook raised" in ei.value.result.findings[0]

    def test_absent_gitleaks_refuses_via_runner(self, tmp_path):
        # The required-gate refusal surfaces through run_pack_hooks too.
        ctx = _ctx({"lib/fn.py": "x = 1\n"}, tmp_path)
        with pytest.raises(PackHookError) as ei:
            run_pack_hooks(ctx, gitleaks_runner=_FakeGitleaks(absent=True))
        assert ei.value.result.name == "gitleaks"


# ===========================================================================
# provenance_from_results
# ===========================================================================

class TestProvenance:
    def test_shape_all_ok(self):
        results = [
            HookResult(name="ast_check", ok=True),
            HookResult(name="gitleaks", ok=True, details={"scanned": 3}),
        ]
        prov = provenance_from_results(results)
        assert prov["all_ok"] is True
        assert [h["name"] for h in prov["hooks"]] == ["ast_check", "gitleaks"]
        assert prov["hooks"][0]["ok"] is True

    def test_empty_results(self):
        prov = provenance_from_results([])
        assert prov == {"hooks": [], "all_ok": True}

    def test_failed_marks_all_ok_false(self):
        results = [HookResult(name="ast_check", ok=False, findings=["x"])]
        prov = provenance_from_results(results)
        assert prov["all_ok"] is False
        assert prov["hooks"][0]["findings"] == ["x"]
