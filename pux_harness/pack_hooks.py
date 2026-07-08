"""Pack-time validation hooks — the ``PACK_HOOK_REGISTRY`` (dynamic-tools P4).

A pack is a SHIPPABLE artifact, so it is validated BEFORE the tarball is
written: a syntax-broken agent function or a leaked secret REFUSES the pack
(verify-or-die — a pack that "looks fine" but won't compile or leaks a key is
worse than no pack). Hooks run as an ordered registry over the COLLECTED files
(``pack_org`` calls :func:`run_pack_hooks` after collection, before tar);

This is the thin pux glue ([[rely-on-upstream]]): the heavy lifting is reused
upstream tools — ``gitleaks`` (secret detection) + Python's stdlib ``ast``
(the "ruff-AST" check needs no external binary). Each hook is a small callable;
``PACK_HOOK_REGISTRY`` is the ordered list. P5 reserves a provenance slot
(hooks seed ``manifest.json`` → ``provenance.hooks``, the basis for P5's
``provenance.json``).

Hook contract:
  - ``HookContext`` — what a hook sees (org, org_dir, collected files, manifest).
  - ``HookResult`` — name / ok / skipped / findings. ``ok=False`` REFUSES the
    pack (raises :class:`PackHookError`); ``skipped=True`` records a loud note
    but does not refuse (an absent OPTIONAL tool, never an absent required one).
  - Required hooks (gitleaks, ast) that CANNOT run raise ``ok=False`` with a
    clear install message — there is no silent skip of a security gate
    ([[no-fallbacks-no-aliases]]).
"""
from __future__ import annotations

import ast
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:  # Manifest is only needed for type hints; avoid a hard import cycle.
    from pux_harness.manifest import Manifest
except Exception:  # pragma: no cover - manifest always present in-tree
    Manifest = Any  # type: ignore[assignment, misc]


@dataclass
class HookContext:
    """The collected pack state a hook validates. ``files`` maps archive path
    → host path (everything that will ship, BEFORE the tarball is written)."""

    org: str
    org_dir: Path
    files: dict[str, Path]
    manifest: "Manifest"


@dataclass
class HookResult:
    """Outcome of one hook. ``ok=False`` refuses the pack."""

    name: str
    ok: bool = True
    skipped: bool = False
    findings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "skipped": self.skipped,
            "findings": list(self.findings),
        }


class PackHookError(Exception):
    """Raised when a required hook fails — the pack is REFUSED (no tar written).

    Carries the failing result + every result run so far, so the CLI can print
    the full hook report (not just the first failure)."""

    def __init__(self, result: HookResult, all_results: list[HookResult]):
        self.result = result
        self.all_results = all_results
        findings = "; ".join(result.findings) or "(no detail)"
        super().__init__(f"pack hook {result.name!r} FAILED for: {findings}")


PackHook = Callable[[HookContext], HookResult]


# ---------------------------------------------------------------------------
# Hook 1 — AST / syntax check (stdlib ``ast``; the "ruff-AST" gate, no binary)
# ---------------------------------------------------------------------------

def ast_check_hook(ctx: HookContext) -> HookResult:
    """Every collected ``.py`` must parse (``ast.compile`` in 'exec' mode).

    Targets agent-authored + graduated functions (``lib/functions/``,
    ``sandbox/functions/``) AND any org-source ``.py`` — a shipped module that
    won't compile is a broken pack. Uses stdlib ``ast`` (no ruff binary needed);
    a SyntaxError is a hard fail (the agent retries authoring — see
    ``dynamic.make_function``'s compile() gate)."""
    findings: list[str] = []
    checked = 0
    for archive_path, host_path in sorted(ctx.files.items()):
        if not archive_path.endswith(".py"):
            continue
        # Skip the vendored runtime kit (pux_harness/kit/*) — it is the trusted
        # harness itself, already compile-checked at install time.
        if "pux_harness/kit/" in archive_path or archive_path.startswith("pux_harness/"):
            continue
        try:
            src = host_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            findings.append(f"{archive_path}: unreadable ({exc})")
            continue
        checked += 1
        try:
            compile(src, str(host_path), "exec")
        except SyntaxError as exc:
            findings.append(
                f"{archive_path}:{exc.lineno}: SyntaxError: {exc.msg}"
            )
    return HookResult(
        name="ast_check",
        ok=not findings,
        findings=findings,
        details={"checked": checked},
    )


# ---------------------------------------------------------------------------
# Hook 2 — gitleaks secret scan (host-side ``gitleaks`` CLI; reuse-first)
# ---------------------------------------------------------------------------

def _gitleaks_available(runner: Callable | None = None) -> bool:
    """Is the ``gitleaks`` binary on PATH? (Injectable runner for tests.)"""
    run = runner or subprocess.run
    try:
        out = run(
            ["gitleaks", "version"],
            capture_output=True, text=True, timeout=10.0, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False
    return getattr(out, "returncode", 1) == 0


def _run_gitleaks(
    staging: Path, *, runner: Callable | None = None, timeout: float = 120.0
) -> tuple[int, list[dict[str, Any]]]:
    """Run ``gitleaks detect --no-git`` over ``staging``; return (exit_code,
    findings). Exit 0 = clean, 1 = leaks found, other = tool error. Findings
    parsed from the JSON report (authoritative over exit code)."""
    run = runner or subprocess.run
    report = staging / ".gitleaks-report.json"
    cmd = [
        "gitleaks", "detect",
        "--source", str(staging),
        "--no-git",
        "--report-format", "json",
        "--report-path", str(report),
        "--no-banner",
        "--exit-code", "1",
    ]
    try:
        out = run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return 127, []
    code = getattr(out, "returncode", 1)
    findings: list[dict[str, Any]] = []
    if report.is_file():
        try:
            data = json.loads(report.read_text() or "[]")
            if isinstance(data, list):
                findings = data
        except (OSError, json.JSONDecodeError):
            findings = []
    return code, findings


def _stage_files(ctx: HookContext, staging: Path) -> dict[str, str]:
    """Copy collected files into ``staging`` mirroring archive paths (so the
    gitleaks finding's ``File`` maps back to a shipped path). Returns
    ``{abs_staged_path: archive_path}`` for back-mapping. Skips unreadable."""
    mapping: dict[str, str] = {}
    for archive_path, host_path in sorted(ctx.files.items()):
        # Don't scan the vendored kit — trusted harness source, not org content.
        if "pux_harness/kit/" in archive_path or archive_path.startswith("pux_harness/"):
            continue
        dest = staging / archive_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(host_path.read_bytes())
        except (OSError, PermissionError):
            continue
        mapping[str(dest)] = archive_path
    return mapping


def gitleaks_hook(
    ctx: HookContext, *, runner: Callable | None = None
) -> HookResult:
    """Scan every shipped file for secrets via ``gitleaks detect --no-git``.

    REQUIRED: if ``gitleaks`` is absent the pack is REFUSED with a clear install
    message (no silent skip of a security gate — the whole point of P4 is that
    a leaked key refuses the pack). Leaks → ``ok=False`` with per-file findings.

    Detection is **pattern + entropy driven** (gitleaks' rule set): provider
    PATs (``ghp_…``, ``sk-…``-shopify, etc.) and high-entropy values assigned to
    key-like names are caught; a contrived low-entropy string may pass. That is
    the right tradeoff for a secret scanner — REAL leaks carry tell-tale prefixes
    and high entropy, which is exactly what the rules key on. The
    ``data/``/``.pux/`` HARD_EXCLUDE is the PRIMARY secret boundary; this hook is
    the defense-in-depth scan over what DOES ship (org source + lib/functions)."""
    if not _gitleaks_available(runner=runner):
        return HookResult(
            name="gitleaks",
            ok=False,
            findings=["gitleaks binary not found on PATH — install it to pack "
                      "(P4 secrets gate; reuse-first via the gitleaks CLI)"],
        )
    with tempfile.TemporaryDirectory(prefix="pux-pack-gitleaks-") as tmp:
        staging = Path(tmp)
        mapping = _stage_files(ctx, staging)
        if not mapping:
            return HookResult(name="gitleaks", ok=True, details={"scanned": 0})
        code, findings = _run_gitleaks(staging, runner=runner)
    # exit 0 = clean; 1 = leaks (expected); anything else = tool failure.
    if code not in (0, 1):
        return HookResult(
            name="gitleaks",
            ok=False,
            findings=[f"gitleaks exited {code} (tool error) — cannot validate"],
            details={"scanned": len(mapping), "exit_code": code},
        )
    # Back-map staged paths → archive paths for a legible report.
    pretty: list[str] = []
    for f in findings:
        staged = f.get("File", "")
        arc = mapping.get(staged, Path(staged).name if staged else "?")
        rule = f.get("RuleID", "?")
        line = f.get("StartLine", "?")
        pretty.append(f"{arc}:{line} [{rule}]")
    return HookResult(
        name="gitleaks",
        ok=not pretty,
        findings=pretty,
        details={"scanned": len(mapping), "leaks": len(pretty)},
    )


# ---------------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------------

#: The ordered pack-time validation pipeline. Each hook sees the collected
#: files; the first ``ok=False`` refuses the pack. Add P5's provenance hook here.
PACK_HOOK_REGISTRY: list[PackHook] = [
    ast_check_hook,
    gitleaks_hook,
]


def run_pack_hooks(
    ctx: HookContext, *, registry: list[PackHook] | None = None,
    gitleaks_runner: Callable | None = None,
) -> list[HookResult]:
    """Run every hook in order. Returns all results on success; raises
    :class:`PackHookError` on the first failure (carrying all results so far).

    ``gitleaks_runner`` injects the subprocess runner into the gitleaks hook
    (offline/deterministic tests). Hooks that don't accept ``runner`` ignore it.
    """
    hooks = registry if registry is not None else PACK_HOOK_REGISTRY
    results: list[HookResult] = []
    for hook in hooks:
        try:
            if hook is gitleaks_hook and gitleaks_runner is not None:
                res = hook(ctx, runner=gitleaks_runner)
            else:
                res = hook(ctx)
        except Exception as exc:  # a hook bug must NOT silently pass the pack
            res = HookResult(
                name=getattr(hook, "__name__", str(hook)),
                ok=False,
                findings=[f"hook raised: {exc!r}"],
            )
        results.append(res)
        if not res.ok:
            raise PackHookError(res, results)
    return results


def provenance_from_results(results: list[HookResult]) -> dict[str, Any]:
    """Fold hook results into the manifest's ``provenance.hooks`` block — the
    audit surface (P5 turns this into a standalone ``provenance.json`` with
    SHA-256 layer digests)."""
    return {
        "hooks": [r.to_dict() for r in results],
        "all_ok": all(r.ok for r in results) if results else True,
    }
