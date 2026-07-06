"""Host-side prep hooks — run BEFORE ``SandboxContainer.create()`` (so before
``policy.validate_env``). Each hook captures its ``helper_script``'s stdout into
the env vars named in ``exports`` (value ``stdout`` -> captured stdout), which
then flow through the existing ``credentials.required`` / ``browser.cookies_env``
/ ``policy.env_vars`` / in-image ``seed-cookies.sh`` chain UNCHANGED — one
mechanism, not two.

Self-contained: stdlib + ``subprocess`` only; imports ``policy`` for the
dataclass (no Docker, no ``pux_harness.agent`` -> stays importable under
``--check-contract`` / offline). The harness process env (``os.environ``) is the
sink — ``container.create()`` does ``os.environ.update(run_host_setup(...))``
before ``validate_env`` so the existing cred/cookie chain consumes the exports
with no signature change.

``python_deps`` install into a cached per-hook uv venv at
``<project>/.pux/venvs/<name>/`` (gitignored, runner-owned). The cache is keyed
on the dep LIST — editing ``python_deps`` invalidates it (the ``.installed``
marker stores the deps, not just a flag). No silent skip on failure
(no-fallbacks rule): a hook that errors raises ``HostSetupError`` loud.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence

from pux_harness.sandbox import policy


class HostSetupError(Exception):
    """A host_setup hook failed: missing/unnamed hook, bad export target,
    missing helper script, or a non-zero subprocess exit. Raised loud — a host
    that opts into host_setup is opting INTO prep, so a failure is not a silent
    skip."""


# Allowed values in a hook's ``exports`` mapping. Today only ``stdout`` (capture
# the helper script's stdout into the named env var). A closed set so a typo'd
# source (``stderr``/``file:...``) fails loud instead of silently producing an
# empty export.
_EXPORT_SOURCES = frozenset({"stdout"})


def _run(
    cmd: Sequence[str], cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Thin subprocess wrapper — tests monkeypatch THIS (not ``subprocess.run``)
    so the runner is exercised for real with no uv/venv/network."""
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _venv_dir(project_root: Path, name: str) -> Path:
    return project_root / ".pux" / "venvs" / name


def _ensure_venv(hook: policy.HostSetupHook, project_root: Path) -> Path:
    """Create + populate the per-hook venv (cached on the dep list). Returns the
    venv's ``bin/python`` path. Idempotent: a venv whose ``.installed`` marker
    matches the current dep list skips both create + install. Deps changed ->
    marker mismatch -> recreate + reinstall + refresh marker."""
    venv = _venv_dir(project_root, hook.name)
    python = venv / "bin" / "python"
    marker = venv / ".installed"
    expected = "\n".join(hook.python_deps) + "\n"
    if python.is_file() and marker.is_file() and marker.read_text() == expected:
        return python
    venv.mkdir(parents=True, exist_ok=True)
    # Create the venv (uv is on PATH — the harness runs under it).
    cp = _run(["uv", "venv", "--python", "3.12", str(venv)], project_root, dict(os.environ))
    if cp.returncode != 0:
        raise HostSetupError(
            f"host_setup {hook.name!r}: uv venv failed (rc={cp.returncode}): "
            f"{cp.stderr.strip()}"
        )
    if hook.python_deps:
        cp = _run(
            ["uv", "pip", "install", "--python", str(python), *hook.python_deps],
            project_root,
            dict(os.environ),
        )
        if cp.returncode != 0:
            raise HostSetupError(
                f"host_setup {hook.name!r}: uv pip install failed (rc={cp.returncode}): "
                f"{cp.stderr.strip()}"
            )
    marker.write_text(expected)
    return python


def _run_hook(hook: policy.HostSetupHook, python: Path, project_root: Path) -> str:
    """Run the helper script in the venv; return its stdout. The script path is
    project-relative (or absolute) and must exist — a typo'd path is a loud
    failure, not a silent skip."""
    rel = Path(hook.helper_script)
    script = rel if rel.is_absolute() else project_root / rel
    if not script.is_file():
        raise HostSetupError(
            f"host_setup {hook.name!r}: helper_script {hook.helper_script!r} "
            f"not found at {script}"
        )
    cp = _run(
        [str(python), str(script), *hook.args], project_root, dict(os.environ)
    )
    if cp.returncode != 0:
        raise HostSetupError(
            f"host_setup {hook.name!r}: helper_script exited rc={cp.returncode}: "
            f"{cp.stderr.strip()}"
        )
    return cp.stdout


def run_host_setup(
    pol: policy.Policy | None, project_root: str | Path
) -> dict[str, str]:
    """Run every host_setup hook in declaration order, returning the merged env
    exports. Empty/None policy -> no hooks -> {}. Fails loud on: an unnamed
    hook, an unsupported export source, a missing helper script, or any
    non-zero subprocess exit. One hook's export may not clobber another's
    (collision = misconfiguration)."""
    hooks = policy.host_setup_hooks(pol)
    out: dict[str, str] = {}
    root = Path(project_root)
    for hook in hooks:
        if not hook.name:
            raise HostSetupError(f"host_setup: every hook needs a name; got {hook!r}")
        bad = sorted({v for v in hook.exports.values() if v not in _EXPORT_SOURCES})
        if bad:
            raise HostSetupError(
                f"host_setup {hook.name!r}: unsupported export source(s) {bad}; "
                f"allowed: {sorted(_EXPORT_SOURCES)}"
            )
        python = _ensure_venv(hook, root)
        stdout = _run_hook(hook, python, root)
        for var in hook.exports:
            if var in out:
                raise HostSetupError(
                    f"host_setup: export {var!r} produced by >1 hook (collision)"
                )
            out[var] = stdout
    return out
