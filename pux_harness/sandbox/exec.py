"""The thin exec seam — one ``BaseSandbox`` for the process.

Replaces the old Docker plumbing (``container.py`` + ``docker_exec.py`` +
``backend.py`` + ``host_setup.py``, ~2,600 LOC) with an upstream
``deepagents.backends.sandbox.BaseSandbox``. The backend is constructed by
``shared_backend()``:

* ``PUX_SANDBOX=openshell`` (default) → ``OpenShellSandbox`` over the NVIDIA
  OpenShell gateway (``langchain_nvidia_openshell``).
* ``PUX_SANDBOX=local`` → a host-filesystem backend (deepagents ships one;
  no container required — used by tests / ``kit compile``).

The 13 specialist tools (browser/desktop/grader/python/…) now take
``BaseSandbox`` directly — the portable langchain tool contract. No adapter.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

from deepagents.backends.sandbox import BaseSandbox

# Workspace root inside the sandbox — the project mount + the image WORKDIR.
# Kept as a constant because a handful of specialist tools (declared/dynamic)
# + grader descriptions reference it for in-sandbox path wayfinding.
WORKSPACE_ROOT = "/sandbox/workspace"


# --- project wayfinding (host-side, no docker) -----------------------------

def resolve_project_path() -> str:
    """Absolute project path. ``PUX_PROJECT_PATH`` wins, else the repo root.

    The sandbox workspace is bind-mounted from this path. URL schemes are
    rejected: their colons corrupt path parsing."""
    from pux_harness.kit._paths import project_root  # noqa: PLC0415
    p = os.environ.get("PUX_PROJECT_PATH")
    if not p:
        p = str(project_root())
        sys.stderr.write(
            "pux sandbox: WARNING — PUX_PROJECT_PATH unset; binding "
            f"{WORKSPACE_ROOT} to the harness repo fallback ({p}).\n"
        )
    if "://" in p:
        raise ValueError(f"sandboxes require a local path; received a URL: {p!r}")
    return os.path.abspath(p)


# --- prep jobs (moved from container.py — slimmed, no container ensure) ----

def prepare(
    org: str,
    project_path: str | None = None,
    sandbox: BaseSandbox | None = None,
    universal_warmup: bool = False,
) -> list[dict[str, Any]]:
    """Run declared ``jobs:`` from ``policy.yaml`` inside the sandbox, before
    the agent loop. Returns a list of ``{name, status, error, duration}`` dicts.

    With OpenShell the sandbox auto-starts on first exec — no container ensure
    needed. ``universal_warmup`` (the serve path) additionally probes the
    run-completion webhook endpoint so a webhook-less client can observe
    completions; ``direct`` leaves it False (no serve in direct mode)."""
    from pux_harness.sandbox import policy as _policy  # noqa: PLC0415
    from pux_harness.sandbox.jobs import JobResult, run_jobs  # noqa: PLC0415

    if not project_path:
        project_path = resolve_project_path()
    if sandbox is None:
        sandbox = shared_backend()

    try:
        pol = _policy.load(org, project_path)
    except _policy.NoPolicy:
        pol = None
    specs = _policy.job_specs(pol) if pol is not None else []

    if not specs and not universal_warmup:
        return []

    results: list[JobResult] = list(run_jobs(pol, sandbox)) if specs else []

    if universal_warmup:
        t0 = time.monotonic()
        try:
            _r = sandbox.execute(
                "python3 orgs/_shared/sandbox/warmup_webhook.py", timeout=30
            )
            out, rc = _r.output, _r.exit_code
            results.append(JobResult(
                name="warmup_webhook",
                status="ok" if rc == 0 else "failed",
                error=None if rc == 0 else (out[-500:] if out else f"exit {rc}"),
                duration=time.monotonic() - t0,
            ))
        except Exception as exc:  # noqa: BLE001 - prep must never block the run
            results.append(JobResult(
                name="warmup_webhook", status="failed",
                error=str(exc)[:500], duration=time.monotonic() - t0,
            ))

    return [
        {"name": r.name, "status": r.status, "error": r.error,
         "duration": round(r.duration, 1)}
        for r in results
    ]


# --- process singletons ---------------------------------------------------

_backend: BaseSandbox | None = None
# Hold the entered openshell.Sandbox so it isn't GC'd (its __exit__ tears down
# the sandbox). Lives at module scope for the process lifetime.
_openshell_sb: Any = None


def shared_backend() -> BaseSandbox:
    """One ``BaseSandbox`` for the process (lazy)."""
    global _backend
    if _backend is None:
        _backend = _make_backend()
    return _backend


def _make_backend() -> BaseSandbox:
    mode = os.environ.get("PUX_SANDBOX", "openshell")
    if mode == "openshell":
        return _make_openshell_backend()
    if mode == "local":
        return _make_local_backend()
    raise RuntimeError(
        f"PUX_SANDBOX={mode!r} unsupported (use 'openshell' or 'local')."
    )


def _make_openshell_backend() -> BaseSandbox:
    global _openshell_sb
    try:
        import openshell  # noqa: PLC0415
        from langchain_nvidia_openshell import OpenShellSandbox  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "PUX_SANDBOX=openshell but the OpenShell SDK is not installed. "
            "pip install openshell langchain-nvidia-openshell, then start the "
            "gateway (see ~/.local/share/pux/openshell/README.md)."
        ) from exc
    ws = os.environ.get("PUX_SANDBOX_WORKSPACE", "default")
    _openshell_sb = openshell.Sandbox(
        workspace=ws, name=f"pux-{os.getpid()}", delete_on_exit=False,
    )
    _openshell_sb.__enter__()  # hold open for the process lifetime
    return OpenShellSandbox(sandbox=_openshell_sb)


def _make_local_backend() -> BaseSandbox:
    """Host-shell backend — no container. Runs commands directly on the host.
    Used by tests / ``kit compile`` / anywhere the OpenShell gateway isn't
    available. deepagents' ``LocalShellBackend`` implements the full
    ``BaseSandbox`` surface (``execute``/``ls``/``read``/``upload_files``/…)."""
    from deepagents.backends.local_shell import LocalShellBackend  # noqa: PLC0415
    return LocalShellBackend()
