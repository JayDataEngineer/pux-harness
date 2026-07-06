"""In-sandbox prep-job runner (Phase 14).

Runs declared ``jobs:`` from ``policy.yaml`` INSIDE the container (after
``create()``, before the agent loop). Each job executes a Python script via
``exec_client.exec()`` — scripts reach external services over the existing
egress allowlist (no new network surface).

Failure semantics: **warn-and-continue**. One bad file doesn't kill 500.
Each job logs failures but the runner continues to the next. The caller
(``prepare()``) gets a full results list with per-job status.

Idempotency is delegated to the scripts themselves — file caches,
SurrealDB UPSERTs, and batch-mode skip-already-done make repeat runs cheap.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from pux_harness.sandbox.docker_exec import DockerExecClient, ExecTimeout
from pux_harness.sandbox.policy import Policy, job_specs

log = logging.getLogger("pux.jobs")


@dataclass
class JobResult:
    """Outcome of one prep job."""

    name: str
    status: str  # "ok" | "failed" | "timeout"
    error: str | None = None  # stderr snippet on failure
    duration: float = 0.0  # wall-clock seconds


def run_jobs(pol: Policy | None, exec_client: DockerExecClient) -> list[JobResult]:
    """Run each declared job in-sandbox. Warn-and-continue on failure.

    Returns a list of ``JobResult`` — one per job, in declaration order.
    The caller can inspect for failures without the runner having raised.
    """
    specs = job_specs(pol)
    if not specs:
        return []

    results: list[JobResult] = []
    for spec in specs:
        if not spec.name:
            log.warning("skipping unnamed job (script=%s)", spec.script)
            results.append(JobResult(name=spec.script or "<unnamed>", status="failed",
                                     error="job has no name"))
            continue
        if not spec.script:
            log.warning("job %s: no script declared", spec.name)
            results.append(JobResult(name=spec.name, status="failed",
                                     error="job has no script"))
            continue

        cmd = f"python3 {spec.script}"
        if spec.args:
            cmd += " " + " ".join(spec.args)

        t0 = time.monotonic()
        try:
            output, exit_code = exec_client.exec(
                cmd, timeout=spec.timeout if spec.timeout > 0 else None
            )
            if exit_code == 0:
                status = "ok"
                error = None
            else:
                status = "failed"
                # Last 500 chars of output for diagnostics
                error = output[-500:] if output else f"exit code {exit_code}"
        except ExecTimeout:
            status = "timeout"
            error = f"exceeded {spec.timeout}s"
        except Exception as exc:
            status = "failed"
            error = str(exc)[:500]

        duration = time.monotonic() - t0
        results.append(JobResult(
            name=spec.name, status=status, error=error, duration=duration,
        ))

        if status != "ok":
            log.warning("job %s %s (%.1fs): %s", spec.name, status, duration, error)
        else:
            log.info("job %s ok (%.1fs)", spec.name, duration)

    return results
