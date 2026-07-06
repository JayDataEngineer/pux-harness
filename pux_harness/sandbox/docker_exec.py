"""Direct Docker exec — the Python-native sandbox execution path.

The harness reaches the pux-sandbox container via the Docker SDK's
``ExecCreate``/``ExecAttach``, with no JSON-RPC hop or middleman.

The container is discovered by its ``openshell.project-path`` label, decoupled
from the Go binary's ``orchestrator-sandbox-<id>`` naming convention. The
container *lifecycle* (create/start/stop) moved into the harness too
(``container.py``); the exec path now **self-boots**: when no running container
is found for the project, ``container.SandboxContainer.ensure()`` creates one
(with this process's ``PUX_ORG`` policy applied) instead of failing. The Go
``task start`` is no longer required to drive the harness.

Why ``tty=False``: the Go binary reads the Docker attach stream raw
(``io.Copy``) so it needs ``TTY=true`` to dodge Docker's multiplexed 8-byte
frame headers. The Python SDK's ``exec_run`` parses that framing for us, so
``tty=False`` gives clean combined stdout+stderr bytes with no terminal ``\\r``
translation — behaviorally equivalent for our text + base64 payloads, and
cleaner output for the inherited ``_build_*_cmd`` scripts. Proven against the
live container (8a probe): ``exec_run(['bash','-c',...], tty=False)`` returns
``ExecResult(exit_code=int, output=bytes)``.

Timeout: ``docker.from_env(timeout=300)`` sets the SDK's HTTP read timeout —
a 300s ceiling carried over from the Go bridge that preceded this path. Per-command
deadlines are enforced via a thread-based ``.result(timeout=…)`` wrapper
— ``exec(cmd, timeout=N)`` raises ``ExecTimeout`` after ``N``
seconds so callers (e.g. ``describe_image``'s 120s) can map it to a clean
result envelope instead of hanging to the 300s socket ceiling. The Docker
SDK is blocking and can't be interrupted, so a timed-out call keeps running
in the background until it finishes or the socket ceiling hits — acceptable
for a single-tenant dev harness, and the caller already got its signal.
"""
from __future__ import annotations

import concurrent.futures
import os

import docker
from docker.errors import APIError, NotFound

from pux_harness.kit._paths import project_root


class ExecTimeout(Exception):
    """Raised when an ``exec(timeout=N)`` deadline elapses. Callers map this to
    a tool-result envelope (e.g. describe_image's ``reason:"timeout"``) rather
    than letting the call hang."""


# Small shared pool for the timed-exec wrapper. ``max_workers`` is intentionally
# modest — exec is sequential in practice (one agent, one tool call at a time);
# the pool only exists because the deadline needs a separate thread to block on.
_EXEC_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Host project root whose sandbox label we filter on — injected via the kit's
# resolver (NOT the install path) and resolved LIVE at use-site (no import-time
# snapshot). ``_resolve_project`` still honors ``PUX_PROJECT_PATH`` as the live
# override.
PROJECT_LABEL = "openshell.project-path"
_DEFAULT_TIMEOUT = 300  # 5min socket ceiling


def _resolve_project() -> str:
    """Absolute project path whose sandbox label we filter on.

    ``PUX_PROJECT_PATH`` overrides (the Go binary honors the same env var);
    otherwise the repo root. Absolute because Docker bind labels store the
    absolute host path (verified: the live container is labeled with the full
    ``/home/ubuntu/.../auto-developer-orchestrator`` path).
    """
    return os.path.abspath(os.environ.get("PUX_PROJECT_PATH") or str(project_root()))


def _discover(client: docker.DockerClient, project_path: str) -> str | None:
    """Find the running sandbox container for ``project_path`` by label, or None.

    Every sandbox is labeled ``openshell.project-path=<abs>``. >1 running match
    is the single-tenant anomaly — we raise loudly so the operator notices
    rather than silently exec-ing the wrong container. No match → None (the
    caller decides whether to boot one).
    """
    try:
        containers = client.containers.list(
            filters={"label": f"{PROJECT_LABEL}={project_path}", "status": "running"},
        )
    except APIError as exc:  # docker daemon unreachable / permission
        raise RuntimeError(f"docker list failed: {exc}") from exc
    if not containers:
        return None
    if len(containers) > 1:
        names = sorted(c.name for c in containers)
        raise RuntimeError(
            f"single-tenant invariant violated: {len(containers)} running containers "
            f"match project {project_path!r}: {names}"
        )
    return containers[0].name


class DockerExecClient:
    """Exec-only client over the Docker SDK.

    Caches the container name so the label-filter lookup runs once per process —
    the hot path is ``exec()``, not discovery. When ``boot=True`` (default) and
    no running container is found, ``container.SandboxContainer.ensure()`` boots
    one (with the process's ``PUX_ORG`` policy) so the harness is self-starting.
    """

    def __init__(
        self,
        container: str | None = None,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
        boot: bool = True,
    ):
        self._client = docker.from_env(timeout=timeout)
        self._container = container
        self._boot = boot

    @property
    def container(self) -> str:
        if self._container is None:
            found = _discover(self._client, _resolve_project())
            if found is not None:
                self._container = found
            elif self._boot:
                # Lazy import — avoids a docker_exec↔container cycle at import time.
                from pux_harness.sandbox.container import SandboxContainer

                self._container = SandboxContainer(client=self._client).ensure()
            else:
                raise RuntimeError(
                    "no running pux-sandbox container found (boot disabled). "
                    "Start one with `task start` or enable boot."
                )
        return self._container

    def _do_exec(self, command: str):
        """The blocking docker call, isolated so ``exec(timeout=…)`` can run it
        on a worker thread and enforce a deadline via ``.result(timeout=…)``."""
        try:
            return self._client.containers.get(self.container).exec_run(
                ["bash", "-c", command],
                tty=False,
                demux=False,  # combined stream; SDK parses Docker framing
                stdin=False,
            )
        except NotFound as exc:
            raise RuntimeError(
                f"sandbox container {self.container!r} vanished mid-run: {exc}"
            ) from exc

    def exec(self, command: str, *, timeout: int | None = None) -> tuple[str, int]:
        """Run ``bash -c <command>`` in the sandbox; return (output, exit_code).

        Output is the combined stdout+stderr, utf-8 decoded (errors replaced)
        so binary-ish payloads (base64 from upload/download helpers) survive.
        ``exit_code`` is 0 on success, non-zero on container-side failure — the
        caller decides whether non-zero is an error (the inherited fs scripts
        append ``|| true`` so they report 0; a raw command failing surfaces its
        real exit code).

        ``timeout`` (seconds) caps wall-clock: raises ``ExecTimeout`` past it.
        ``None`` (default) blocks to the client's 300s socket ceiling — the
        original behavior, unchanged for the fs/shell tools that never pass one.
        """
        if timeout is None:
            result = self._do_exec(command)
        else:
            future = _EXEC_POOL.submit(self._do_exec, command)
            try:
                result = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as exc:
                raise ExecTimeout(
                    f"exec timed out after {timeout}s: {command[:120]!r}"
                ) from exc
        out = result.output
        if isinstance(out, (bytes, bytearray)):
            out = out.decode("utf-8", "replace")
        return out, int(result.exit_code) if result.exit_code is not None else 0


_client: DockerExecClient | None = None


def get_exec_client(container: str | None = None) -> DockerExecClient:
    """Build a fresh exec client (container auto-discovered if not given)."""
    return DockerExecClient(container=container)


def shared_exec() -> DockerExecClient:
    """One exec client for the process — created lazily so importing this
    module never touches Docker (keeps tests + ``--help`` offline-cheap)."""
    global _client
    if _client is None:
        _client = get_exec_client()
    return _client


if __name__ == "__main__":  # pragma: no cover - operator smoke probe
    ec = get_exec_client()
    out, code = ec.exec("cat /etc/hostname && echo --- && ls /sandbox/workspace | head -3")
    print(f"[container={ec.container} exit={code}]")
    print(out)
