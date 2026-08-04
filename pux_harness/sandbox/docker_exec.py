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
import io
import os
import subprocess
import sys
import tarfile

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
_EXEC_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# Host project root whose sandbox label we filter on — injected via the kit's
# resolver (NOT the install path) and resolved LIVE at use-site (no import-time
# snapshot). ``_resolve_project`` still honors ``PUX_PROJECT_PATH`` as the live
# override.
PROJECT_LABEL = "openshell.project-path"
_DEFAULT_TIMEOUT = 120  # 2min hard ceiling for ANY exec call
# Ceiling used when the caller asks for "no-timeout" semantics. deepagents'
# filesystem middleware documents ``timeout=0`` to the model as
# "Use 0 for no-timeout execution on backends that support it"
# (filesystem.py:414/1683). The model passes 0 expecting unbounded patience
# for long-running commands; without this ceiling, ``future.result(timeout=0)``
# trips INSTANTLY and surfaces as a misleading "model stream stalled" notice.
# We can't actually block forever — ``future.result(timeout=N)`` needs a real
# number — so we use the Docker SDK's HTTP read ceiling (``docker.from_env``
# is built with ``timeout=300``). 300s is the most the SDK will give us on a
# single read anyway; going higher would just shift the failure to a less
# legible ``read timeout`` at the HTTP layer.
_NO_TIMEOUT_CEILING = 300


def _resolve_timeout(timeout: int | None) -> int:
    """Coerce a caller-supplied timeout into an effective deadline.

    - ``None``: use ``_DEFAULT_TIMEOUT`` (120s).
    - ``<= 0``: deepagents API contract — "no-timeout execution". We can't
      block forever, so use ``_NO_TIMEOUT_CEILING`` (300s, the Docker SDK's
      HTTP read ceiling). This honors the model's intent for long-running
      commands (builds, recursive pux invocations, long queries) instead of
      failing instantly with the pre-fix ``future.result(timeout=0)``.
    - ``> 0``: use the supplied value as-is (already clamped by the
      filesystem middleware's ``max_execute_timeout`` of 3600s).
    """
    if timeout is None:
        return _DEFAULT_TIMEOUT
    if timeout <= 0:
        return _NO_TIMEOUT_CEILING
    return timeout


def _resolve_project() -> str:
    """Absolute project path whose sandbox label we filter on.

    ``PUX_PROJECT_PATH`` overrides (the Go binary honors the same env var);
    otherwise the repo root. Absolute because Docker bind labels store the
    absolute host path (verified: the live container is labeled with the full
    ``/home/ubuntu/.../auto-developer-orchestrator`` path).

    **Foot-gun warning:** when ``PUX_PROJECT_PATH`` is unset we fall back to the
    harness repo — the same silent cross-project leak ``container.resolve_project_path``
    guards against. Emit a loud stderr line so a launcher that forgot to pin the
    edit target is VISIBLE, not silent. See ``acp.run_acp`` for the hard guard
    on the editor-serving path.
    """
    p = os.environ.get("PUX_PROJECT_PATH")
    if not p:
        import sys as _sys
        _harness = str(project_root())
        _sys.stderr.write(
            "pux sandbox: WARNING — PUX_PROJECT_PATH unset; resolving sandbox "
            f"label against the harness repo fallback ({_harness}). If this agent "
            "was spawned against a DIFFERENT project, export "
            "PUX_PROJECT_PATH=<your project>.\n"
        )
        p = _harness
    return os.path.abspath(p)


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
        # Publish the resolved container name so downstream tool-server
        # resolution can expand ``${PUX_SANDBOX_CONTAINER}`` in stdio MCP
        # server args (the sandbox-browser MCP entry uses this to spawn
        # ``docker exec -i ${PUX_SANDBOX_CONTAINER} mc_browser.py``). Set
        # unconditionally — this property is the single source of truth for
        # "what container does this process talk to" and runs once per
        # process (the cache short-circuits subsequent reads). Guarded for
        # tests that inject a non-string _container (MagicMock etc.).
        if isinstance(self._container, str):
            os.environ["PUX_SANDBOX_CONTAINER"] = self._container
        return self._container

    # Host-side env vars that cross the Docker boundary on every exec.
    # ``docker exec`` does NOT inherit the host process's env — the container
    # has its own env set at creation time. These structural parameters (data
    # folder, run dir) are injected by ``--data`` / the harness and MUST reach
    # the agent inside the container. Without this, ``$DATA_DIR`` is empty
    # inside the sandbox and the agent can't find its raw data.
    _PASSTHROUGH_ENV = (
        "DATA_DIR",            # --data parameter: raw data folder for preprocessing
        "RUN_DIR",             # pipeline output dir (when explicitly set)
        "PUX_PROJECT_PATH",
    )

    def _passthrough_env(self) -> dict[str, str] | None:
        """Harvest passthrough vars from the host process env.

        Returns a dict suitable for ``exec_run(environment=...)`` — ADDS to
        the container's env for this exec, does NOT replace it. ``None`` when
        nothing is set (skip the kwarg entirely so old SDK versions that don't
        accept ``environment=`` still work).
        """
        env = {}
        for key in self._PASSTHROUGH_ENV:
            val = os.environ.get(key)
            if val:
                env[key] = val
        return env or None

    def _do_exec(self, command: str):
        """The blocking docker call, isolated so ``exec(timeout=…)`` can run it
        on a worker thread and enforce a deadline via ``.result(timeout=…)``.

        Tolerates a vanished container: if the cached name is gone (e.g.
        ``reset_session`` force-removed it, or an external ``docker rm``), clear
        the cache so the ``container`` property re-ensures a fresh one, then
        retry once. Without this, every tool call after a reset would die with
        'vanished mid-run' until the agent process is restarted.

        Forwards ``_PASSTHROUGH_ENV`` vars (DATA_DIR, RUN_DIR, etc.) from the
        host process into the container so ``--data`` / structural parameters
        reach the agent. Without this, ``$DATA_DIR`` is empty inside the
        container and the agent can't find its raw data."""
        env = self._passthrough_env()
        try:
            return self._client.containers.get(self.container).exec_run(
                ["bash", "-c", command],
                tty=False,
                demux=False,  # combined stream; SDK parses Docker framing
                stdin=False,
                environment=env,
            )
        except NotFound:
            # Cached container vanished (external reset/remove). Clear the cache
            # so the property re-ensures, then retry exactly once.
            self._container = None
        try:
            return self._client.containers.get(self.container).exec_run(
                ["bash", "-c", command],
                tty=False,
                demux=False,
                stdin=False,
                environment=env,
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
        ``None`` (default) uses ``_DEFAULT_TIMEOUT`` (120s); ``0`` or negative
        uses ``_NO_TIMEOUT_CEILING`` (300s, per the deepagents "no-timeout"
        contract — see ``_resolve_timeout``). EVERY call goes
        through the thread-pool deadline wrapper — this is the critical guard
        against a frozen container. Without it, a container whose process
        namespace is stuck (``procReady not received``) accepts TCP connections
        at the kernel level but never responds to the exec, and the Docker
        SDK's HTTP read timeout doesn't reliably fire for exec streaming
        responses. The thread wrapper is the only reliable kill switch.
        """
        effective_timeout = _resolve_timeout(timeout)
        future = _EXEC_POOL.submit(self._do_exec, command)
        try:
            result = future.result(timeout=effective_timeout)
        except concurrent.futures.TimeoutError as exc:
            raise ExecTimeout(
                f"exec timed out after {effective_timeout}s: {command[:120]!r}"
                ) from exc
        out = result.output
        if isinstance(out, (bytes, bytearray)):
            out = out.decode("utf-8", "replace")
        return out, int(result.exit_code) if result.exit_code is not None else 0

    def upload_file(self, container_path: str, data: bytes,
                    *, timeout: int | None = None) -> None:
        """Upload bytes to a path in the container using Docker's tar API.

        This replaces the old ``upload_files`` path that ran
        ``python3 -c <script> <path> <base64>`` with the base64 payload as a
        CLI argument — which hit Linux ``ARG_MAX`` (~128KB) and made every
        large upload fail with "argument list too long": the compact
        middleware (offloading conversation history to sandbox files),
        twitter image posts, and any file > ~100KB all died silently here.

        ``put_archive`` ships the bytes as a tar stream over the Docker API's
        HTTP body — no argv, no shell quoting, no size ceiling below the
        Docker daemon's own (far higher) limits. Single file per call.
        """
        container = self._client.containers.get(self.container)
        parent = os.path.dirname(container_path) or "/"
        basename = os.path.basename(container_path)

        def _do() -> None:
            container.exec_run(["mkdir", "-p", parent])
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as tar:
                info = tarfile.TarInfo(name=basename)
                info.size = len(data)
                info.mode = 0o644
                info.mtime = 0  # deterministic
                tar.addfile(info, io.BytesIO(data))
            stream.seek(0)
            container.put_archive(parent, stream)

        effective_timeout = _resolve_timeout(timeout)
        future = _EXEC_POOL.submit(_do)
        try:
            future.result(timeout=effective_timeout)
        except concurrent.futures.TimeoutError as exc:
            raise ExecTimeout(
                f"upload_file timed out after {effective_timeout}s: "
                f"{container_path} ({len(data)} bytes)"
            ) from exc

    def download_file(self, container_path: str,
                      *, timeout: int | None = None) -> bytes:
        """Download a file from the container via the Docker tar API.

        Symmetric with ``upload_file``; avoids the ``_DOWNLOAD_PY`` base64
        stdout path that breaks for large files (output truncation, decode
        errors). Returns the raw file bytes.
        """
        container = self._client.containers.get(self.container)
        parent = os.path.dirname(container_path) or "/"
        basename = os.path.basename(container_path)

        def _do() -> bytes:
            stream, _stat = container.get_archive(container_path)
            with tarfile.open(fileobj=io.BytesIO(b"".join(stream)), mode="r") as tar:
                member = tar.getmember(basename) if basename in tar.getnames() else None
                # The tar may store the file under its basename or the full path;
                # fall back to the first regular file.
                if member is None:
                    for m in tar.getmembers():
                        if m.isfile():
                            member = m
                            break
                if member is None:
                    raise FileNotFoundError(f"{container_path} not in archive")
                return tar.extractfile(member).read()

        effective_timeout = _resolve_timeout(timeout)
        future = _EXEC_POOL.submit(_do)
        try:
            return future.result(timeout=effective_timeout)
        except concurrent.futures.TimeoutError as exc:
            raise ExecTimeout(
                f"download_file timed out after {effective_timeout}s: "
                f"{container_path}"
            ) from exc


class LocalExecClient:
    """Subprocess-backed exec client — the no-Docker fallback.

    Same interface as ``DockerExecClient`` (``exec``, ``upload_file``,
    ``download_file``, ``container``). Used when the harness runs INSIDE a
    sandbox that has no Docker socket of its own — the "pux-coder inside the
    sandbox" case: the agent's tools run against the local filesystem instead
    of being forwarded to a sibling container via ``docker exec``.

    Triggered by:
    - ``PUX_EXEC_MODE=local`` (explicit override — runs commands in-process),
    - ``docker.from_env()`` failing (no socket, no daemon, permission denied).

    Commands run via ``bash -c`` with cwd pinned to ``PUX_PROJECT_PATH`` (the
    in-sandbox workspace root, e.g. ``/sandbox/workspace`` — matches the
    Docker WORKDIR the DockerExecClient hits). Paths passed to
    ``upload_file`` / ``download_file`` are treated as LOCAL filesystem paths;
    in local mode the "container" IS this process's namespace.

    Why this exists: without it, ``build_graph()`` calls ``shared_exec()`` at
    agent-construction time, which unconditionally built a ``DockerExecClient``
    and called ``docker.from_env()`` — dying with
    ``docker.errors.DockerException: FileNotFoundError`` before the agent
    could even start, whenever the harness was invoked from inside a sandbox
    that doesn't mount the host daemon socket. The fallback lets the harness
    self-heal: same Python, same tool surface, no monkey-patching of the
    ``docker`` SDK.
    """

    def __init__(
        self,
        container: str | None = None,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        self._container = (
            container
            or os.environ.get("PUX_SANDBOX_CONTAINER")
            or "local"
        )
        # Publish so downstream ``${PUX_SANDBOX_CONTAINER}`` expansion in
        # stdio MCP args sees *something*. (Stdio MCP servers that themselves
        # need ``docker exec`` will still fail in local mode — that's a
        # separate, narrower problem. The exec/fs/shell surface — the part
        # that was crashing at build time — works.)
        os.environ["PUX_SANDBOX_CONTAINER"] = self._container

    @property
    def container(self) -> str:
        return self._container

    def _cwd(self) -> str | None:
        # In-sandbox: PUX_PROJECT_PATH=/sandbox/workspace (matches the Docker
        # WORKDIR). On host: the repo root. Either way, commands resolve
        # project-relative paths correctly. Fall back to None (inherit caller
        # cwd) if unset or bogus.
        p = os.environ.get("PUX_PROJECT_PATH")
        if p and os.path.isdir(p):
            return p
        return None

    def exec(self, command: str, *, timeout: int | None = None) -> tuple[str, int]:
        """Run ``bash -c <command>`` in-process; return (output, exit_code).

        Output is combined stdout+stderr, utf-8 decoded (errors replaced) —
        byte-for-byte the same envelope ``DockerExecClient.exec`` returns, so
        ``PuxSandboxBackend.execute`` and every tool (``read_file``'s
        ``_build_*_cmd``, declared tools, the grader) work unchanged.
        """
        effective_timeout = _resolve_timeout(timeout)
        try:
            r = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                timeout=effective_timeout,
                cwd=self._cwd(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecTimeout(
                f"local exec timed out after {effective_timeout}s: "
                f"{command[:120]!r}"
            ) from exc
        out = (r.stdout or b"") + (r.stderr or b"")
        return out.decode("utf-8", "replace"), int(r.returncode)

    def upload_file(self, container_path: str, data: bytes,
                    *, timeout: int | None = None) -> None:
        """Write bytes to a local path. In local mode the container_path IS
        a host path — mkdir -p the parent and write atomically."""
        parent = os.path.dirname(container_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Atomic write: write to a sibling temp file and rename, so a partial
        # write never leaves a corrupt file visible to a concurrent reader.
        tmp = f"{container_path}.pux-tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, container_path)

    def download_file(self, container_path: str,
                      *, timeout: int | None = None) -> bytes:
        """Read bytes from a local path."""
        with open(container_path, "rb") as f:
            return f.read()


# Union type for type hints — anywhere that accepts a ``DockerExecClient``
# also accepts a ``LocalExecClient`` (same surface).
ExecClient = DockerExecClient | LocalExecClient


_client: ExecClient | None = None
_docker_avail_cached: bool | None = None


def _docker_socket_path() -> str | None:
    """Resolve the filesystem path of the Docker socket from ``DOCKER_HOST``,
    or ``/var/run/docker.sock`` if unset. Returns ``None`` for non-unix
    schemes (tcp/ssh/npipe) where we can't pre-check existence."""
    host = os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock")
    if host.startswith("unix://"):
        return host[len("unix://"):]
    return None


def _docker_available() -> bool:
    """Probe whether ``docker.from_env()`` can REALLY reach a daemon. Cached.

    Two-stage check so we can't be fooled by a monkey-patched ``docker`` SDK
    (the workaround the pux-coder agent previously used inside sandboxes that
    had no socket — it replaced ``docker/__init__.py`` with a fake whose
    ``from_env()`` returned a subprocess-shimming client):

    1. **Socket existence.** If ``DOCKER_HOST`` points at a unix socket that
       isn't on disk, no Docker client — real or fake — can reach a daemon.
       Short-circuit ``False`` without importing ``docker`` at all.
    2. **Daemon probe.** ``docker.from_env(timeout=2)`` eagerly calls
       ``_retrieve_server_version()``; it raises ``DockerException``
       synchronously when the socket is absent or the daemon is down.

    Both stages must pass for ``True``. Probed once per process.
    """
    global _docker_avail_cached
    if _docker_avail_cached is not None:
        return _docker_avail_cached
    sock = _docker_socket_path()
    if sock is not None and not os.path.exists(sock):
        # Socket absent → no daemon possible, even if a fake SDK is installed.
        _docker_avail_cached = False
        return False
    try:
        docker.from_env(timeout=2)
        _docker_avail_cached = True
    except Exception:
        # DockerException, FileNotFoundError, ConnectionError — anything means
        # "no daemon reachable from here."
        _docker_avail_cached = False
    return _docker_avail_cached


def get_exec_client(container: str | None = None) -> ExecClient:
    """Build a fresh exec client (container auto-discovered if not given).

    Backend selection:
    - ``PUX_EXEC_MODE=local``  → ``LocalExecClient`` (subprocess, no Docker).
    - ``PUX_EXEC_MODE=docker`` → ``DockerExecClient`` (hard — crashes if no
      daemon, matching pre-fix behavior, for ops paths that MUST have Docker).
    - unset + daemon reachable → ``DockerExecClient`` (the default).
    - unset + daemon unreachable → ``LocalExecClient`` with a loud stderr
      notice. This is the self-heal path: lets ``pux direct`` run from inside
      a sandbox that has no Docker socket, instead of crashing at
      ``build_graph`` time.
    """
    mode = os.environ.get("PUX_EXEC_MODE", "").lower()
    if mode == "local":
        return LocalExecClient(container=container)
    if mode == "docker":
        return DockerExecClient(container=container)
    if _docker_available():
        return DockerExecClient(container=container)
    # Self-heal: no Docker socket + no explicit override. Fall back to local
    # subprocess so the harness runs instead of dying at agent-build time.
    sys.stderr.write(
        "pux sandbox: no Docker daemon reachable (PUX_EXEC_MODE unset, "
        "docker.from_env() failed). Falling back to LocalExecClient — "
        "commands will run in-process against PUX_PROJECT_PATH. Set "
        "PUX_EXEC_MODE=local to silence this, or PUX_EXEC_MODE=docker to "
        "force a hard failure.\n"
    )
    return LocalExecClient(container=container)


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
