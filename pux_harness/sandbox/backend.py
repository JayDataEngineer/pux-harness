"""PuxSandboxBackend — a deepagents ``BaseSandbox`` over a Docker exec client.

Shape A (decided by probe in Phase 3): subclass
``deepagents.backends.sandbox.BaseSandbox`` and implement only its four
abstract primitives — ``execute``, ``id``, ``upload_files``, ``download_files``.
The inherited ``ls/read/write/edit/grep/glob`` (and all ``a*`` async variants)
run small ``python3``/``grep`` scripts *through our* ``execute()``, so they work
the moment ``execute()`` does.

The 13 specialist tools (``python``/``browser_*``/``desktop_*``/
``describe_image``/skills) are also native Python (``native_tools.py``)
and run through this same ``DockerExecClient``. The backend (fs/shell) and
the specialists are two Python surfaces over one ``docker exec`` path into
the container.
"""
from __future__ import annotations

import base64
import shlex
from collections import deque

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from pux_harness.sandbox.docker_exec import DockerExecClient

# Workspace root inside the sandbox container — the project bind-mount and the
# Dockerfile ``WORKDIR``. ``BaseSandbox.glob`` defaults ``path`` to ``/`` when
# the model omits it (``search_path = path or "/"`` → ``os.chdir("/")``), and a
# bare ``glob("**/*.py")`` then walks the ENTIRE container (site-packages,
# ``/usr``, ``/proc`` …) and blows the 20s ``GLOB_TIMEOUT``. Default an
# unscoped glob to the project workspace instead: that is what the model means
# by "here", and the search stays bounded. ``grep``/``ls`` are unaffected —
# they run via ``docker exec`` in the container's WORKDIR (this root), so an
# unscoped grep already searches the workspace, not ``/``. Only ``glob`` chdir's
# to ``/``, so only ``glob`` is overridden here.
WORKSPACE_ROOT = "/sandbox/workspace"

# python3 snippets used for byte-accurate upload/download via execute(). Base64
# carries the payload so text and binary share one path; the snippets are quoted
# as a single shell argv so embedded quotes/newlines in paths can't break out.
_UPLOAD_PY = (
    "import base64,sys,os;"
    "p=sys.argv[1];"
    "d=os.path.dirname(p);"
    "os.makedirs(d,exist_ok=True) if d else None;"
    "open(p,'wb').write(base64.b64decode(sys.argv[2]))"
)
_DOWNLOAD_PY = (
    "import base64,sys;"
    "sys.stdout.buffer.write(base64.b64encode(open(sys.argv[1],'rb').read()))"
)


class PuxSandboxBackend(BaseSandbox):
    """deepagents sandbox backed by a direct ``docker exec`` (Phase 8a)."""

    def __init__(self, exec_client: DockerExecClient):
        self._exec = exec_client
        self._id: str | None = None
        # Every command run through native execute() — including the inherited
        # ls/read/glob/grep/write/edit (they all build a cmd + call execute()).
        # Observation-only: turns "did the subagent use native fs tools?" from
        # inference into direct evidence. Bounded so a long-lived server
        # process can't leak memory here.
        self.execute_log: deque[str] = deque(maxlen=2048)

    # --- the four abstract primitives --------------------------------------

    @property
    def id(self) -> str:
        # Not invoked on the framework hot path (no `.id` reads in
        # middleware/graph), but abstract-required. Lazily reflect the real
        # container hostname so it's meaningful if ever logged.
        if self._id is None:
            out, _ = self._exec.exec("cat /etc/hostname")
            self._id = out.strip() or self._exec.container
        return self._id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        self.execute_log.append(command)
        output, exit_code = self._exec.exec(command, timeout=timeout)
        # Non-zero exit in the container: the inherited _build_*_cmd scripts
        # append `2>/dev/null` + `|| true`, so a non-zero exit here is a real
        # failure (or a raw command the model ran) — surface it verbatim.
        return ExecuteResponse(output=output, exit_code=exit_code, truncated=False)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        out: list[FileUploadResponse] = []
        for path, data in files:
            b64 = base64.b64encode(data).decode("ascii")
            cmd = ("python3 -c " + shlex.quote(_UPLOAD_PY)
                   + f" {shlex.quote(path)} {shlex.quote(b64)}")
            res = self.execute(cmd)
            if res.exit_code != 0:
                out.append(FileUploadResponse(
                    path=path, error=f"upload failed: {res.output}"))
            else:
                out.append(FileUploadResponse(path=path, error=None))
        return out

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        out: list[FileDownloadResponse] = []
        for path in paths:
            cmd = "python3 -c " + shlex.quote(_DOWNLOAD_PY) + f" {shlex.quote(path)}"
            res = self.execute(cmd)
            if res.exit_code != 0:
                out.append(FileDownloadResponse(
                    path=path, content=None, error=f"download failed: {res.output}"))
                continue
            # b64decode(validate=False) discards transport noise (\r\n); a
            # short payload means the script itself errored before printing.
            try:
                content = base64.b64decode(res.output.strip())
            except Exception as exc:  # noqa: BLE001 - surface, don't raise
                out.append(FileDownloadResponse(
                    path=path, content=None, error=f"b64 decode failed: {exc}"))
                continue
            out.append(FileDownloadResponse(path=path, content=content, error=None))
        return out

    # ls/read/write/edit/grep + async variants: inherited from BaseSandbox.
    #
    # glob/aglob: OVERRIDDEN to default an omitted ``path`` to ``WORKSPACE_ROOT``
    # (see WORKSPACE_ROOT above). ``BaseSandbox.glob`` defaults to ``/`` → full
    # -container walk → 20s timeout. Everything else delegates to ``super()``
    # unchanged, so an explicit ``path`` (incl. ``/``) still works.
    def glob(self, pattern: str, path: str | None = None):
        return super().glob(pattern, path=path or WORKSPACE_ROOT)

    async def aglob(self, pattern: str, path: str | None = None):
        return await super().aglob(pattern, path=path or WORKSPACE_ROOT)
