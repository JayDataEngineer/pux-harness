"""PuxSandboxBackend — a deepagents ``BaseSandbox`` over a Docker exec client.

Shape A (decided by probe): subclass
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

from pux_harness.sandbox.docker_exec import DockerExecClient, ExecTimeout

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


def _strip_ws_root(path: str) -> str:
    """Convert container-absolute workspace paths to project-relative.

    ``/sandbox/workspace/CANVAS-GAPS.md`` → ``CANVAS-GAPS.md``;
    ``/sandbox/workspace`` → ``.``. Paths outside the workspace are returned
    unchanged. This makes tool output read like Claude Code (relative paths),
    not like a Docker exec log — the agent never needs to know about
    ``/sandbox/workspace`` at all.
    """
    if path == WORKSPACE_ROOT:
        return "."
    if path.startswith(WORKSPACE_ROOT + "/"):
        return path[len(WORKSPACE_ROOT) + 1:]
    return path


def _relativize_entries(entries: list) -> list:
    """Strip WORKSPACE_ROOT from the ``path`` key of each entry/match dict.

    Works for ``LsResult.entries``, ``GlobResult.matches``, and
    ``GrepResult.matches`` — they're all lists of dicts with a ``path`` key.
    Mutates in place AND returns for chaining (the result dataclasses are not
    frozen)."""
    for entry in entries:
        if isinstance(entry, dict):
            p = entry.get("path")
            if isinstance(p, str):
                entry["path"] = _strip_ws_root(p)
    return entries

# python3 snippets used for byte-accurate upload/download via execute(). Base64
# carries the payload so text and binary share one path; the snippets are quoted
# as a single shell argv so embedded quotes/newlines in paths can't break out.
_UPLOAD_PY = (
    "import base64,sys,os;"
    "p=sys.argv[1];"
    "d=os.path.dirname(p);"
    "os.makedirs(d,exist_ok=True) if d else None;"
    "open(p,'wb').write(base64.b64decode(sys.stdin.read()))"
)
_DOWNLOAD_PY = (
    "import base64,sys;"
    "sys.stdout.buffer.write(base64.b64encode(open(sys.argv[1],'rb').read()))"
)


class PuxSandboxBackend(BaseSandbox):
    """deepagents sandbox backed by a direct ``docker exec``."""

    def __init__(self, exec_client: DockerExecClient):
        self._exec = exec_client
        self._id: str | None = None
        # Every command run through native execute() — including the inherited
        # ls/read/glob/grep/write/edit (they all build a cmd + call execute()).
        # Observation-only: turns "did the subagent use native fs tools?" from
        # inference into direct evidence. Bounded so a long-lived server
        # process can't leak memory here.
        self.execute_log: deque[str] = deque(maxlen=2048)
        # Depth counter: >0 when inherited methods (read/write/edit/ls/glob/grep)
        # call execute() internally. When >0, execute() skips path-stripping so
        # file CONTENT and structured command output isn't corrupted. Only
        # direct agent calls to execute("find /") get the prefix stripped.
        self._internal_depth = 0

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
        try:
            output, exit_code = self._exec.exec(command, timeout=timeout)
        except ExecTimeout as exc:
            # A tool-side timeout is NOT a crash — the sandbox capped the
            # command at its wall-clock budget (``_resolve_timeout`` in
            # ``docker_exec.py``). Surface a clean ExecuteResponse so the
            # deepagents FilesystemMiddleware returns a ToolMessage the model
            # can act on ("the command took too long — split it into shorter
            # steps"), instead of letting ExecTimeout walk up to the model
            # node where ``retry_on_stream_stall`` correctly skips it (it's
            # deterministic) but the prompt-boundary wrapper then mislabels
            # the failure as a "model stream stalled" notice. Exit 124 is
            # the GNU ``timeout`` convention.
            preview = command if len(command) <= 200 else command[:197] + "..."
            return ExecuteResponse(
                output=(
                    f"[execute] timeout: {exc}.\n"
                    f"Command preview: {preview}\n"
                    f"The sandbox capped this command at its wall-clock "
                    f"budget. For long-running work, write progress to a "
                    f"marker file and poll across separate tool calls."
                ),
                exit_code=124,
                truncated=False,
            )
        # Strip the container workspace prefix from raw shell output so the
        # agent sees project-relative paths. ``find /``, ``pwd``, ``ls``,
        # ``tree`` all print /sandbox/workspace/foo — the agent copies that
        # into read_file calls and gets confused. Strip to BARE relative
        # names (CANVAS-GAPS.md, not ./CANVAS-GAPS.md) so read_file resolves
        # them correctly from the container WORKDIR. Handle pwd's bare
        # /sandbox/workspace (no trailing slash) → ".". Only strip when called
        # DIRECTLY by the agent (_internal_depth == 0); inherited
        # read/write/edit/grep call execute() internally and their output
        # (file content, structured grep lines) must not be touched.
        if output and self._internal_depth == 0 and "/sandbox/workspace" in output:
            output = output.replace("/sandbox/workspace/", "").replace(
                "/sandbox/workspace", "."
            )
        # Non-zero exit in the container: the inherited _build_*_cmd scripts
        # append `2>/dev/null` + `|| true`, so a non-zero exit here is a real
        # failure (or a raw command the model ran) — surface it verbatim.
        return ExecuteResponse(output=output, exit_code=exit_code, truncated=False)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        out: list[FileUploadResponse] = []
        for path, data in files:
            # Tar-based upload via Docker put_archive. The old path ran
            # ``python3 -c <script> <path> <base64>`` with the base64 payload
            # as a CLI arg — which hit Linux ARG_MAX (~128KB) and broke the
            # compact/summarization middleware (conversation history offload),
            # twitter image posts, and any large upload. put_archive ships the
            # bytes as a tar HTTP body: no argv, no shell quoting, no ceiling.
            try:
                self._exec.upload_file(path, data)
                out.append(FileUploadResponse(path=path, error=None))
            except Exception as exc:  # noqa: BLE001 - surface, don't raise
                out.append(FileUploadResponse(
                    path=path, error=f"upload failed: {exc}"))
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

    # read/write/edit (+ async variants): OVERRIDDEN to set _internal_depth
    # so execute() knows NOT to strip /sandbox/workspace from their output
    # (file content would be corrupted). These methods call execute()
    # internally — the depth guard prevents the strip.
    def read(self, file_path: str, offset: int = 0, limit: int = 2000):
        self._internal_depth += 1
        try:
            return super().read(file_path, offset=offset, limit=limit)
        finally:
            self._internal_depth -= 1

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000):
        self._internal_depth += 1
        try:
            return await super().aread(file_path, offset=offset, limit=limit)
        finally:
            self._internal_depth -= 1

    def write(self, file_path: str, content: str):
        self._internal_depth += 1
        try:
            return super().write(file_path, content)
        finally:
            self._internal_depth -= 1

    async def awrite(self, file_path: str, content: str):
        self._internal_depth += 1
        try:
            return await super().awrite(file_path, content)
        finally:
            self._internal_depth -= 1

    def edit(self, file_path: str, old_string: str, new_string: str,
             replace_all: bool = False):
        self._internal_depth += 1
        try:
            return super().edit(file_path, old_string, new_string,
                                replace_all=replace_all)
        finally:
            self._internal_depth -= 1

    async def aedit(self, file_path: str, old_string: str, new_string: str,
                    replace_all: bool = False):
        self._internal_depth += 1
        try:
            return await super().aedit(file_path, old_string, new_string,
                                       replace_all=replace_all)
        finally:
            self._internal_depth -= 1

    # ls/glob/grep (+ async variants): OVERRIDDEN to (a) default an omitted
    # ``path`` to ``WORKSPACE_ROOT`` (``BaseSandbox.glob`` defaults to ``/`` →
    # full-container walk → 20s timeout), (b) set _internal_depth so execute()
    # doesn't corrupt their structured command output, and (c) strip
    # ``WORKSPACE_ROOT`` from result paths so the agent sees clean
    # project-relative paths like ``CANVAS-GAPS.md``.

    def ls(self, path: str):
        self._internal_depth += 1
        try:
            result = super().ls(path)
        finally:
            self._internal_depth -= 1
        if result.entries:
            _relativize_entries(result.entries)
        return result

    async def als(self, path: str):
        self._internal_depth += 1
        try:
            result = await super().als(path)
        finally:
            self._internal_depth -= 1
        if result.entries:
            _relativize_entries(result.entries)
        return result

    def glob(self, pattern: str, path: str | None = None):
        self._internal_depth += 1
        try:
            result = super().glob(pattern, path=path or WORKSPACE_ROOT)
        finally:
            self._internal_depth -= 1
        if result.matches:
            _relativize_entries(result.matches)
        return result

    async def aglob(self, pattern: str, path: str | None = None):
        self._internal_depth += 1
        try:
            result = await super().aglob(pattern, path=path or WORKSPACE_ROOT)
        finally:
            self._internal_depth -= 1
        if result.matches:
            _relativize_entries(result.matches)
        return result

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None):
        self._internal_depth += 1
        try:
            result = super().grep(pattern, path=path or WORKSPACE_ROOT, glob=glob)
        finally:
            self._internal_depth -= 1
        if result.matches:
            _relativize_entries(result.matches)
        return result

    async def agrep(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ):
        self._internal_depth += 1
        try:
            result = await super().agrep(pattern, path=path or WORKSPACE_ROOT,
                                         glob=glob)
        finally:
            self._internal_depth -= 1
        if result.matches:
            _relativize_entries(result.matches)
        return result
