"""On-demand GitHub-release bootstrap for stdio MCP server binaries.

When a stdio ``ToolServerSpec`` carries a ``github:`` block, ``ensure_server``
makes sure the named binary is locally present — fetching it from the GitHub
release ONLY when ``shutil.which`` AND the local cache both miss. Idempotent: a
PATH-resolved binary or a warm cache short-circuits with zero network.

This is the FALLBACK path. The happy path is that the operator installed the
binary themselves (it's on PATH) and the harness never touches the network.

Cache layout: ``<project_root>/.pux/mcp-servers/<name>/<version>/<binary>`` —
the same ``.pux/`` app cache ``scripts/bootstrap-vision.sh`` uses
(``.pux/models/``). ``project_root()`` is the ONE shared app-root resolver.

Failure model: every failure is logged + returns ``None`` (never raises). The
caller (``McpSessionManager.open``) then lets the existing per-server probe fail
→ logged ERROR + zero tools + the org still starts. A broken download is
indistinguishable from a broken server, which the harness already handles
gracefully (per-server isolation).

Auth: the GitHub releases API is called with a Bearer ``GITHUB_TOKEN`` (or
``GITHUB_PERSONAL_ACCESS_TOKEN``) if present — 5000 req/hr vs 60 unauth. The
token is NOT required (public releases work unauth); it only lifts rate limits.
The server binary's OWN auth (its PAT) is a separate concern handled by the
spec's ``env`` (see ``tool_servers._substitute_spec``).
"""
from __future__ import annotations

import contextlib
import logging
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import zipfile
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import httpx

from pux_harness.agent.tool_servers import ToolServerSpec
from pux_harness.kit._paths import project_root

_log = logging.getLogger(__name__)

_API = "https://api.github.com"
_TIMEOUT = 60.0


def platform_tokens() -> dict[str, str]:
    """Map this host to the ``{os}``/``{arch}`` tokens used in a ``github:``
    asset glob. github-mcp-server asset naming is ``<name>_Linux_x86_64.tar.gz``
    — capitalized OS, gnu-ish arch."""
    osname = {
        "linux": "Linux",
        "darwin": "Darwin",
        "win32": "Windows",
    }.get(sys.platform, sys.platform.title())
    machine = platform.machine().lower()
    arch = {
        "x86_64": "x86_64", "amd64": "x86_64",
        "aarch64": "arm64", "arm64": "arm64",
        "i386": "i386", "i686": "i386",
        "riscv64": "riscv64",
    }.get(machine, machine)
    return {"os": osname, "arch": arch}


def select_asset(asset_glob: str, asset_names: list[str]) -> str | None:
    """Render the ``{os}``/``{arch}`` tokens into the glob and return the single
    matching release asset name. ``None`` on zero or ambiguous match (fail-safe:
    never guess which binary to run)."""
    tokens = platform_tokens()
    pat = asset_glob.format(**tokens)
    matches = [n for n in asset_names if fnmatchcase(n, pat)]
    if len(matches) != 1:
        return None
    return matches[0]


def _extract_binary(archive: Path, binary: str, dest_dir: Path) -> Path | None:
    """Extract ONLY the named binary from a ``.tar.gz``/``.zip`` archive into
    ``dest_dir`` (flattened — no subdirs). Returns its path, or ``None`` if the
    binary isn't in the archive. Reads via ``extractfile``/``read`` (no member
    extraction) so there's no path-traversal / no ``filter=`` deprecation."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / binary
    name = archive.name.lower()
    try:
        if name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive, "r:gz") as tar:
                members = [
                    m for m in tar.getmembers()
                    if Path(m.name).name == binary and m.isfile()
                ]
                if not members:
                    return None
                data = tar.extractfile(members[0])
                if data is None:
                    return None
                out.write_bytes(data.read())
        elif name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                names = [n for n in zf.namelist() if Path(n).name == binary]
                if not names:
                    return None
                out.write_bytes(zf.read(names[0]))
        else:
            _log.error("mcp_bootstrap: unknown archive type: %s", archive.name)
            return None
    except Exception as e:  # noqa: BLE001 — any archive failure → None
        _log.error("mcp_bootstrap: extract failed for %s: %s", archive.name, e)
        return None
    if not out.is_file():
        return None
    os.chmod(out, 0o755)
    return out


def _api_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get(
        "GITHUB_PERSONAL_ACCESS_TOKEN"
    )
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def fetch_release(repo: str, version: str) -> dict[str, Any] | None:
    """GET the release metadata (``latest`` → ``/releases/latest``; else
    ``/releases/tags/<version>``). ``None`` on any network/HTTP failure."""
    url = (
        f"{_API}/repos/{repo}/releases/latest"
        if version == "latest"
        else f"{_API}/repos/{repo}/releases/tags/{version}"
    )
    try:
        r = httpx.get(url, headers=_api_headers(), timeout=_TIMEOUT,
                      follow_redirects=True)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001 — any failure → None (caller skips)
        _log.error("mcp_bootstrap: release fetch failed for %s@%s: %s",
                   repo, version, e)
        return None


def _download(url: str, dest: Path) -> bool:
    """Stream ``url`` to ``dest``. ``True`` on success. ``False`` (logged) on any
    failure — a partial file is removed so a retry isn't poisoned."""
    try:
        with httpx.stream("GET", url, timeout=_TIMEOUT,
                          follow_redirects=True) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return True
    except Exception as e:  # noqa: BLE001
        _log.error("mcp_bootstrap: download failed %s: %s", url, e)
        with contextlib.suppress(OSError):
            dest.unlink()
        return False


def ensure_server(spec: ToolServerSpec) -> Path | None:
    """Ensure the stdio binary for *spec* is locally present; return its absolute
    path or ``None``. ``None`` for non-stdio specs or specs with no ``github``
    block (the caller leaves ``spec.command`` alone in that case).

    Resolution order: (1) ``shutil.which`` — the happy PATH-installed path, zero
    network; (2) the local cache — warm ``.pux/mcp-servers/...``, zero network;
    (3) the FALLBACK download from the GitHub release. Extraction is atomic
    (scratch dir → rename) so the cache never holds a partial binary.
    """
    if spec.transport != "stdio" or not spec.github:
        return None
    gh = spec.github
    binary = gh["binary"]
    version = gh["version"]
    cache_root = project_root() / ".pux" / "mcp-servers" / spec.name / version
    cached = cache_root / binary

    # (1) happy path — already on PATH.
    on_path = shutil.which(spec.command)
    if on_path:
        return Path(on_path)
    # (2) cache hit — a complete binary from a prior fetch.
    if cached.is_file() and os.access(cached, os.X_OK):
        return cached

    # (3) FALLBACK — fetch from the release.
    release = fetch_release(gh["repo"], version)
    if not release:
        return None
    assets = release.get("assets", []) or []
    asset_name = select_asset(gh["asset"], [a.get("name", "") for a in assets])
    if not asset_name:
        _log.error(
            "mcp_bootstrap: no release asset matched %r for %s@%s (assets: %s)",
            gh["asset"], spec.name, version,
            [a.get("name") for a in assets][:10],
        )
        return None
    asset_url = next(
        (a.get("browser_download_url") for a in assets
         if a.get("name") == asset_name),
        None,
    )
    if not asset_url:
        return None

    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        archive = td_path / asset_name
        if not _download(asset_url, archive):
            return None
        scratch = td_path / "out"
        scratch.mkdir()
        extracted = _extract_binary(archive, binary, scratch)
        if not extracted:
            _log.error("mcp_bootstrap: %s not found in asset %s",
                       binary, asset_name)
            return None
        final = cache_root / binary
        # atomic: extract into scratch, then move into the cache so a failed
        # run never leaves a partial binary at the cache path.
        shutil.move(str(extracted), str(final))
    os.chmod(final, 0o755)
    _log.info("mcp_bootstrap: fetched %s %s -> %s", spec.name, version, final)
    return final
