"""mcp_bootstrap — the GitHub-release FALLBACK for stdio MCP binaries (Phase B).

``ensure_server`` is the contract: make the named binary locally present, hitting
the network ONLY when ``shutil.which`` and the cache both miss. These tests prove
each branch of that resolution order + the helpers it composes, with NO real
network (httpx is mocked) and a REAL tar.gz/zip fixture (so extraction + chmod +
atomic-move are proven against the actual archive types github-mcp-server ships).

The live download + handshake is Phase E; this file is the deterministic unit
proof the fallback rests on.
"""
from __future__ import annotations

import contextlib
import io
import os
import platform as _platform_mod
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from pux_harness.agent import mcp_bootstrap as mb
from pux_harness.agent.tool_servers import ToolServerSpec


# --- fakes -------------------------------------------------------------------

class _FakeResp:
    def __init__(self, *, payload=None, error=None, chunks=None,
                 chunk_error=None):
        self._payload = payload
        self._error = error
        self._chunks = chunks
        self._chunk_error = chunk_error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._payload

    def iter_bytes(self):
        for c in self._chunks or []:
            yield c
        if self._chunk_error:
            raise self._chunk_error


def _spec(*, command="github-mcp-server", github=None, transport="stdio",
          name="github") -> ToolServerSpec:
    return ToolServerSpec(
        name=name, kind="mcp", transport=transport, command=command,
        args=["stdio"], env={}, github=github,
    )


def _gh():
    return {
        "repo": "github/github-mcp-server",
        "asset": "github-mcp-server_*{os}*{arch}*.tar.gz",
        "binary": "github-mcp-server",
        "version": "latest",
    }


def _make_targz(path: Path, binary: str = "github-mcp-server") -> bytes:
    """Build an in-memory tar.gz carrying the binary + junk, return its bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def _add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        _add(binary, b"#!/bin/sh\necho fake-mcp\n")
        _add("LICENSE", b"MIT-ish\n")
        _add("docs/README.md", b"# github-mcp-server\n")
    data = buf.getvalue()
    path.write_bytes(data)
    return data


def _make_zip(path: Path, binary: str = "github-mcp-server") -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(binary, b"#!/bin/sh\necho fake-mcp\n")
        zf.writestr("LICENSE", b"MIT-ish\n")


# --- platform_tokens ---------------------------------------------------------

def test_platform_tokens_linux_x86_64(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_platform_mod, "machine", lambda: "x86_64")
    assert mb.platform_tokens() == {"os": "Linux", "arch": "x86_64"}


def test_platform_tokens_linux_arm64(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_platform_mod, "machine", lambda: "aarch64")
    assert mb.platform_tokens() == {"os": "Linux", "arch": "arm64"}


def test_platform_tokens_darwin_arm64(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(_platform_mod, "machine", lambda: "arm64")
    assert mb.platform_tokens() == {"os": "Darwin", "arch": "arm64"}


def test_platform_tokens_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(_platform_mod, "machine", lambda: "x86_64")
    assert mb.platform_tokens()["os"] == "Windows"


# --- select_asset (real github-mcp-server asset names) -----------------------

_REAL_ASSETS = [
    "github-mcp-server_Darwin_arm64.tar.gz",
    "github-mcp-server_Darwin_x86_64.tar.gz",
    "github-mcp-server_Linux_arm64.tar.gz",
    "github-mcp-server_Linux_x86_64.tar.gz",
    "github-mcp-server_Windows_arm64.zip",
    "github-mcp-server_Windows_x86_64.zip",
]


def test_select_asset_linux_x86_64(monkeypatch):
    monkeypatch.setattr(mb, "platform_tokens",
                        lambda: {"os": "Linux", "arch": "x86_64"})
    assert mb.select_asset(
        "github-mcp-server_*{os}*{arch}*.tar.gz", _REAL_ASSETS,
    ) == "github-mcp-server_Linux_x86_64.tar.gz"


def test_select_asset_linux_arm64(monkeypatch):
    monkeypatch.setattr(mb, "platform_tokens",
                        lambda: {"os": "Linux", "arch": "arm64"})
    assert mb.select_asset(
        "github-mcp-server_*{os}*{arch}*.tar.gz", _REAL_ASSETS,
    ) == "github-mcp-server_Linux_arm64.tar.gz"


def test_select_asset_windows_zip(monkeypatch):
    monkeypatch.setattr(mb, "platform_tokens",
                        lambda: {"os": "Windows", "arch": "x86_64"})
    assert mb.select_asset(
        "github-mcp-server_*{os}*{arch}*.zip", _REAL_ASSETS,
    ) == "github-mcp-server_Windows_x86_64.zip"


def test_select_asset_no_match_returns_none(monkeypatch):
    monkeypatch.setattr(mb, "platform_tokens",
                        lambda: {"os": "Linux", "arch": "x86_64"})
    assert mb.select_asset("*.zip", _REAL_ASSETS) is None  # no linux zip


def test_select_asset_ambiguous_returns_none(monkeypatch):
    monkeypatch.setattr(mb, "platform_tokens",
                        lambda: {"os": "Linux", "arch": "x86_64"})
    assets = ["a_Linux_x86_64.tar.gz", "b_Linux_x86_64.tar.gz"]
    assert mb.select_asset("*{os}*{arch}*.tar.gz", assets) is None


# --- _extract_binary (real archives) ----------------------------------------

def test_extract_binary_from_targz_picks_binary_and_chmods(tmp_path):
    archive = tmp_path / "release.tar.gz"
    _make_targz(archive)
    out_dir = tmp_path / "out"
    out = mb._extract_binary(archive, "github-mcp-server", out_dir)
    assert out is not None and out.is_file()
    assert out.name == "github-mcp-server"
    # ONLY the binary lands in out_dir — LICENSE/docs are not extracted.
    assert sorted(p.name for p in out_dir.iterdir()) == ["github-mcp-server"]
    assert out.stat().st_mode & stat.S_IXUSR  # executable


def test_extract_binary_from_zip(tmp_path):
    archive = tmp_path / "release.zip"
    _make_zip(archive)
    out_dir = tmp_path / "out"
    out = mb._extract_binary(archive, "github-mcp-server", out_dir)
    assert out is not None and out.is_file()
    assert out.stat().st_mode & stat.S_IXUSR


def test_extract_binary_missing_from_archive_returns_none(tmp_path):
    archive = tmp_path / "release.tar.gz"
    _make_targz(archive)
    out = mb._extract_binary(archive, "not-the-binary", tmp_path / "out")
    assert out is None


def test_extract_binary_unknown_archive_type_returns_none(tmp_path):
    archive = tmp_path / "release.foo"
    archive.write_bytes(b"junk")
    out = mb._extract_binary(archive, "github-mcp-server", tmp_path / "out")
    assert out is None


# --- ensure_server resolution order ------------------------------------------

def test_ensure_server_non_stdio_returns_none():
    spec = _spec(transport="http", github=_gh())
    assert mb.ensure_server(spec) is None


def test_ensure_server_no_github_block_returns_none():
    spec = _spec(transport="stdio", github=None)
    assert mb.ensure_server(spec) is None


def test_ensure_server_path_resolved_short_circuits_no_network(monkeypatch):
    """shutil.which hit → return it; fetch_release must NOT be called."""
    spec = _spec(github=_gh())
    monkeypatch.setattr(mb.shutil, "which", lambda cmd: "/usr/local/bin/github-mcp-server")
    def _boom(*a, **k):
        raise AssertionError("fetch_release must not run when binary is on PATH")
    monkeypatch.setattr(mb, "fetch_release", _boom)
    assert mb.ensure_server(spec) == Path("/usr/local/bin/github-mcp-server")


def test_ensure_server_cache_hit_short_circuits_no_network(monkeypatch, tmp_path):
    """A complete cached binary → returned; fetch_release must NOT be called."""
    spec = _spec(github=_gh())
    monkeypatch.setattr(mb, "project_root", lambda: tmp_path)
    monkeypatch.setattr(mb.shutil, "which", lambda cmd: None)
    cache_bin = (tmp_path / ".pux" / "mcp-servers" / "github" / "latest"
                 / "github-mcp-server")
    cache_bin.parent.mkdir(parents=True)
    cache_bin.write_bytes(b"#!/bin/sh\necho cached\n")
    os.chmod(cache_bin, 0o755)
    def _boom(*a, **k):
        raise AssertionError("fetch_release must not run on a cache hit")
    monkeypatch.setattr(mb, "fetch_release", _boom)
    assert mb.ensure_server(spec) == cache_bin


def test_ensure_server_downloads_and_caches(monkeypatch, tmp_path):
    """Cold cache + binary missing → the FALLBACK fetches, extracts, and the
    cached binary is executable + returned."""
    spec = _spec(github=_gh())
    monkeypatch.setattr(mb, "project_root", lambda: tmp_path)
    monkeypatch.setattr(mb.shutil, "which", lambda cmd: None)

    release = {"assets": [
        {"name": "github-mcp-server_Linux_x86_64.tar.gz",
         "browser_download_url": "https://codeload/x"},
    ]}
    monkeypatch.setattr(mb, "fetch_release", lambda repo, ver: release)
    monkeypatch.setattr(mb, "platform_tokens",
                        lambda: {"os": "Linux", "arch": "x86_64"})

    def _fake_download(url, dest):
        _make_targz(dest)  # write a real fixture tarball to dest
        return True
    monkeypatch.setattr(mb, "_download", _fake_download)

    out = mb.ensure_server(spec)
    expected = (tmp_path / ".pux" / "mcp-servers" / "github" / "latest"
                / "github-mcp-server")
    assert out == expected
    assert out.is_file()
    assert out.stat().st_mode & stat.S_IXUSR
    assert out.read_bytes().startswith(b"#!/bin/sh")


def test_ensure_server_fetch_release_failure_returns_none(monkeypatch, tmp_path):
    spec = _spec(github=_gh())
    monkeypatch.setattr(mb, "project_root", lambda: tmp_path)
    monkeypatch.setattr(mb.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(mb, "fetch_release", lambda repo, ver: None)
    assert mb.ensure_server(spec) is None


def test_ensure_server_no_asset_match_returns_none(monkeypatch, tmp_path):
    spec = _spec(github=_gh())
    monkeypatch.setattr(mb, "project_root", lambda: tmp_path)
    monkeypatch.setattr(mb.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(mb, "fetch_release",
                        lambda repo, ver: {"assets": [{"name": "other.tar.gz"}]})
    monkeypatch.setattr(mb, "platform_tokens",
                        lambda: {"os": "Linux", "arch": "x86_64"})
    assert mb.ensure_server(spec) is None


def test_ensure_server_download_failure_returns_none(monkeypatch, tmp_path):
    spec = _spec(github=_gh())
    monkeypatch.setattr(mb, "project_root", lambda: tmp_path)
    monkeypatch.setattr(mb.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(mb, "fetch_release", lambda repo, ver: {"assets": [
        {"name": "github-mcp-server_Linux_x86_64.tar.gz",
         "browser_download_url": "https://x"}]})
    monkeypatch.setattr(mb, "platform_tokens",
                        lambda: {"os": "Linux", "arch": "x86_64"})
    monkeypatch.setattr(mb, "_download", lambda url, dest: False)
    assert mb.ensure_server(spec) is None
    # and no partial binary was left in the cache.
    cache = tmp_path / ".pux" / "mcp-servers" / "github" / "latest"
    assert not (cache / "github-mcp-server").exists()


def test_ensure_server_pinned_version_paths_cache_dir(monkeypatch, tmp_path):
    """A pinned version (not 'latest') shapes the cache dir + the fetch URL."""
    spec = _spec(github={**_gh(), "version": "v1.5.0"})
    monkeypatch.setattr(mb, "project_root", lambda: tmp_path)
    monkeypatch.setattr(mb.shutil, "which", lambda cmd: None)
    captured = {}
    def _fetch(repo, ver):
        captured["repo"], captured["ver"] = repo, ver
        return {"assets": [
            {"name": "github-mcp-server_Linux_x86_64.tar.gz",
             "browser_download_url": "https://x"}]}
    monkeypatch.setattr(mb, "fetch_release", _fetch)
    monkeypatch.setattr(mb, "platform_tokens",
                        lambda: {"os": "Linux", "arch": "x86_64"})
    monkeypatch.setattr(mb, "_download",
                        lambda url, dest: (_make_targz(dest), True)[1])
    out = mb.ensure_server(spec)
    assert "v1.5.0" in str(out)
    assert captured == {"repo": "github/github-mcp-server", "ver": "v1.5.0"}


# --- the httpx seams (fetch_release / _download) -----------------------------

def test_fetch_release_uses_token_and_latest_url(monkeypatch):
    calls = {}
    monkeypatch.setenv("GITHUB_TOKEN", "sek")
    def _fake_get(url, **kwargs):
        calls["url"] = url
        calls["headers"] = kwargs.get("headers", {})
        return _FakeResp(payload={"assets": []})
    monkeypatch.setattr(mb.httpx, "get", _fake_get)
    assert mb.fetch_release("github/github-mcp-server", "latest") == {"assets": []}
    assert calls["url"].endswith("/repos/github/github-mcp-server/releases/latest")
    assert calls["headers"]["Authorization"] == "Bearer sek"


def test_fetch_release_tag_url_for_pinned_version(monkeypatch):
    calls = {}
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    def _fake_get(url, **kwargs):
        calls["url"] = url
        calls["headers"] = kwargs.get("headers", {})
        return _FakeResp(payload={"assets": []})
    monkeypatch.setattr(mb.httpx, "get", _fake_get)
    mb.fetch_release("o/r", "v1.2.3")
    assert calls["url"].endswith("/repos/o/r/releases/tags/v1.2.3")
    assert "Authorization" not in calls["headers"]  # no token → no header


def test_fetch_release_network_failure_returns_none(monkeypatch):
    def _fake_get(url, **kwargs):
        return _FakeResp(error=RuntimeError("boom"))
    monkeypatch.setattr(mb.httpx, "get", _fake_get)
    assert mb.fetch_release("o/r", "latest") is None


def test_download_writes_bytes(monkeypatch, tmp_path):
    dest = tmp_path / "asset.tar.gz"
    payload = b"BINARY-DATA"
    @contextlib.contextmanager
    def _fake_stream(method, url, **kwargs):
        yield _FakeResp(chunks=[payload[:4], payload[4:]])
    monkeypatch.setattr(mb.httpx, "stream", _fake_stream)
    assert mb._download("https://x", dest) is True
    assert dest.read_bytes() == payload


def test_download_failure_removes_partial(monkeypatch, tmp_path):
    dest = tmp_path / "asset.tar.gz"
    @contextlib.contextmanager
    def _fake_stream(method, url, **kwargs):
        # yields a chunk then breaks mid-stream
        yield _FakeResp(chunks=[b"partial"], chunk_error=RuntimeError("conn reset"))
    monkeypatch.setattr(mb.httpx, "stream", _fake_stream)
    assert mb._download("https://x", dest) is False
    assert not dest.exists()  # partial bytes were removed
