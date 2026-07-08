"""Unit tests for ``pux_harness.lockfile`` — the org dep pin snapshot.

A lockfile pins an org's external deps for reproducibility: github MCP refs →
commit SHAs (best-effort ``git ls-remote``), ``sandbox.deps.{pip,apt}`` recorded
as-declared. These tests are FULLY OFFLINE: the subprocess runner is injected
(no real ``git ls-remote`` / network), and the org tree + catalog are synthetic
``tmp_path`` fixtures.

What's pinned here:
  - SHA resolution parses ``git ls-remote`` output correctly + never raises
    (offline / bad-ref / non-SHA → ``resolved: false``).
  - MCP refs are gathered from the org's policy ``tool_servers`` list × the
    shared catalog's ``github.repo`` entries (non-github servers skipped).
  - pip/apt deps recorded as-declared (verbatim, no resolution).
  - the on-disk YAML round-trips + ships in a pack (org.lock.yaml is a
    DEFAULT_INCLUDE — Decision 4).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from pux_harness import lockfile as lf


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeProc:
    """Stand-in for subprocess.run's CompletedProcess."""
    stdout: str
    returncode: int = 0


def _sha_runner_factory(sha: str | None, repo_to_sha: dict[str, str] | None = None):
    """Return a fake ``subprocess.run``: maps (repo, ref) → a ``git ls-remote``
    line ``"<sha>\trefs/heads/main"``. ``repo_to_sha`` gives per-repo SHAs;
    ``sha`` is the fallback. A returncode≠0 simulates a failed resolution."""
    def _runner(cmd, **kwargs):
        # cmd = ["git", "ls-remote", "https://github.com/<repo>", "<ref>"]
        url = cmd[2] if len(cmd) > 2 else ""
        repo = url.rsplit("/", 1)[-1] if "/" in url else url
        chosen = (repo_to_sha or {}).get(repo, sha)
        if chosen is None:
            return _FakeProc(stdout="", returncode=1)
        return _FakeProc(stdout=f"{chosen}\trefs/heads/main\n")
    return _runner


def _make_org(root: Path, name: str = "acme", *, tool_servers=None,
              pip=None, apt=None) -> Path:
    """Write a synthetic org tree under <root>/orgs/<name>/ with a policy.yaml
    + the shared catalog at <root>/orgs/_shared/tool_servers.yaml."""
    org_dir = root / "orgs" / name
    org_dir.mkdir(parents=True, exist_ok=True)
    (org_dir / "AGENTS.md").write_text(f"# {name}\n")
    sandbox = {}
    if pip is not None or apt is not None:
        sandbox["deps"] = {}
        if pip is not None:
            sandbox["deps"]["pip"] = pip
        if apt is not None:
            sandbox["deps"]["apt"] = apt
    policy: dict = {}
    if tool_servers is not None:
        policy["tool_servers"] = tool_servers
    if sandbox:
        policy["sandbox"] = sandbox
    if policy:
        (org_dir / "policy.yaml").write_text(yaml.safe_dump(policy))
    shared = root / "orgs" / "_shared"
    shared.mkdir(parents=True, exist_ok=True)
    return org_dir


def _write_catalog(root: Path, servers: dict) -> None:
    (root / "orgs" / "_shared" / "tool_servers.yaml").write_text(
        yaml.safe_dump(servers)
    )


# ---------------------------------------------------------------------------
# SHA resolution — best-effort, never raises
# ---------------------------------------------------------------------------

class TestResolveGithubSha:
    def test_parses_ls_remote_line(self):
        runner = _sha_runner_factory("a" * 40)
        assert lf.resolve_github_sha("o/r", "latest", runner=runner) == "a" * 40

    def test_latest_maps_to_HEAD_ref(self):
        """`version="latest"` queries HEAD, not the literal string 'latest'."""
        seen = {}
        def runner(cmd, **kw):
            seen["ref"] = cmd[-1]
            seen["url"] = cmd[2]
            return _FakeProc(stdout="b" * 40 + "\tHEAD\n")
        sha = lf.resolve_github_sha("o/r", "latest", runner=runner)
        assert sha == "b" * 40
        assert seen["ref"] == "HEAD"
        assert seen["url"] == "https://github.com/o/r"

    def test_pinned_version_is_passed_as_ref(self):
        seen = {}
        def runner(cmd, **kw):
            seen["ref"] = cmd[-1]
            return _FakeProc(stdout="c" * 40 + "\trefs/tags/v1.5.0\n")
        sha = lf.resolve_github_sha("o/r", "v1.5.0", runner=runner)
        assert sha == "c" * 40
        assert seen["ref"] == "v1.5.0"

    def test_failed_returncode_yields_none(self):
        runner = _sha_runner_factory(None)
        assert lf.resolve_github_sha("o/r", "latest", runner=runner) is None

    def test_empty_stdout_yields_none(self):
        runner = _sha_runner_factory(None)  # → returncode 1, empty
        assert lf.resolve_github_sha("o/r", "latest", runner=runner) is None

    def test_non_sha_output_yields_none(self):
        """Garbage (not a hex SHA) is rejected, not stored."""
        def runner(cmd, **kw):
            return _FakeProc(stdout="not-a-sha\tHEAD\n")
        assert lf.resolve_github_sha("o/r", "latest", runner=runner) is None

    def test_missing_git_binary_yields_none(self, monkeypatch):
        """No git on PATH → FileNotFoundError → None, never raises."""
        def boom(cmd, **kw):
            raise FileNotFoundError("git not found")
        assert lf.resolve_github_sha("o/r", "latest", runner=boom) is None

    def test_subprocess_timeout_yields_none(self):
        def boom(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 10)
        assert lf.resolve_github_sha("o/r", "latest", runner=boom) is None


# ---------------------------------------------------------------------------
# gather — MCP refs + declared deps
# ---------------------------------------------------------------------------

class TestCollectMcpRefs:
    def test_gathers_github_refs_from_catalog(self, tmp_path):
        root = tmp_path
        _make_org(root, tool_servers=["gh-server", "docker-server"])
        _write_catalog(root, {
            "gh-server": {"github": {"repo": "octo/srv", "version": "v1.2.0"}},
            "docker-server": {"docker": {"image": "foo:latest"}},  # not github
        })
        refs = lf.collect_mcp_refs(root / "orgs" / "acme", root)
        assert len(refs) == 1
        assert refs[0].name == "gh-server"
        assert refs[0].repo == "octo/srv"
        assert refs[0].version == "v1.2.0"
        assert refs[0].resolved is False  # not resolved until build_lock

    def test_latest_version_default_when_unspecified(self, tmp_path):
        root = tmp_path
        _make_org(root, tool_servers=["gh-server"])
        _write_catalog(root, {"gh-server": {"github": {"repo": "octo/srv"}}})
        refs = lf.collect_mcp_refs(root / "orgs" / "acme", root)
        assert refs[0].version == "latest"

    def test_no_tool_servers_block_yields_empty(self, tmp_path):
        root = tmp_path
        _make_org(root)  # no tool_servers
        _write_catalog(root, {"gh-server": {"github": {"repo": "octo/srv"}}})
        assert lf.collect_mcp_refs(root / "orgs" / "acme", root) == []

    def test_missing_catalog_entry_skipped(self, tmp_path):
        """An org referencing a catalog entry that doesn't exist is skipped
        (not an error — the catalog may be partial)."""
        root = tmp_path
        _make_org(root, tool_servers=["ghost"])
        _write_catalog(root, {})
        assert lf.collect_mcp_refs(root / "orgs" / "acme", root) == []


class TestCollectDeps:
    def test_records_pip_and_apt_as_declared(self, tmp_path):
        root = tmp_path
        _make_org(root, pip=["requests==2.31.0", "rich"], apt=["ffmpeg", "curl"])
        pip, apt = lf.collect_deps(root / "orgs" / "acme")
        assert pip == ["requests==2.31.0", "rich"]
        assert apt == ["ffmpeg", "curl"]

    def test_no_sandbox_block_yields_empty(self, tmp_path):
        root = tmp_path
        _make_org(root)
        pip, apt = lf.collect_deps(root / "orgs" / "acme")
        assert pip == [] and apt == []

    def test_no_deps_subkey_yields_empty(self, tmp_path):
        root = tmp_path
        org_dir = root / "orgs" / "acme"
        org_dir.mkdir(parents=True)
        (org_dir / "policy.yaml").write_text(yaml.safe_dump({"sandbox": {"image": "x"}}))
        pip, apt = lf.collect_deps(org_dir)
        assert pip == [] and apt == []


# ---------------------------------------------------------------------------
# build + serialize + write (offline; runner injected)
# ---------------------------------------------------------------------------

class TestBuildLock:
    def test_resolves_refs_and_records_deps(self, tmp_path):
        root = tmp_path
        _make_org(root, tool_servers=["gh"], pip=["rich"])
        _write_catalog(root, {"gh": {"github": {"repo": "octo/srv"}}})
        runner = _sha_runner_factory("d" * 40)
        lock = lf.build_lock("acme", root / "orgs" / "acme", root, runner=runner)
        assert lock.name == "acme"
        assert len(lock.mcp_servers) == 1
        assert lock.mcp_servers[0].sha == "d" * 40
        assert lock.mcp_servers[0].resolved is True
        assert lock.pip == ["rich"]
        assert lock.apt == []

    def test_unresolved_ref_recorded_honestly(self, tmp_path):
        """Offline / bad-ref → resolved=False, sha=None (not faked)."""
        root = tmp_path
        _make_org(root, tool_servers=["gh"])
        _write_catalog(root, {"gh": {"github": {"repo": "octo/srv"}}})
        runner = _sha_runner_factory(None)
        lock = lf.build_lock("acme", root / "orgs" / "acme", root, runner=runner)
        assert lock.mcp_servers[0].resolved is False
        assert lock.mcp_servers[0].sha is None

    def test_resolved_at_is_deterministic_when_now_injected(self, tmp_path):
        from datetime import datetime, timezone
        root = tmp_path
        _make_org(root)
        fixed = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
        lock = lf.build_lock("acme", root / "orgs" / "acme", root, now=fixed)
        assert lock.resolved_at == "2026-07-08T12:00:00+00:00"


class TestSerialize:
    def test_dict_shape_is_aps_rhyming(self, tmp_path):
        root = tmp_path
        _make_org(root, tool_servers=["gh"], pip=["rich"], apt=["curl"])
        _write_catalog(root, {"gh": {"github": {"repo": "octo/srv"}}})
        runner = _sha_runner_factory("e" * 40)
        lock = lf.build_lock("acme", root / "orgs" / "acme", root, runner=runner)
        d = lf.lock_to_dict(lock)
        assert d["lock_version"] == lf.LOCK_VERSION
        assert d["package"]["name"] == "acme"
        assert "resolved_at" in d
        assert d["dependencies"]["mcp_servers"] == [{
            "name": "gh", "repo": "octo/srv", "version": "latest",
            "sha": "e" * 40, "resolved": True,
        }]
        assert d["dependencies"]["pip"] == ["rich"]
        assert d["dependencies"]["apt"] == ["curl"]

    def test_render_lock_round_trips_through_yaml(self, tmp_path):
        root = tmp_path
        _make_org(root, tool_servers=["gh"])
        _write_catalog(root, {"gh": {"github": {"repo": "octo/srv", "version": "v1"}}})
        runner = _sha_runner_factory("f" * 40)
        lock = lf.build_lock("acme", root / "orgs" / "acme", root, runner=runner)
        text = lf.render_lock(lock)
        parsed = yaml.safe_load(text)
        assert parsed["lock_version"] == lf.LOCK_VERSION
        assert parsed["dependencies"]["mcp_servers"][0]["sha"] == "f" * 40


class TestWriteLock:
    def test_writes_org_lock_yaml(self, tmp_path):
        root = tmp_path
        org_dir = root / "orgs" / "acme"
        _make_org(root, pip=["rich"])
        lock = lf.build_lock("acme", org_dir, root,
                             runner=_sha_runner_factory(None))
        path = lf.write_lock(org_dir, lock)
        assert path == org_dir / "org.lock.yaml"
        assert path.is_file()
        parsed = yaml.safe_load(path.read_text())
        assert parsed["package"]["name"] == "acme"
        assert parsed["dependencies"]["pip"] == ["rich"]


# ---------------------------------------------------------------------------
# lock_org end-to-end (offline) + pack-shipping contract
# ---------------------------------------------------------------------------

class TestLockOrg:
    def test_lock_org_writes_file_resolving_org_path(self, tmp_path, monkeypatch):
        """``lock_org`` resolves the org via ``_org_path`` + project_root and
        writes org.lock.yaml. We point the kit's project_root at our tmp tree."""
        root = tmp_path
        _make_org(root, tool_servers=["gh"])
        _write_catalog(root, {"gh": {"github": {"repo": "octo/srv"}}})
        path = lf.lock_org(
            "acme", project_root=root, runner=_sha_runner_factory("1" * 40),
        )
        assert path.name == "org.lock.yaml"
        parsed = yaml.safe_load(path.read_text())
        assert parsed["dependencies"]["mcp_servers"][0]["resolved"] is True

    def test_lock_org_unknown_org_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            lf.lock_org("no-such-org", project_root=tmp_path,
                        runner=_sha_runner_factory(None))


class TestLockShipsInPack:
    """Decision 4: org.lock.yaml travels in the pack (DEFAULT_INCLUDE). A
    consumer gets the pinned deps the operator vetted."""

    def test_org_lock_yaml_is_default_include(self):
        from pux_harness.manifest import DEFAULT_INCLUDE
        assert "org.lock.yaml" in DEFAULT_INCLUDE

    def test_collect_pack_files_picks_up_lockfile(self, tmp_path):
        from pux_harness.manifest import collect_pack_files, load_manifest
        org_dir = tmp_path / "orgs" / "acme"
        _make_org(tmp_path)
        # Write a lockfile alongside the source primitives.
        lock = lf.OrgLock(name="acme")
        lf.write_lock(org_dir, lock)
        collected = collect_pack_files(org_dir, load_manifest(org_dir))
        rels = {Path(k).name for k in collected}
        assert "org.lock.yaml" in rels
