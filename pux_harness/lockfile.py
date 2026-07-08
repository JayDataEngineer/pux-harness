"""Declarative org lockfile — pin an org's external deps for reproducibility.

``pux lock --org X`` snapshots the org's declared external dependencies into
``orgs/<org>/org.lock.yaml`` (committed by default — Decision 4) so the org's
dep set is reproducible + auditable across time, not "whatever was on the
registry when you ran it".

Reuse-first ([[rely-on-upstream]]): github MCP refs resolve to commit SHAs via
the ``git ls-remote`` CLI (host-side shell-out — NOT a python lib).
``sandbox.deps.{apt,pip}`` are recorded AS-DECLARED (operators already pin the
critical ones; full uv-style version resolution is a future step — recorded
honestly as ``resolved: as_declared``, never faked).

Best-effort resolution: a missing ``git``, no network, or a bad ref records the
entry as ``resolved: false`` rather than raising — a lockfile is ALWAYS
writable, online or offline. The lock is a snapshot, not a build step.

Schema (APS-shaped to rhyme with the manifest — [[plan-dynamic-tools-and-export]]):
  lock_version: "1.0"
  package: {name, version}
  resolved_at: <iso8601 utc>
  dependencies:
    mcp_servers: [{name, repo, version, sha, resolved}]
    pip: [...]   # as-declared from sandbox.deps.pip
    apt: [...]   # as-declared from sandbox.deps.apt

The lockfile ships in the pack (``org.lock.yaml`` is a DEFAULT_INCLUDE) so a
consumer gets the EXACT pinned deps the operator vetted.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

LOCK_VERSION = "1.0"

# A git commit SHA is 40 lowercase hex chars (loosely accept 7+ for short SHAs
# that ls-remote never emits, but be strict about the real 40-char form).
_SHA_MIN = 7


@dataclass
class McpRef:
    """A github-sourced MCP server the org consumes — resolved to a SHA when
    reachable, recorded unresolved otherwise."""

    name: str
    repo: str                 # owner/name
    version: str = "latest"   # "latest" → HEAD; else a tag/branch ref
    sha: str | None = None
    resolved: bool = False


@dataclass
class OrgLock:
    lock_version: str = LOCK_VERSION
    name: str = ""
    version: str = "0.0.0"
    resolved_at: str = ""
    mcp_servers: list[McpRef] = field(default_factory=list)
    pip: list[str] = field(default_factory=list)
    apt: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# github SHA resolution (host-side git shell-out — best-effort, never raises)
# ---------------------------------------------------------------------------

def _is_sha(token: str) -> bool:
    return len(token) >= _SHA_MIN and all(c in "0123456789abcdef" for c in token.lower())


def resolve_github_sha(
    repo: str, version: str, *, runner: Callable | None = None, timeout: float = 10.0
) -> str | None:
    """Resolve ``owner/name`` + ``version`` to a commit SHA via ``git ls-remote``.

    ``version="latest"`` → the default branch tip (``HEAD``); any other value is
    treated as a tag/branch ref. Returns ``None`` on ANY failure (no git binary,
    no network, bad ref, non-SHA output) — the caller records ``resolved=False``.
    ``runner`` is the subprocess runner (injectable for tests; defaults to
    :func:`subprocess.run`)."""
    run = runner or subprocess.run
    ref = "HEAD" if version == "latest" else version
    url = f"https://github.com/{repo}"
    try:
        out = run(
            ["git", "ls-remote", url, ref],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    stdout = getattr(out, "stdout", "") or ""
    returncode = getattr(out, "returncode", 1)
    if returncode != 0 or not stdout.strip():
        return None
    # Output: "<sha>\t<refname>" per line. Take the first token of the first line.
    first_line = stdout.splitlines()[0]
    token = first_line.split()[0] if first_line.split() else ""
    return token if _is_sha(token) else None


# ---------------------------------------------------------------------------
# gather the org's declared external deps (raw YAML — no env substitution)
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _as_str_list(val: Any) -> list[str]:
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    return []


def collect_mcp_refs(org_dir: Path, project_root: Path) -> list[McpRef]:
    """The org's github MCP refs: from policy's ``tool_servers`` list → the
    shared catalog entries that declare a ``github.repo``. Non-github servers
    (docker compose, stdio binary on PATH) are skipped — they have no mutable
    ref to pin."""
    refs: list[McpRef] = []
    policy = _load_yaml(org_dir / "policy.yaml")
    server_names = _as_str_list(policy.get("tool_servers"))
    if not server_names:
        return refs
    catalog_path = project_root / "orgs" / "_shared" / "tool_servers.yaml"
    catalog = _load_yaml(catalog_path)
    for name in server_names:
        entry = catalog.get(name)
        if not isinstance(entry, dict):
            continue
        gh = entry.get("github")
        if not isinstance(gh, dict) or not gh.get("repo"):
            continue
        refs.append(McpRef(
            name=name,
            repo=str(gh["repo"]),
            version=str(gh.get("version", "latest")),
        ))
    return refs


def collect_deps(org_dir: Path) -> tuple[list[str], list[str]]:
    """``sandbox.deps.{pip,apt}`` as-declared (raw — operators pin critical
    ones; the lock records the declaration verbatim)."""
    policy = _load_yaml(org_dir / "policy.yaml")
    sandbox = policy.get("sandbox") or {}
    deps = sandbox.get("deps") or {}
    if not isinstance(deps, dict):
        return [], []
    return _as_str_list(deps.get("pip")), _as_str_list(deps.get("apt"))


# ---------------------------------------------------------------------------
# build + serialize
# ---------------------------------------------------------------------------

def build_lock(
    org: str, org_dir: Path, project_root: Path, *,
    now: datetime | None = None, runner: Callable | None = None,
) -> OrgLock:
    """Assemble an :class:`OrgLock`: gather MCP refs + deps, resolve github
    SHAs best-effort. ``now`` is injectable (defaults to UTC now) so tests are
    deterministic; ``runner`` injects the subprocess runner for offline tests."""
    now = now or datetime.now(timezone.utc)
    mcp_refs = collect_mcp_refs(org_dir, project_root)
    for ref in mcp_refs:
        ref.sha = resolve_github_sha(ref.repo, ref.version, runner=runner)
        ref.resolved = ref.sha is not None
    pip, apt = collect_deps(org_dir)
    return OrgLock(
        name=org,
        resolved_at=now.isoformat(),
        mcp_servers=mcp_refs,
        pip=pip,
        apt=apt,
    )


def lock_to_dict(lock: OrgLock) -> dict[str, Any]:
    """Serialize to the org.lock.yaml shape (APS-shaped — rhymes with the
    manifest's package/capabilities/dependencies)."""
    return {
        "lock_version": lock.lock_version,
        "package": {"name": lock.name, "version": lock.version},
        "resolved_at": lock.resolved_at,
        "dependencies": {
            "mcp_servers": [
                {
                    "name": r.name,
                    "repo": r.repo,
                    "version": r.version,
                    "sha": r.sha,
                    "resolved": r.resolved,
                }
                for r in lock.mcp_servers
            ],
            "pip": list(lock.pip),
            "apt": list(lock.apt),
        },
    }


def render_lock(lock: OrgLock) -> str:
    """YAML text for the lockfile (block style, insertion-ordered, readable)."""
    return yaml.safe_dump(lock_to_dict(lock), sort_keys=False, default_flow_style=False)


def write_lock(org_dir: Path, lock: OrgLock) -> Path:
    """Write ``<org_dir>/org.lock.yaml``; return its path."""
    path = org_dir / "org.lock.yaml"
    path.write_text(render_lock(lock))
    return path


def lock_org(
    org: str, project_root: Path | None = None, *,
    now: datetime | None = None, runner: Callable | None = None,
) -> Path:
    """``pux lock --org X``: resolve + write the org's lockfile. Returns the
    written path. ``project_root`` drives BOTH org resolution and the catalog
    lookup (mirrors how ``pack_org`` threads a foreign root). Raises
    ``FileNotFoundError`` if the org dir is absent under that root."""
    from pux_harness.kit._paths import (
        project_root as _default_project_root,
        search_org_dir,
    )

    root = Path(project_root) if project_root is not None else _default_project_root()
    org_dir = search_org_dir(org, root)  # raises FileNotFoundError if absent
    lock = build_lock(org, org_dir, root, now=now, runner=runner)
    return write_lock(org_dir, lock)
