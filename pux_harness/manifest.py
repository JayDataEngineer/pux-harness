"""Declarative pack manifest — what an org ships is DECLARED, not implicit.

Replaces the old ``export._collect_org_files`` hardcoded-dir allowlist with a
**default-deny** collection driven by ``package.include`` globs in ``org.yaml``
(the npm ``files`` / Python wheel ``MANIFEST.in`` model). Anything NOT matching
an include glob is left behind; ``data/`` and ``.pux/`` are PERMANENT excludes
(the credential-leak contract — belt-and-suspenders on top of default-deny).

The schema is **pux-owned but APS-shaped** (``manifest_version`` / ``name`` /
``version`` / ``package`` / ``capabilities`` / ``dependencies``) so a future flip
to full APS conformance is a config change, not a rewrite (Decision 1). P3 only
EXERCISES ``package.{include,exclude}`` for collection; ``capabilities`` +
``dependencies`` are parsed and surfaced into the archive's ``manifest.json`` as
the receiver's audit surface (the receiver inspects them BEFORE running — they
are NOT inferred from code).

This module is pure data: no tarball, no exec, no Docker — fully unit-testable.
"""
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The manifest schema version. Bump on a breaking schema change (field
# renames/removals). Consumers gate on this.
MANIFEST_VERSION = "2.0"


# Default-DENY include globs when an org's ``org.yaml`` has NO ``package:``
# block. This is the OLD ``_collect_org_files`` allowlist (core files + the
# agents/skills/sandbox/config recursive dirs) now expressed DECLARATIVELY, plus
# ``lib/**`` (level (c) dynamic tools — they travel in the pack per P1/P2).
# Anything under the org tree NOT matched here is left behind by construction.
DEFAULT_INCLUDE: tuple[str, ...] = (
    "AGENTS.md",
    "org.yaml",
    "profile.yaml",
    "policy.yaml",
    "Dockerfile",
    "agents/**",
    "skills/**",
    "sandbox/**",
    "config/**",
    "lib/**",            # level (c): functions/, index.yaml, .archive/ (agent-authored)
    "org.lock.yaml",     # P3 lockfile: pinned MCP SHAs + declared deps (Decision 4)
)

# PERMANENT excludes — applied AFTER include matching and ALWAYS win, even if an
# org-declared include glob would otherwise match. ``data/`` holds live secrets
# (browser-session cookies, API tokens); ``.pux/`` is transient runtime state;
# ``__pycache__``/``*.pyc`` are build noise. This is the new permanent form of
# the ``data/`` exclusion contract (previously a hand-comment in the allowlist).
HARD_EXCLUDE: tuple[str, ...] = (
    "data/**",
    ".pux/**",
    "**/__pycache__/**",
    "**/*.pyc",
)


@dataclass
class PackageSpec:
    """The ``package:`` block — what ships + package identity."""

    name: str               # defaults to the org slug at load time
    version: str = "0.0.0"
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)   # org-declared extras


@dataclass
class Manifest:
    """A loaded pack manifest (parsed from ``org.yaml``'s top-level keys)."""

    manifest_version: str = MANIFEST_VERSION
    package: PackageSpec = field(default_factory=lambda: PackageSpec(name=""))
    # Raw audit surface — parsed but NOT validated in P3 (deep validation is the
    # capability-unification concern). Surfaced verbatim into manifest.json.
    capabilities: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, Any] = field(default_factory=dict)
    # "default" (no package: block → DEFAULT_INCLUDE) | "declared" (org.yaml has one)
    source: str = "default"


def _match_any(rel: str, patterns) -> bool:
    """True if ``rel`` (posix, org-relative) matches any glob in ``patterns``.

    Uses :func:`fnmatch.fnmatchcase` — case-sensitive (deterministic across
    platforms). fnmatch's ``*`` spans ``/``, so ``agents/**`` matches at any
    depth (``agents/a``, ``agents/a/b``); ``**`` is just two ``*`` and behaves
    the same as one. A bare ``AGENTS.md`` matches only an org-root file of that
    exact name (fnmatch anchors the whole string)."""
    return any(fnmatch.fnmatchcase(rel, p) for p in patterns)


def load_manifest(org_dir: Path, org_name: str | None = None) -> Manifest:
    """Load an org's pack manifest from ``<org_dir>/org.yaml``.

    If ``org.yaml`` has no ``package:`` block, returns a Manifest with
    :data:`DEFAULT_INCLUDE` globs (the no-op-default path — collection behaves
    exactly like the legacy allowlist, now declaratively). If it has one, the
    declared ``include``/``exclude`` globs drive collection (default-deny against
    the org's own declaration). :data:`HARD_EXCLUDE` is ALWAYS merged in.

    ``org_name`` defaults to ``org_dir.name`` and seeds ``package.name`` when the
    org doesn't declare one. Lenient: a malformed ``org.yaml`` falls back to the
    default manifest (pack never dies on a bad manifest block — it ships the
    safe default rather than nothing).
    """
    name = org_name or org_dir.name
    default = Manifest(
        package=PackageSpec(name=name, include=list(DEFAULT_INCLUDE)),
        source="default",
    )

    org_yaml = org_dir / "org.yaml"
    if not org_yaml.is_file():
        return default

    try:
        raw = yaml.safe_load(org_yaml.read_text()) or {}
    except yaml.YAMLError:
        return default
    if not isinstance(raw, dict):
        return default

    pkg_block = raw.get("package")
    if not isinstance(pkg_block, dict):
        # capabilities/dependencies may still be declared without a package:
        # block — surface them on the default manifest.
        default.capabilities = _as_dict(raw.get("capabilities"))
        default.dependencies = _as_dict(raw.get("dependencies"))
        return default

    inc = pkg_block.get("include")
    exc = pkg_block.get("exclude")
    include = _as_str_list(inc) if inc is not None else list(DEFAULT_INCLUDE)
    exclude = _as_str_list(exc)

    return Manifest(
        manifest_version=str(pkg_block.get("manifest_version", MANIFEST_VERSION)),
        package=PackageSpec(
            name=str(pkg_block.get("name", name)),
            version=str(pkg_block.get("version", "0.0.0")),
            include=include,
            exclude=exclude,
        ),
        capabilities=_as_dict(raw.get("capabilities")),
        dependencies=_as_dict(raw.get("dependencies")),
        source="declared",
    )


def effective_excludes(manifest: Manifest) -> list[str]:
    """The full exclude list applied at collection time: the org-declared
    excludes PLUS :data:`HARD_EXCLUDE` (hard excludes always win)."""
    return [*manifest.package.exclude, *HARD_EXCLUDE]


def collect_pack_files(org_dir: Path, manifest: Manifest) -> dict[str, Path]:
    """Default-DENY collection of the org's own files → ``{archive_rel: host}``.

    Walks ``org_dir`` recursively. A file ships IFF its org-relative posix path
    matches an include glob AND does NOT match any exclude (hard excludes
    always win). Unmatched files are left behind. Unreadable files (perm/IO)
    are skipped, never fatal. Keys are ``orgs/<name>/<rel>`` — the flattened
    archive layout consumers see (mirrors the legacy collector's output shape).

    Secret-bearing dirs (``data/``, ``.pux/``) and ``__pycache__/`` are PRUNED
    during the walk (:func:`os.walk` ``dirs[:]`` mutation) — we never descend
    into them, so a live secret under ``data/`` is never even opened. This is
    stricter than match-and-skip: the credential contract holds even if a future
    include glob widened to ``**``.
    """
    files: dict[str, Path] = {}
    if not org_dir.is_dir():
        return files

    org_name = manifest.package.name or org_dir.name
    includes = manifest.package.include or list(DEFAULT_INCLUDE)
    excludes = effective_excludes(manifest)

    for dirpath, dirnames, filenames in os.walk(org_dir):
        # Prune secret/transient subtrees IN PLACE so os.walk never descends.
        # Relative-to-org posix path of the current dir ("" at the org root).
        rel_dir = Path(dirpath).relative_to(org_dir).as_posix()
        if rel_dir == ".":
            rel_dir = ""
        dirnames[:] = [
            d for d in dirnames
            if not _is_pruned(f"{rel_dir}/{d}" if rel_dir else d, excludes)
        ]

        for fname in filenames:
            rel = f"{rel_dir}/{fname}" if rel_dir else fname
            if not _match_any(rel, includes):
                continue
            if _match_any(rel, excludes):
                continue
            path = Path(dirpath) / fname
            try:
                path.read_bytes()   # verify readability (skip on perm/IO failure)
            except (PermissionError, OSError):
                continue
            files[f"orgs/{org_name}/{rel}"] = path

    return files


def _is_pruned(rel_dir: str, excludes: list[str]) -> bool:
    """True if a directory (org-relative posix) should NOT be descended into.

    Recognizes the directory-shaped exclude patterns :data:`HARD_EXCLUDE` uses:
      * ``data/**``        → prune ``data`` and anything under ``data/``
      * ``**/__pycache__/**`` → prune any dir whose basename is ``__pycache__``
      * bare ``data``      → prune a top-level dir of that exact name
    File-only excludes (``**/*.pyc``) never prune a dir (they filter files)."""
    base = rel_dir.rsplit("/", 1)[-1]
    for pat in excludes:
        if pat.startswith("**/") and pat.endswith("/**"):
            # **/<name>/**  → prune any dir whose basename == <name>
            if base == pat[3:-3]:
                return True
        elif pat.endswith("/**"):
            stem = pat[:-3]   # "data/**" -> "data"
            if rel_dir == stem or rel_dir.startswith(stem + "/"):
                return True
        elif "/" not in pat and "*" not in pat:
            # A bare top-level name like "data" — prune an exact top-level dir.
            if rel_dir == pat:
                return True
    return False


def manifest_metadata(manifest: Manifest) -> dict[str, Any]:
    """The manifest block written into the archive's ``manifest.json`` — the
    receiver's audit surface (what ships, what the org can do, what it needs),
    NOT secrets. Identity + declared include/exclude + capabilities +
    dependencies."""
    return {
        "manifest_version": manifest.manifest_version,
        "source": manifest.source,
        "package": {
            "name": manifest.package.name,
            "version": manifest.package.version,
            "include": list(manifest.package.include),
            "exclude": effective_excludes(manifest),
        },
        "capabilities": dict(manifest.capabilities),
        "dependencies": dict(manifest.dependencies),
    }


# --- lenient coercion helpers (a bad manifest block never kills a pack) ------


def _as_dict(val: Any) -> dict[str, Any]:
    return dict(val) if isinstance(val, dict) else {}


def _as_str_list(val: Any) -> list[str]:
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    return []
