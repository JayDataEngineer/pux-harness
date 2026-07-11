"""Unit tests for ``pux_harness.manifest`` — the declarative default-deny pack
manifest (P3).

The manifest replaces the old hardcoded ``export._collect_org_files`` allowlist:
what an org ships is DECLARED via ``package.include`` globs in ``org.yaml``, not
implicit. ``data/``/``.pux/`` are PERMANENT excludes (the credential-leak
contract). This module is pure data (no tarball/exec/Docker), so every behavior
is exercised directly against synthetic orgs in ``tmp_path``.

The decisive test is :func:`test_pack_contents_equal_declared_manifest` — the P3
prove criterion: the files a pack collects are EXACTLY the manifest's declared
includes minus the excludes. Nothing implicit, nothing missing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pux_harness.manifest import (
    DEFAULT_INCLUDE,
    HARD_EXCLUDE,
    _is_pruned,
    _match_any,
    collect_pack_files,
    effective_excludes,
    load_manifest,
    manifest_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_org(org_dir: Path, files: dict[str, str]) -> Path:
    """Lay out a synthetic org: ``files`` is ``{relpath: text}`` under org_dir."""
    org_dir.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = org_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return org_dir


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

class TestLoadManifestDefault:
    """No ``package:`` block → the DEFAULT manifest (DEFAULT_INCLUDE globs)."""

    def test_no_org_yaml_returns_default_include(self, tmp_path):
        org_dir = tmp_path / "acme"
        org_dir.mkdir()
        m = load_manifest(org_dir)
        assert m.source == "default"
        assert m.package.include == list(DEFAULT_INCLUDE)
        assert m.package.name == "acme"   # seeded from the dir name

    def test_org_yaml_without_package_block_still_default(self, tmp_path):
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": "agents: [worker]\n",
        })
        m = load_manifest(org_dir)
        assert m.source == "default"
        assert m.package.include == list(DEFAULT_INCLUDE)

    def test_capabilities_surface_without_package_block(self, tmp_path):
        """An org may declare capabilities/dependencies WITHOUT a package: block
        — they ride on the default manifest and surface into the audit output."""
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": (
                "agents: [worker]\n"
                "capabilities:\n  tools: [web_research]\n"
                "dependencies:\n  pip: [requests]\n"
            ),
        })
        m = load_manifest(org_dir)
        assert m.source == "default"
        assert m.capabilities == {"tools": ["web_research"]}
        assert m.dependencies == {"pip": ["requests"]}


class TestLoadManifestDeclared:
    """A ``package:`` block drives a default-deny collection against the org's
    OWN declaration."""

    def test_declared_include_replaces_default(self, tmp_path):
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": (
                "package:\n"
                "  name: acme-pro\n"
                "  version: 1.2.0\n"
                "  include:\n"
                "    - org.yaml\n"
                "    - agents/**\n"
            ),
        })
        m = load_manifest(org_dir)
        assert m.source == "declared"
        assert m.package.name == "acme-pro"
        assert m.package.version == "1.2.0"
        assert m.package.include == ["org.yaml", "agents/**"]
        # DEFAULT_INCLUDE is NOT merged in — declaration is authoritative.
        assert "skills/**" not in m.package.include

    def test_declared_exclude_added(self, tmp_path):
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": (
                "package:\n"
                "  include: [\"**\"]\n"
                "  exclude: [\"secrets/**\"]\n"
            ),
        })
        m = load_manifest(org_dir)
        assert m.package.exclude == ["secrets/**"]
        assert "secrets/**" in effective_excludes(m)

    def test_lenient_bad_yaml_falls_back_to_default(self, tmp_path):
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": "package: [this is\n  not: valid: yaml\n",
        })
        m = load_manifest(org_dir)
        assert m.source == "default"
        assert m.package.include == list(DEFAULT_INCLUDE)

    def test_lenient_package_not_a_dict_falls_back(self, tmp_path):
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": "package: just-a-string\n",
        })
        m = load_manifest(org_dir)
        assert m.source == "default"


# ---------------------------------------------------------------------------
# effective_excludes + HARD_EXCLUDE
# ---------------------------------------------------------------------------

class TestEffectiveExcludes:
    """Hard excludes ALWAYS win, appended after org-declared excludes."""

    def test_hard_excludes_always_merged(self, tmp_path):
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": "package:\n  include: ['**']\n  exclude: ['tmp/**']\n",
        })
        m = load_manifest(org_dir)
        exc = effective_excludes(m)
        # Org-declared first ...
        assert "tmp/**" in exc
        # ... then EVERY hard exclude (always present, regardless of declaration).
        for hard in HARD_EXCLUDE:
            assert hard in exc

    def test_hard_excludes_present_even_with_no_declaration(self, tmp_path):
        m = load_manifest(tmp_path / "acme")   # default manifest
        exc = effective_excludes(m)
        for hard in HARD_EXCLUDE:
            assert hard in exc


# ---------------------------------------------------------------------------
# collect_pack_files — the default-deny collection
# ---------------------------------------------------------------------------

class TestCollectPackFilesDefaultDeny:
    """A file ships IFF it matches an include AND doesn't match an exclude."""

    def test_default_manifest_collects_core_files(self, tmp_path):
        org_dir = _write_org(tmp_path / "acme", {
            "AGENTS.md": "# acme\n",
            "org.yaml": "agents: [worker]\n",
            "agents/worker.md": "body\n",
            "skills/mining/SKILL.md": "# skill\n",
        })
        m = load_manifest(org_dir)
        files = collect_pack_files(org_dir, m)
        keys = set(files)
        assert any(k.endswith("AGENTS.md") for k in keys)
        assert any(k.endswith("org.yaml") for k in keys)
        assert any("agents/worker.md" in k for k in keys)
        assert any("skills/mining/SKILL.md" in k for k in keys)
        # Keys are the flattened archive layout: orgs/<name>/<rel>.
        assert all(k.startswith("orgs/acme/") for k in keys)

    def test_unmatched_file_left_behind(self, tmp_path):
        """The crux of default-deny: a file NOT under any include glob is dropped,
        even though it exists in the tree."""
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": "agents: [worker]\n",
            "agents/worker.md": "body\n",
            "scratchpad.notes": "private musings\n",   # not in DEFAULT_INCLUDE
        })
        m = load_manifest(org_dir)
        files = collect_pack_files(org_dir, m)
        assert not any("scratchpad.notes" in k for k in files), (
            "unmatched file shipped — default-deny broken"
        )

    def test_declared_glob_narrows_collection(self, tmp_path):
        """A declared manifest ships ONLY its include globs — skills/ drop out
        when the org declares include: [org.yaml, agents/**]."""
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": (
                "package:\n  include:\n    - org.yaml\n    - agents/**\n"
            ),
            "agents/worker.md": "body\n",
            "skills/mining/SKILL.md": "# skill\n",   # NOT declared → left behind
        })
        m = load_manifest(org_dir)
        files = collect_pack_files(org_dir, m)
        assert any("agents/worker.md" in k for k in files)
        assert not any("skills/" in k for k in files), (
            "undeclared skills/ shipped — declaration not authoritative"
        )

    def test_lib_directory_travels_in_pack(self, tmp_path):
        """Level (c) dynamic tools (``lib/**``) ship in the pack (P1/P2 contract
        — an exported org keeps its learned functions)."""
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": "agents: [worker]\n",
            "lib/functions/analyze.py": "def run(**k): ...\n",
            "lib/index.yaml": "functions: {}\n",
        })
        m = load_manifest(org_dir)
        files = collect_pack_files(org_dir, m)
        assert any("lib/functions/analyze.py" in k for k in files)
        assert any("lib/index.yaml" in k for k in files)


# ---------------------------------------------------------------------------
# HARD_EXCLUDE dir-pruning — the credential-leak contract
# ---------------------------------------------------------------------------

class TestHardExcludePrune:
    """``data/`` / ``.pux/`` / ``__pycache__/`` are PRUNED during the walk — the
    walker never descends into them, so a live secret is never even opened. This
    holds even if an include glob widened to ``**``."""

    @pytest.mark.parametrize("secret_dir,secret_file", [
        ("data", ".session.json"),
        (".pux", "state.db"),
    ])
    def test_secret_dirs_pruned_under_wildcard_include(
        self, tmp_path, secret_dir, secret_file
    ):
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": "package:\n  include: ['**']\n",   # widest possible
            "AGENTS.md": "# acme\n",
            f"{secret_dir}/{secret_file}": '{"cookies":"LIVE-SECRET"}',
        })
        m = load_manifest(org_dir)
        files = collect_pack_files(org_dir, m)
        leaked = [k for k in files if f"/{secret_dir}/" in k or k.endswith(f"/{secret_dir}")]
        assert not leaked, f"{secret_dir}/ leaked under '**' include: {leaked}"
        # The legitimate top-level file still ships (proves '**' is active).
        assert any(k.endswith("AGENTS.md") for k in files)

    def test_pycache_and_pyc_excluded(self, tmp_path):
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": "agents: [worker]\n",
            "agents/worker.md": "body\n",
            "agents/__pycache__/worker.cpython-311.pyc": "bytes",
            "lib/functions/__pycache__/analyze.pyc": "bytes",
        })
        m = load_manifest(org_dir)
        files = collect_pack_files(org_dir, m)
        assert not any("__pycache__" in k for k in files)
        assert not any(k.endswith(".pyc") for k in files)

    def test_org_declared_exclude_also_prunes(self, tmp_path):
        """An org can declare its OWN exclude dir; it prunes the walk too (not
        just filters files — a dir exclude stops descent)."""
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": (
                "package:\n"
                "  include: ['**']\n"
                "  exclude: ['drafts/**']\n"
            ),
            "AGENTS.md": "# acme\n",
            "drafts/notes.md": "rough\n",
            "drafts/sub/more.md": "rougher\n",
        })
        m = load_manifest(org_dir)
        files = collect_pack_files(org_dir, m)
        assert not any("/drafts/" in k for k in files)


# ---------------------------------------------------------------------------
# _is_pruned + _match_any internals
# ---------------------------------------------------------------------------

class TestPruneInternals:
    def test_is_pruned_basename_pattern(self):
        # **/__pycache__/** → any dir whose basename is __pycache__
        assert _is_pruned("foo/__pycache__", list(HARD_EXCLUDE))
        assert _is_pruned("__pycache__", list(HARD_EXCLUDE))

    def test_is_pruned_prefix_pattern(self):
        # data/** → "data" itself + anything under data/
        assert _is_pruned("data", list(HARD_EXCLUDE))
        assert _is_pruned("data/sub", list(HARD_EXCLUDE))
        assert not _is_pruned("database", list(HARD_EXCLUDE))   # not a prefix

    def test_is_pruned_file_only_pattern_does_not_prune_dir(self):
        # **/*.pyc is a file filter — it must NOT prune a directory.
        assert not _is_pruned("anything", ["**/*.pyc"])

    def test_match_any_anchored_bare_name(self):
        # A bare "AGENTS.md" matches only an org-root file of that exact name.
        assert _match_any("AGENTS.md", ["AGENTS.md"])
        assert not _match_any("agents/AGENTS.md", ["AGENTS.md"])

    def test_match_any_glob_spans_separator(self):
        # fnmatch '*' spans '/', so agents/** matches at any depth.
        assert _match_any("agents/a.md", ["agents/**"])
        assert _match_any("agents/sub/a.md", ["agents/**"])


# ---------------------------------------------------------------------------
# manifest_metadata — the archive audit surface
# ---------------------------------------------------------------------------

class TestManifestMetadata:
    def test_default_metadata_shape(self, tmp_path):
        m = load_manifest(tmp_path / "acme")
        md = manifest_metadata(m)
        assert md["source"] == "default"
        assert md["package"]["name"] == "acme"
        # Effective excludes (hard + declared) are surfaced, not just declared.
        for hard in HARD_EXCLUDE:
            assert hard in md["package"]["exclude"]

    def test_declared_metadata_carries_capabilities(self, tmp_path):
        org_dir = _write_org(tmp_path / "acme", {
            "org.yaml": (
                "package:\n  name: acme-pro\n  version: 9.9.9\n"
                "capabilities:\n  mcp: [web_research]\n"
                "dependencies:\n  pip: [httpx]\n"
            ),
        })
        m = load_manifest(org_dir)
        md = manifest_metadata(m)
        assert md["package"]["name"] == "acme-pro"
        assert md["package"]["version"] == "9.9.9"
        assert md["capabilities"] == {"mcp": ["web_research"]}
        assert md["dependencies"] == {"pip": ["httpx"]}


# ---------------------------------------------------------------------------
# THE P3 PROVE CRITERION — pack contents == manifest declaration
# ---------------------------------------------------------------------------

def test_pack_contents_equal_declared_manifest(tmp_path):
    """Contract: the files ``collect_pack_files`` returns are EXACTLY the
    manifest's declared includes, minus every exclude. Nothing implicit ships;
    nothing declared is silently dropped. This is the P3 replacement for the old
    hardcoded-allowlist test — except now the expectation is DERIVED from the
    declaration, not a Python tuple."""
    org_dir = _write_org(tmp_path / "acme", {
        "org.yaml": (
            "package:\n"
            "  include:\n"
            "    - org.yaml\n"
            "    - AGENTS.md\n"
            "    - agents/**\n"
            "  exclude:\n"
            "    - agents/_draft/**\n"
        ),
        "AGENTS.md": "# acme\n",
        "agents/keeper.md": "body\n",
        "agents/_draft/wip.md": "draft\n",
        "skills/ignored/SKILL.md": "# not declared\n",   # not in include
        "data/.secret.json": '{"token":"LIVE"}',          # hard-excluded
    })
    m = load_manifest(org_dir)
    files = collect_pack_files(org_dir, m)
    shipped_rels = sorted(k.removeprefix("orgs/acme/") for k in files)

    # EXACTLY these (declared includes, minus excludes), nothing else.
    assert shipped_rels == [
        "AGENTS.md",
        "agents/keeper.md",
        "org.yaml",
    ], f"pack contents diverge from manifest declaration: {shipped_rels}"
