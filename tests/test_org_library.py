"""Phase 7 — the cross-project org library (``pux:`` namespace + PUX_ORG_PATHS).

Two mechanisms let a consumer app REUSE shipped library org bases (or another
project's orgs) without vendoring:

* ``$PUX_ORG_PATHS`` — colon-separated extra ``orgs/``-shaped roots. Org
  resolution + ``discover_orgs`` search the project's ``orgs/`` first, then
  each entry.
* the ``pux:`` namespace — ``pux:<base>`` resolves ONLY against the shipped
  library bases (``pux_harness/kit/bases/<base>/``). Used in ``extends:`` (org
  inheritance) + roster / agent-extends (a library agent).

The kit layer is ``project_root``-parameterized, so these tests build a tmp
project root + (for PUX_ORG_PATHS) monkeypatch the env — no harness shim. The
contract rule ``pux-namespace-resolvable`` is proven at the orchestrator layer.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from pux_harness.kit import _paths
from pux_harness.kit.loaders import (
    _load_agent_spec,
    build_system_prompt,
    discover_orgs,
    org_agent_slugs,
    org_extends,
)
from pux_harness.kit.loaders import _org_path  # the ONE resolver delegate


# --- the shipped copilot-kit base ------------------------------------------


def test_library_bases_dir_ships_copilot_kit():
    """``kit/bases/copilot-kit/`` ships in the package — AGENTS.md + org.yaml +
    a roster agent — so any consumer can ``extends: pux:copilot-kit``."""
    base = _paths.library_bases_dir() / "copilot-kit"
    assert base.is_dir(), "kit/bases/copilot-kit/ missing from the package"
    assert (base / "AGENTS.md").is_file()
    assert (base / "org.yaml").is_file()
    assert (base / "agents" / "copilot-helper.md").is_file()


def test_library_base_ships_in_installed_package():
    """The base is reachable through the INSTALLED package (importlib.resources),
    not just the source tree — i.e. it ships in the wheel. A consumer app that
    pip-installs pux-harness finds it without a local checkout."""
    import importlib.resources

    pkg_root = Path(importlib.resources.files("pux_harness.kit"))
    base = pkg_root / "bases" / "copilot-kit" / "AGENTS.md"
    assert base.is_file(), f"copilot-kit base not packaged: {base}"


# --- search_org_dir: the namespace + multi-root resolver -------------------


def test_search_org_dir_pux_resolves_library_base(tmp_path: Path):
    """``pux:copilot-kit`` resolves to the shipped library base dir."""
    assert _org_path("pux:copilot-kit", tmp_path) == _paths.library_bases_dir() / "copilot-kit"


def test_search_org_dir_pux_dangling_raises(tmp_path: Path):
    """An unknown ``pux:`` base raises FileNotFoundError (no silent fallback)."""
    with pytest.raises(FileNotFoundError, match="pux: base"):
        _org_path("pux:totally-made-up", tmp_path)


def test_search_org_dir_local_wins_over_pux_namespace(tmp_path: Path):
    """A LOCAL org wins over the library base on a bare-name collision — but
    ``pux:`` is the escape-hatch that ALWAYS means the library base."""
    # Local org namesaked "copilot-kit" (a consumer override).
    local = tmp_path / "orgs" / "copilot-kit"
    local.mkdir(parents=True)
    (local / "AGENTS.md").write_text("# local override\n")
    # Bare name -> local; pux: -> library.
    assert _org_path("copilot-kit", tmp_path) == local
    assert _org_path("pux:copilot-kit", tmp_path) == _paths.library_bases_dir() / "copilot-kit"


# --- $PUX_ORG_PATHS ---------------------------------------------------------


def test_extra_org_roots_parses_env_drops_missing(monkeypatch, tmp_path: Path):
    """``$PUX_ORG_PATHS`` is colon-separated; non-existent entries are dropped;
    order preserved."""
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    monkeypatch.setenv("PUX_ORG_PATHS", f"{a}{os.pathsep}/no/such/dir{os.pathsep}{b}")
    assert _paths.extra_org_roots() == [a, b]


def test_search_org_dir_falls_through_to_pux_org_paths(monkeypatch, tmp_path: Path):
    """A bare name not in the project's ``orgs/`` resolves from a
    ``$PUX_ORG_PATHS`` root (top-level, then its ``specialists/``)."""
    extra = tmp_path / "lib"
    (extra / "shared-org").mkdir(parents=True)
    (extra / "shared-org" / "AGENTS.md").write_text("# shared\n")
    monkeypatch.setattr(_paths, "extra_org_roots", lambda: [extra])
    assert _org_path("shared-org", tmp_path / "proj") == extra / "shared-org"


def test_discover_orgs_includes_pux_org_paths(monkeypatch, tmp_path: Path):
    """``discover_orgs`` lists the project's orgs AND ``$PUX_ORG_PATHS`` orgs
    (but NOT library bases — those are opt-in via ``pux:``)."""
    proj = tmp_path / "proj"
    (proj / "orgs" / "local").mkdir(parents=True)
    (proj / "orgs" / "local" / "AGENTS.md").write_text("# local\n")
    extra = tmp_path / "lib"
    (extra / "external").mkdir(parents=True)
    (extra / "external" / "AGENTS.md").write_text("# external\n")
    monkeypatch.setattr(_paths, "extra_org_roots", lambda: [extra])
    discovered = discover_orgs(proj)
    assert "local" in discovered and "external" in discovered


# --- org inheritance from a library base (the headline feature) -------------


def _make_consumer_org(root: Path, name: str, *, extends: str, body: str = "# consumer\n"):
    d = root / "orgs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "AGENTS.md").write_text(body)
    (d / "org.yaml").write_text(f"extends: {extends}\n")
    return d


def test_consumer_extends_library_base_inherits_roster(tmp_path: Path):
    """A consumer org ``extends: pux:copilot-kit`` inherits the base's roster
    (``copilot-helper``) — Phase 5 inheritance working across the library
    boundary. The base's agent resolves through the base's own ``agents/`` dir."""
    _make_consumer_org(tmp_path, "my-app", extends="pux:copilot-kit")
    assert "copilot-helper" in org_agent_slugs("my-app", tmp_path)


def test_consumer_extends_library_base_inherits_prompt(tmp_path: Path):
    """The base's AGENTS.md overlay lands in the consumer's assembled prompt
    (parent + own concatenated own-last)."""
    _make_consumer_org(tmp_path, "my-app", extends="pux:copilot-kit",
                       body="# my-app\n\nMy own CTO prose.\n")
    prompt = build_system_prompt("my-app", project_root=tmp_path)
    assert "co-pilot kit" in prompt.lower()  # inherited from the base
    assert "My own CTO prose." in prompt      # own overlay (own-last)


def test_consumer_specializes_inherited_library_agent(tmp_path: Path):
    """A consumer drops a same-named ``copilot-helper.md`` in its own
    ``agents/`` dir to SPECIALIZE the inherited library agent (own wins,
    child-local-first resolution)."""
    _make_consumer_org(tmp_path, "my-app", extends="pux:copilot-kit")
    adir = tmp_path / "orgs" / "my-app" / "agents"
    adir.mkdir(parents=True)
    (adir / "copilot-helper.md").write_text(
        "---\nname: copilot-helper\ndescription: specialized\n---\n"
        "I am the CONSUMER's specialized helper.\n"
    )
    spec = _load_agent_spec("copilot-helper", "my-app", tmp_path)
    assert spec is not None
    assert "CONSUMER's specialized" in spec["system_prompt"]


def test_consumer_extends_chain_walks_through_library_base(tmp_path: Path):
    """The inheritance chain walks through the ``pux:`` base: a consumer that
    extends ``pux:copilot-kit`` has the base at the chain ROOT, and a child
    ``extends:`` from the consumer still works (multi-level across the
    boundary)."""
    from pux_harness.kit.loaders import org_extends_chain
    _make_consumer_org(tmp_path, "my-app", extends="pux:copilot-kit")
    chain = org_extends_chain("my-app", tmp_path)  # root→child
    assert chain[0] == "pux:copilot-kit"
    assert chain[-1] == "my-app"


# --- pux: agent slugs (roster + agent-extends) ------------------------------


def test_pux_agent_slug_resolves_library_agent(tmp_path: Path):
    """A roster entry ``pux:copilot-helper`` pulls the library agent directly
    (searches the library bases' ``agents/`` dirs, stripped of the prefix)."""
    _make_consumer_org(tmp_path, "my-app", extends="pux:copilot-kit")
    # Override the roster to reference the library agent by pux: slug.
    (tmp_path / "orgs" / "my-app" / "org.yaml").write_text(
        "agents: [pux:copilot-helper]\n"
    )
    spec = _load_agent_spec("pux:copilot-helper", "my-app", tmp_path)
    assert spec is not None
    assert spec["name"] == "copilot-helper"


def test_pux_agent_slug_dangling_returns_none(tmp_path: Path):
    """An unknown ``pux:`` agent slug resolves to None (the loader's contract);
    the harness contract surfaces it as ``pux-namespace-resolvable``."""
    _make_consumer_org(tmp_path, "my-app", extends="pux:copilot-kit")
    assert _load_agent_spec("pux:no-such-agent", "my-app", tmp_path) is None
    assert _paths.resolve_library_agent("pux:no-such-agent") is None


def test_resolve_library_agent_accepts_bare_or_namespaced():
    """``resolve_library_agent`` is idempotent on the prefix — namespaced or
    bare both find the shipped agent."""
    assert _paths.resolve_library_agent("pux:copilot-helper") is not None
    assert _paths.resolve_library_agent("copilot-helper") is not None
