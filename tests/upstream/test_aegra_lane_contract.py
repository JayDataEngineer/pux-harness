"""Static contract test for the Aegra prod-runtime manifest (``aegra.json``).

Aegra ([[aegra-verified]]) is pux's FREE prod AP runtime — an OSS self-hosted
langgraph-api/LangGraph-Platform drop-in (FastAPI + PostgreSQL, Apache-2.0,
SAME ``langgraph_sdk`` ``threads/runs/stream`` wire format), resolving
[[langgraph-api-license-gate]]. It auto-discovers ``aegra.json`` with PRIORITY
over ``langgraph.json`` (aegra_api/config.py).

This test pins the GENERATOR OUTPUT contract — the part that is NOT a live
server (a pytest-booted Aegra needs PostgreSQL, so the live end-to-end proof
lives in ``scripts/aegra_smoke.py`` instead). It guards:

* ``aegra.json`` on disk == ``gen_aegra_json.render()`` (CI --check equivalent);
* every graph spec is a FILE-PATH form (``./pux_harness/runtime/upstream.py:X``)
  — Aegra's graph loader (``_load_graph_from_file``) has NO module-import branch,
  so module-path specs (langgraph.json's form) would fail with "Graph file not
  found". This divergence is INTENTIONAL and the test makes it a contract;
* the target file (``upstream.py``) exists and exposes every ``graph__<slug>``
  attr (so Aegra's ``spec_from_file_location`` load will resolve them);
* ``http.app`` (the custom-app seam — EventBus + jobs under Aegra) +
  ``python_version`` mirror langgraph.json;
* the graph_id set matches the discovered orgs/ tree (one source of truth).

See [[upstream-protocol-pivot]], [[plan-p3-server-rest-retirement]].
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# pux-harness/ (this file: tests/upstream/test_aegra_lane_contract.py)
HARNESS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = HARNESS_ROOT.parent  # .../auto-developer-orchestrator (owns orgs/)
AEGRA_MANIFEST = HARNESS_ROOT / "aegra.json"
LANGGRAPH_MANIFEST = HARNESS_ROOT / "langgraph.json"
UPSTREAM_PY = HARNESS_ROOT / "pux_harness" / "runtime" / "upstream.py"


@pytest.fixture(scope="module")
def gen_module():
    """Import the generator as a module (it lives in scripts/, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "gen_aegra_json", HARNESS_ROOT / "scripts" / "gen_aegra_json.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_aegra_json"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _set_pux_project_root(monkeypatch):
    monkeypatch.setenv("PUX_PROJECT_ROOT", str(REPO_ROOT))


def test_aegra_json_matches_generator(gen_module, monkeypatch):
    """The committed aegra.json must equal what the generator emits now (CI guard)."""
    _set_pux_project_root(monkeypatch)
    assert AEGRA_MANIFEST.is_file(), "aegra.json missing — run gen_aegra_json.py"
    on_disk = AEGRA_MANIFEST.read_text()
    assert on_disk == gen_module.render(), (
        "aegra.json is stale — run: uv run python scripts/gen_aegra_json.py"
    )


def test_graph_specs_are_file_paths_not_module_paths(gen_module, monkeypatch):
    """Aegra's loader is file-path-only; every spec must point at upstream.py."""
    _set_pux_project_root(monkeypatch)
    specs = gen_module.graph_specs()
    assert specs, "no graph specs emitted"
    for graph_id, spec in specs.items():
        # Aegra resolves file paths relative to the config dir (pux-harness/).
        assert spec.startswith("./"), (
            f"{graph_id}: spec {spec!r} is not a file-path form "
            "(Aegra has no module-import branch) — would fail to load"
        )
        file_part, _, export = spec.partition(":")
        assert file_part.endswith("upstream.py"), f"{graph_id}: not pointing at upstream.py"
        assert export.startswith("graph__"), f"{graph_id}: export {export!r} not a graph__ attr"


def test_upstream_py_exposes_every_graph_attr(gen_module, monkeypatch):
    """The file Aegra loads must actually expose each declared graph__<slug> attr."""
    _set_pux_project_root(monkeypatch)
    assert UPSTREAM_PY.is_file(), "upstream.py not found"
    specs = gen_module.graph_specs()
    # Load upstream.py as Aegra does (spec_from_file_location), then check attrs.
    loaded = importlib.util.spec_from_file_location("_aegra_contract_upstream", UPSTREAM_PY)
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)  # type: ignore[union-attr]
    for graph_id, spec in specs.items():
        export = spec.partition(":")[2]
        assert hasattr(module, export), (
            f"{graph_id}: upstream.py has no attr {export} "
            "(factory registration loop did not run on file-load)"
        )


def test_http_app_and_python_version_mirror_langgraph_json():
    """The custom-app seam + python pin must match langgraph.json (one source)."""
    aegra = json.loads(AEGRA_MANIFEST.read_text())
    lg = json.loads(LANGGRAPH_MANIFEST.read_text())
    assert aegra["http"] == lg["http"], "http.app seam diverged from langgraph.json"
    assert aegra.get("python_version") == lg.get("python_version") == "3.13"
    assert aegra["http"]["app"] == "pux_harness.runtime.custom_app:app"


def test_graph_id_set_matches_discovered_orgs(gen_module, monkeypatch):
    """aegra.json's graph_ids must equal the orgs/ tree (no drift, no stale)."""
    _set_pux_project_root(monkeypatch)
    on_disk = json.loads(AEGRA_MANIFEST.read_text())
    assert set(on_disk["graphs"]) == set(gen_module.graph_specs())


def test_aegra_spec_form_intentionally_differs_from_langgraph_json():
    """The file-path (Aegra) vs module-path (langgraph) divergence is a CONTRACT,
    not drift — pin it so an accidental 'fix' (making them identical) is caught."""
    aegra = json.loads(AEGRA_MANIFEST.read_text())
    lg = json.loads(LANGGRAPH_MANIFEST.read_text())
    assert set(aegra["graphs"]) == set(lg["graphs"]), "graph_id sets diverged"
    for gid in aegra["graphs"]:
        a_spec = aegra["graphs"][gid]
        l_spec = lg["graphs"][gid]
        # Same export attr, DIFFERENT address form (file vs module).
        assert a_spec.split(":")[-1] == l_spec.split(":")[-1], f"{gid}: export attr diverged"
        assert a_spec != l_spec, (
            f"{gid}: aegra spec collapsed to langgraph's module-path form — "
            "Aegra cannot load module paths; the file-path divergence is required"
        )
