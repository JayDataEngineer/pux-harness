"""Stage 2 (import hygiene) — the slim kit core's import boundary is a PERMANENT
contract, not a passing today-property.

Two complementary lock-ins (both ``verify-or-die``):

* **Runtime (lives in ``test_kit_compile.py::test_import_isolation_no_docker_no_heavy_subsystem``):**
  a fresh subprocess does ``from pux_harness import compile_org`` and asserts
  neither ``docker`` nor any heavy ``pux_harness`` subsystem
  (``sandbox``/``browser``/``context``) lands in ``sys.modules``. Proves the
  import GRAPH is clean today.
* **Source (here):** the AST tripwire ``kit-import-isolation`` fires the moment
  a heavy import (or a sibling-``pux_harness``-subsystem import) is added to
  the kit source — eager OR lazy (``ast.walk`` descends into function bodies).
  Proves the SOURCE can't silently re-couple the slim core to the heavy
  run-time, which is the precondition for Stage 3 splitting the heavy deps into
  optional extras.

The tripwire resolves relative imports to absolute first, so a within-kit
``from .loaders import ...`` (resolves to ``pux_harness.kit.loaders``) is NOT a
false positive.

The tripwire lives in ``tests/harness/tripwire_checks.py`` — the TEST half of
the former ``pux_harness/agent/contract.py`` (a permanent, repo-wide gate that
is optional and never deployed). Being imported by this suite IS its
registration: there is no runtime global gate anymore.
"""
from __future__ import annotations

from tests.harness.tripwire_checks import (
    _HEAVY_MODULE_ROOTS,
    _kit_import_isolation,
    _scan_for_heavy_imports,
)

KIT_PKG = ["pux_harness", "kit"]


# --- green on the real repo ------------------------------------------------


def test_tripwire_green_on_real_kit():
    """The real ``pux_harness/kit/**`` + top-level ``__init__.py`` import nothing
    heavy — the tripwire is clean on the shipped source."""
    assert _kit_import_isolation() == [], _kit_import_isolation()


# --- provocation: heavy module import trips --------------------------------


def test_trips_on_heavy_absolute_import(tmp_path):
    """``import docker`` in a kit file is a hard failure."""
    src = tmp_path / "compile.py"
    src.write_text("import docker\n")
    vs = _scan_for_heavy_imports(src, KIT_PKG)
    assert len(vs) == 1, vs
    assert vs[0].rule == "kit-import-isolation"
    assert vs[0].severity == "error"
    assert "docker" in vs[0].message


def test_trips_on_heavy_from_import(tmp_path):
    """``from fastapi import FastAPI`` is just as much a leak."""
    src = tmp_path / "server.py"
    src.write_text("from fastapi import FastAPI\n")
    vs = _scan_for_heavy_imports(src, KIT_PKG)
    assert len(vs) == 1, vs
    assert vs[0].rule == "kit-import-isolation"
    assert "fastapi" in vs[0].message


def test_trips_on_lazy_import_inside_function(tmp_path):
    """A deferred ``import docker`` inside a function body still trips — the kit
    must not reference docker AT ALL, not even lazily."""
    src = tmp_path / "compile.py"
    src.write_text(
        "def f():\n"
        "    import docker  # lazily imported, but still a kit->heavy coupling\n"
        "    return docker\n"
    )
    vs = _scan_for_heavy_imports(src, KIT_PKG)
    assert any(v.rule == "kit-import-isolation" and "docker" in v.message for v in vs), vs


def test_trips_on_every_declared_heavy_root(tmp_path):
    """Every root in ``_HEAVY_MODULE_ROOTS`` is rejected — none can sneak in."""
    for root in _HEAVY_MODULE_ROOTS:
        src = tmp_path / f"{root}_probe.py"
        src.write_text(f"import {root}\n")
        vs = _scan_for_heavy_imports(src, KIT_PKG)
        assert any(root in v.message for v in vs), f"{root} not flagged: {vs}"


# --- provocation: sibling subsystem reach trips ----------------------------


def test_trips_on_sibling_subsystem_absolute(tmp_path):
    """``from pux_harness.sandbox import X`` couples the kit to the heavy layer."""
    src = tmp_path / "compile.py"
    src.write_text("from pux_harness.sandbox import classify_slug\n")
    vs = _scan_for_heavy_imports(src, KIT_PKG)
    assert len(vs) == 1, vs
    assert "pux_harness.sandbox" in vs[0].message


def test_trips_on_top_level_init_importing_subsystem(tmp_path):
    """The top-level ``pux_harness/__init__.py`` may import ONLY ``pux_harness.kit``."""
    src = tmp_path / "__init__.py"
    src.write_text("from pux_harness.agent import orgs\n")
    vs = _scan_for_heavy_imports(src, ["pux_harness"])
    assert len(vs) == 1, vs
    assert "pux_harness.agent" in vs[0].message


# --- no false positives: the allowed surface stays quiet --------------------


def test_clean_on_real_kit_imports(tmp_path):
    """A source mirroring the REAL kit imports (stdlib + yaml + deepagents +
    langchain_core + langgraph + within-kit relatives) emits nothing."""
    src = tmp_path / "compile.py"
    src.write_text(
        "from __future__ import annotations\n"
        "import os\n"
        "from pathlib import Path\n"
        "from typing import Any\n"
        "import yaml\n"
        "from deepagents import create_deep_agent\n"
        "from deepagents.backends.filesystem import FilesystemBackend\n"
        "from langchain_core.tools import BaseTool\n"
        "from langgraph.graph.state import CompiledStateGraph\n"
        "from .loaders import _load_agent_spec, build_system_prompt\n"
    )
    assert _scan_for_heavy_imports(src, KIT_PKG) == []


def test_clean_on_within_kit_relative_from_init(tmp_path):
    """``kit/__init__.py`` doing ``from .compile import compile_org`` resolves to
    ``pux_harness.kit.compile`` (second part == 'kit') — NOT a subsystem leak."""
    src = tmp_path / "__init__.py"
    src.write_text(
        "from .compile import compile_org\n"
        "from .loaders import discover_orgs\n"
    )
    assert _scan_for_heavy_imports(src, KIT_PKG) == []


def test_clean_on_top_level_init_reexporting_kit(tmp_path):
    """The actual top-level re-export ``from pux_harness.kit import compile_org``
    is the ONE allowed cross-package import — it stays quiet."""
    src = tmp_path / "__init__.py"
    src.write_text(
        "from __future__ import annotations\n"
        "from pux_harness.kit import compile_org\n"
        "__all__ = ['compile_org']\n"
    )
    assert _scan_for_heavy_imports(src, ["pux_harness"]) == []
