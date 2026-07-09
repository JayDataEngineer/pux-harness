#!/usr/bin/env python3
"""Generate ``aegra.json`` — the Aegra (OSS langgraph-api drop-in) prod runtime
config — from the discovered orgs/ tree.

Aegra (github.com/aegra/aegra) is pux's FREE prod AP runtime (FastAPI +
PostgreSQL, Apache-2.0, SAME ``langgraph_sdk`` ``threads/runs/stream`` wire
format as langgraph-api, but keyless — resolves [[langgraph-api-license-gate]]).
It auto-discovers ``aegra.json`` with PRIORITY over ``langgraph.json``
(aegra_api/config.py::_resolve_config_path). pux ships BOTH manifests:
- ``langgraph.json`` (module-path graph specs) — ``langgraph dev``/``build`` lane.
- ``aegra.json``   (file-path   graph specs) — Aegra prod lane (this file).

WHY file-path specs here (the divergence from langgraph.json): Aegra's graph
loader (``aegra_api/services/langgraph_service.py::_load_graph_from_file``)
treats the graph spec as a FILE PATH only — ``Path(raw_path)`` + ``exists()`` +
``importlib.util.spec_from_file_location`` — it has NO module-import branch
(comment: "Parse path format: './graphs/weather_agent.py:graph'"), unlike
langgraph-api. So pux's module-path specs
(``pux_harness.runtime.upstream:graph__X``) must be rewritten as file paths
(``./pux_harness/runtime/upstream.py:graph__X``) for Aegra. Pointing at the
SAME ``upstream.py`` is sufficient: its top-level factory-registration loop
exposes every ``graph__<slug>`` attr on load (Aegra loads the file once per
graph_id; registration is idempotent). Downstream config accommodation only —
we do NOT fork Aegra ([[rely-on-upstream]]).

Everything else (the ``http.app`` custom-app seam, ``python_version``) mirrors
``langgraph.json`` verbatim so the two manifests share one source of truth
(the orgs/ tree) and differ ONLY in the graph-spec address form.

Usage:  uv run python scripts/gen_aegra_json.py [--check]
  --check: exit 1 if the committed aegra.json would change (CI guard).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # pux-harness/
MANIFEST = ROOT / "aegra.json"
# Aegra resolves file paths relative to the config dir (aegra.json's parent =
# pux-harness/), so this points at the same upstream.py the factories live in.
UPSTREAM_FILE = "./pux_harness/runtime/upstream.py"
HTTP_APP = "pux_harness.runtime.custom_app:app"


def _project_root() -> Path | None:
    """Where the orgs/ tree lives (the parent repo; pux-harness is its submodule)."""
    env = os.environ.get("PUX_PROJECT_ROOT")
    if env:
        return Path(env)
    candidate = ROOT.parent  # .../auto-developer-orchestrator
    return candidate if (candidate / "orgs").is_dir() else None


def graph_specs() -> dict[str, str]:
    """``{graph_id: "./pux_harness/runtime/upstream.py:graph__<slug>"}`` — Aegra
    FILE-PATH specs (NOT module paths) for every discovered org."""
    from pux_harness.runtime.upstream import graph_attr_name

    root = _project_root()
    if root is None:
        # Standalone kit (no orgs/ tree): ship `general` only.
        return {"general": f"{UPSTREAM_FILE}:{graph_attr_name('general')}"}
    from pux_harness.kit.loaders import discover_orgs

    return {
        org: f"{UPSTREAM_FILE}:{graph_attr_name(org)}"
        for org in discover_orgs(root)
    }


def render() -> str:
    return json.dumps(
        {
            "dependencies": ["."],
            "graphs": graph_specs(),
            # Same custom-app seam as langgraph.json — Aegra's app_loader DOES
            # support module-path app specs, so this rides through unchanged.
            "http": {"app": HTTP_APP},
            "python_version": "3.13",
        },
        indent=2,
        sort_keys=True,
    ) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if manifest is stale")
    args = ap.parse_args()

    rendered = render()
    current = MANIFEST.read_text() if MANIFEST.is_file() else ""

    if args.check:
        if current != rendered:
            print(
                "STALE: aegra.json does not match the discovered orgs tree.",
                file=sys.stderr,
            )
            print("Run:  uv run python scripts/gen_aegra_json.py", file=sys.stderr)
            return 1
        print("OK: aegra.json is up to date.")
        return 0

    MANIFEST.write_text(rendered)
    specs = graph_specs()
    print(f"wrote {MANIFEST} ({len(specs)} graph_ids):")
    for gid in sorted(specs):
        print(f"  {gid} -> {specs[gid]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
