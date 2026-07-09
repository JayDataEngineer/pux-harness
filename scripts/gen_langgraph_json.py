#!/usr/bin/env python3
"""Generate ``langgraph.json`` — the upstream contract artifact for the Agent
Protocol (k3s) lane — from the discovered orgs/ tree.

Each org becomes ONE ``graph_id`` (the SDK dispatch key) pointing at its lazy
factory attr in ``pux_harness.runtime.upstream`` (``graph__<slug>``).
``langgraph dev`` (local AP smoke) and ``langgraph build`` (k3s Docker image)
both consume this manifest. Regenerate whenever the orgs/ set changes.

This is the SOURCE OF TRUTH for the manifest — the committed ``langgraph.json``
is its output. Keeps the manifest + the dynamic factory registration in sync
without code-gen of Python (the factories self-register in ``upstream.py``;
only the graph_id -> attr mapping needs to be declared here).

Usage:  uv run python scripts/gen_langgraph_json.py [--check]
  --check: exit 1 if the committed langgraph.json would change (CI guard).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # pux-harness/
MANIFEST = ROOT / "langgraph.json"
MODULE = "pux_harness.runtime.upstream"
# The langgraph-api ``user_router`` custom app — pux's unique surfaces
# (/events* EventBus + /jobs/{org}/* prep) composed UNDER upstream's AP CRUD.
# See pux_harness/runtime/custom_app.py + plan-p3-server-rest-retirement.
HTTP_APP = "pux_harness.runtime.custom_app:app"


def _project_root() -> Path | None:
    """Where the orgs/ tree lives (the parent repo; pux-harness is its submodule)."""
    env = os.environ.get("PUX_PROJECT_ROOT")
    if env:
        return Path(env)
    candidate = ROOT.parent  # .../auto-developer-orchestrator
    return candidate if (candidate / "orgs").is_dir() else None


def graph_specs() -> dict[str, str]:
    """``{graph_id: "module:graph__<slug>"}`` for every discovered org."""
    from pux_harness.runtime.upstream import graph_attr_name

    root = _project_root()
    if root is None:
        # Standalone kit (no orgs/ tree): ship `general` only.
        return {"general": f"{MODULE}:{graph_attr_name('general')}"}
    from pux_harness.kit.loaders import discover_orgs

    return {org: f"{MODULE}:{graph_attr_name(org)}" for org in discover_orgs(root)}


def render() -> str:
    return json.dumps(
        {
            "dependencies": ["."],
            "graphs": graph_specs(),
            # Bind pux's custom surfaces (EventBus + jobs) under upstream's AP
            # CRUD via langgraph-api's ``user_router`` seam (``http.app``).
            "http": {"app": HTTP_APP},
            # pux-harness requires Python >=3.12,<3.14; langgraph-cli's DEFAULT
            # build tag is 3.11 (langgraph_api/config.py DEFAULT_PYTHON_VERSION),
            # which is incompatible. Pin the build image's Python explicitly.
            # (langgraph dev ignores this — it uses the host venv.)
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
                "STALE: langgraph.json does not match the discovered orgs tree.",
                file=sys.stderr,
            )
            print("Run:  uv run python scripts/gen_langgraph_json.py", file=sys.stderr)
            return 1
        print("OK: langgraph.json is up to date.")
        return 0

    MANIFEST.write_text(rendered)
    specs = graph_specs()
    print(f"wrote {MANIFEST} ({len(specs)} graph_ids):")
    for gid in sorted(specs):
        print(f"  {gid} -> {specs[gid]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
