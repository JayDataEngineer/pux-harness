"""The location-independent project-root resolver — the ONE shared seam.

Every harness subsystem that needs the *app* root (where ``orgs/``, ``.pux/``,
``AGENTS.md`` live) calls :func:`project_root` instead of computing it from
``__file__``. Computing the root from the install path
(``Path(__file__).resolve().parents[N]``) only worked while ``harness/`` lived
*inside* the orchestrator repo — it shatters the moment ``pux_harness`` is
installed elsewhere (its own repo, a venv site-packages, a consumer app). The
kit is the natural home: it is the slim, import-pinned core (the Stage-2
``kit-import-isolation`` tripwire guarantees this module can never reach a
heavy subsystem), so the heavy layer can depend on it without a cycle.

Resolution order: explicit ``$PUX_PROJECT_ROOT`` override, else the process
CWD. The kit's own :func:`compile_org` uses the SAME resolver as its
``project_root`` default, so there is ONE source of truth across the kit AND
the heavy harness runtimes (``serve`` / ``acp`` / ``direct`` / ``tui``).
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT_ENV = "PUX_PROJECT_ROOT"


def project_root() -> Path:
    """The app root: ``$PUX_PROJECT_ROOT`` if set, else the resolved CWD.

    ``bin/pux`` exports ``PUX_PROJECT_ROOT=$REPO`` before exec, so a
    harness launched from anywhere still finds the orchestrator's
    ``orgs/`` + ``.pux/``. A standalone kit consumer (no ``bin/pux``)
    passes ``project_root=`` to :func:`compile_org` explicitly, or lets it
    fall back to the CWD — same resolver, same behavior.
    """
    return Path(os.environ.get(_PROJECT_ROOT_ENV, ".")).resolve()
