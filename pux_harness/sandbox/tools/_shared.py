"""Shared constants, utilities, and the specialist roster for the tools/ package.

Extracted from the original monolithic ``tools.py`` to break circular-dependency
risk: every file in this package imports from ``_shared``; ``_shared`` imports
from NOTHING in this package.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from pux_harness.kit._paths import project_root

log = logging.getLogger(__name__)

PUX_PREFIX = "pux_sandbox_"
PUX_GRADER_PREFIX = "pux_grader_"
# The host app root (where ``orgs/`` lives) is injected via the kit's resolver
# and resolved LIVE at use-site (no import-time snapshot) — NOT derived from
# this file's install path. Earlier the root was computed as ``parents[N]`` (N
# bumped when the package split added a dir level); that was a fragile coupling
# to the orchestrator repo layout, gone now.
SKILL_FILE = "SKILL.md"


def _skills_dirs(org: str | None = None) -> list[Path]:
    """Skills-ROOT directories to search, highest-priority first.

    With an ``org``: that org's ``skills/`` wins, then ``orgs/_shared/skills``,
    then every other org's skills (so a cross-org skill is still discoverable).
    Without an ``org`` (the offline ``--check`` smoke path): all roots in stable
    sorted order, no priority. Non-existent dirs are filtered out.

    Scans both ``orgs/`` and ``orgs/specialists/`` for org skills dirs."""
    orgs = project_root() / "orgs"
    roots: list[Path] = []
    if org:
        for candidate in [orgs / org / "skills", orgs / "specialists" / org / "skills"]:
            if candidate.is_dir():
                roots.append(candidate)
                break
    roots.append(orgs / "_shared" / "skills")
    seen = {str(r) for r in roots}
    for base in [orgs, orgs / "specialists"]:
        if base.is_dir():
            for p in sorted(base.glob("*/skills")):
                if str(p) not in seen:
                    roots.append(p)
                    seen.add(str(p))
    return [r for r in roots if r.is_dir()]


# NOTE: ``SPECIALISTS`` / ``SPECIALIST_TOOL_NAMES`` used to live here. They now
# DERIVE from ``REGISTRY`` in ``registry.py`` (the single source of truth) and
# are re-exported through ``__init__.py``. This module stays a leaf — it
# imports nothing from this package — so it cannot host the derived sets
# (that would pull the factory modules into a leaf).


def _tail(text: str, n: int = 800) -> str:
    """Last ``n`` chars of ``text`` — keeps stderr tails out of result
    envelopes without leaking megabytes. Mirrors the Go ``tailOutput`` helper."""
    return text if len(text) <= n else "..." + text[len(text) - n:]


def _result(obj: dict) -> str:
    """Serialize a tool-result dict to the exact JSON the Go bridge surfaced
    (2-space indent + sorted map keys at every level)."""
    return json.dumps(obj, indent=2, sort_keys=True)


class _NoArgs(BaseModel):
    """Schema for argument-less tools (list_skills, browser_tabs, etc.)."""
