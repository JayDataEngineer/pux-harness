"""pux-specific constants + helpers for the org-coupled specialist tools.

Separated from ``_shared.py`` (which is zero-pux / portable) so the portable
tools — python/browser/desktop/grader/media — can be extracted into a
standalone package without dragging org/skill coupling with them.

Only ``skills.py``, ``declared.py``, ``dynamic.py``, and ``registry.py``
import from here.
"""

from __future__ import annotations

from pathlib import Path

from pux_harness.kit._paths import project_root

# The prefix pux applies to specialist tool names for the agent contract layer.
# Portable tools have PLAIN names (``python``, ``browser_navigate``); pux's
# ``make_specialist_tools`` applies this prefix at assembly time so org configs
# and ``resolve_tool_allowlist`` continue matching ``pux_sandbox_*``.
PUX_PREFIX = "pux_sandbox_"
PUX_GRADER_PREFIX = "pux_grader_"
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
