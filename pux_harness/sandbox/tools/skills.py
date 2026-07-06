"""pux_sandbox_list_skills — host-FS skill discovery (the CTO catalog).

The BODY-load path is gone. Native ``SkillsMiddleware``
(on the supervisor) advertises each skill's name + description in the prompt
(progressive disclosure), and the agent peeks a body with the native
``read_file`` — the canonical deepagents path. This module keeps only the
host-side ``list_skills`` CATALOG (a discovery aid that complements the
middleware's focused metadata injection by spanning EVERY org's skills, not
just the active org's + shared). ``pux_sandbox_load_skill`` was removed; the
``skills-peek-via-read-file`` contract tripwire makes a re-introduction a
HARD failure.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from pux_harness.sandbox.tools._shared import PUX_PREFIX, SKILL_FILE, _result, _skills_dirs, _NoArgs


def _parse_skill(raw: str) -> tuple[str, str]:
    """Pull (name, description) from SKILL.md frontmatter."""
    name, desc = "", ""
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            fm = parts[1]
            body = parts[2]
            for line in fm.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, _, val = line.partition(":")
                val = val.strip().strip('"').strip("'")
                if key.strip() == "name":
                    name = val
                elif key.strip() == "description":
                    desc = val
    return name, desc


_LIST_SKILLS_DESC = (
    "List SKILL.md files under the project's skills roots (the active org's "
    "skills first, then orgs/_shared/skills, then every other org's skills). "
    "Each skill is operator-authored markdown with model-facing instructions "
    "(debugging recipes, codebase conventions, domain knowledge). Call this "
    "when starting work on a project to see what specialized guidance is "
    "available; then read_file the ones that apply (the 'path' each entry "
    "gives is the SKILL.md body)."
)


def _list_skills_tool(org: str | None = None) -> StructuredTool:
    def _run() -> str:
        items: list[dict] = []
        seen: set[str] = set()
        for root in _skills_dirs(org):
            for child in sorted(root.iterdir()):
                if not child.is_dir() or child.name in seen:
                    continue
                md = child / SKILL_FILE
                if not md.is_file():
                    continue
                seen.add(child.name)
                name, desc = _parse_skill(md.read_text())
                items.append({"name": name or child.name, "description": desc, "path": str(md)})
        return _result({"skills": items, "count": len(items)})

    return StructuredTool(
        name=PUX_PREFIX + "list_skills", description=_LIST_SKILLS_DESC,
        args_schema=_NoArgs, func=_run,
    )
