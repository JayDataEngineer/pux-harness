"""pux_sandbox_list_skills / pux_sandbox_load_skill — host-FS skill discovery."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

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
    "skills first, then orgs/_shared/skills). Each skill is operator-authored "
    "markdown with model-facing instructions (debugging recipes, codebase "
    "conventions, domain knowledge). Call this when starting work on a project "
    "to see what specialized guidance is available; then call load_skill to "
    "read the ones that apply."
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


class _LoadSkillArgs(BaseModel):
    name: str = Field(..., description="Skill name (the 'name' field from list_skills)")


_LOAD_SKILL_DESC = (
    "Load one skill's full markdown body by name (use list_skills first to "
    "discover names). Returns name, description, source path, and the markdown "
    "content. Read the content carefully — it carries operator-authored "
    "instructions specific to this project."
)


def _load_skill_tool(org: str | None = None) -> StructuredTool:
    def _run(name: str) -> str:
        if not name:
            return _result({"success": False, "error": "missing required parameter 'name'"})
        md: Path | None = None
        for root in _skills_dirs(org):
            candidate = root / name / SKILL_FILE
            if candidate.is_file():
                md = candidate
                break
        if md is None:
            return _result({"success": False, "error": f"skill {name!r} not found"})
        raw = md.read_text()
        nm, desc = _parse_skill(raw)
        body = raw.split("---", 2)[2].strip() if raw.startswith("---") else raw.strip()
        return _result({"name": nm or name, "description": desc, "path": str(md), "content": body})

    return StructuredTool(
        name=PUX_PREFIX + "load_skill", description=_LOAD_SKILL_DESC,
        args_schema=_LoadSkillArgs, func=_run,
    )
