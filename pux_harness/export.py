"""Export an org as a standalone portable archive.

``pux export --org <name>`` produces a ``.tar.gz`` containing every primitive
the org needs to run outside the harness — prompts, agent definitions, skills,
policy, profile, shared dependencies, and a manifest. The archive is
self-contained: a consumer can reconstruct the org's agent graph without the
pux harness.

Archive layout (mirrors the project tree, scoped to what the org uses)::

    <org-name>/
      AGENTS.md                      # root base prompt
      orgs/<name>/
        AGENTS.md                    # CTO prompt
        org.yaml                     # roster
        profile.yaml                 # (if present)
        policy.yaml                  # (if present)
        agents/*.md                  # specialist definitions
        skills/*/SKILL.md            # org-local skills
        sandbox/*.py                 # org-local scripts
        config/                      # (if present)
        Dockerfile                   # (if present)
      # NOTE: org ``data/`` is INTENTIONALLY EXCLUDED. It is runtime state
      # (auth sessions, market data, campaign state — see the root .gitignore,
      # which treats data/ as non-source), often containing live secrets such as
      # browser-session cookies. It is never a primitive an export reconstructs.
      orgs/_shared/
        agents/*.md                  # shared agents the org resolves
        skills/*/SKILL.md            # shared skills the org resolves
        sandbox/*.py                 # shared sandbox helpers the org references
        clients/*.py                 # shared clients the org references
        tool_servers.yaml            # (if policy declares tool_servers)
      manifest.json                  # machine-readable inventory
"""
from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml

from pux_harness.agent.orgs import (
    _load_agent_spec,
    _org_path,
    _orgs_dir,
    _parse_list,
    discover_orgs,
    org_agent_slugs,
)
from pux_harness.kit._paths import project_root as _default_project_root


def _collect_org_files(org_dir: Path) -> dict[str, Path]:
    """Map of archive-relative-path -> host-path for every file in the org dir.

    Paths are normalized so ``orgs/specialists/<name>/`` is exported as
    ``orgs/<name>/`` — the consumer shouldn't care about the host layout.
    """
    files: dict[str, Path] = {}
    if not org_dir.is_dir():
        return files

    org_name = org_dir.name

    # Core files (always include if present)
    for name in (
        "AGENTS.md", "org.yaml", "profile.yaml", "policy.yaml", "Dockerfile",
    ):
        p = org_dir / name
        if p.is_file():
            files[f"orgs/{org_name}/{name}"] = p

    # Recursive dirs. ``data`` is DELIBERATELY ABSENT: it holds runtime state
    # (auth sessions, market data, campaign state), frequently live secrets
    # (e.g. ``.twitter-session.json`` browser cookies). Bundling it would leak
    # credentials into the export archive. See test_export.py's
    # ``test_collect_org_files_excludes_data_dir`` for the permanent contract.
    for dirname in ("agents", "skills", "sandbox", "config"):
        d = org_dir / dirname
        if not d.is_dir():
            continue
        for path in sorted(d.rglob("*")):
            if path.is_file():
                try:
                    path.read_bytes()  # verify readability
                except (PermissionError, OSError):
                    continue
                # Normalize: orgs/specialists/<name>/X -> orgs/<name>/X
                rel_to_org = path.relative_to(org_dir)
                files[f"orgs/{org_name}/{rel_to_org}"] = path

    return files


def _resolve_shared_agents(org: str) -> dict[str, Path]:
    """Shared agents the org needs (slugs from org.yaml that resolve to _shared)."""
    files: dict[str, Path] = {}
    shared_dir = _orgs_dir() / "_shared" / "agents"
    if not shared_dir.is_dir():
        return files

    org_local_agents = _orgs_dir() / org / "agents"
    if not org_local_agents.is_dir():
        org_local_agents = _orgs_dir() / "specialists" / org / "agents"

    for slug in org_agent_slugs(org):
        # If org-local exists, it was already captured by _collect_org_files
        if org_local_agents.is_dir() and (org_local_agents / f"{slug}.md").is_file():
            continue
        shared_path = shared_dir / f"{slug}.md"
        if shared_path.is_file():
            files[f"orgs/_shared/agents/{slug}.md"] = shared_path

    return files


def _resolve_shared_skills(
    org: str, project_root: Path | None = None,
) -> dict[str, Path]:
    """Shared skills referenced by agents' ``skills:`` frontmatter.

    ``project_root`` defaults to the kit's LIVE resolver (no import-time
    snapshot) — per-call overridable so ``export_org`` can thread a tmp root."""
    root = project_root if project_root is not None else _default_project_root()
    files: dict[str, Path] = {}
    shared_skills = _orgs_dir() / "_shared" / "skills"
    if not shared_skills.is_dir():
        return files

    # Walk every agent's skills: field
    for slug in org_agent_slugs(org):
        spec = _load_agent_spec(slug, org)
        if spec is None:
            continue
        for raw in _parse_list(spec.get("skills", [])):
            # raw is project-relative like "orgs/_shared/skills"
            if not isinstance(raw, str) or "_shared" not in raw:
                continue
            skills_root = root / raw
            if not skills_root.is_dir():
                continue
            for skill_dir in sorted(skills_root.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if skill_md.is_file():
                    rel = skill_md.relative_to(root)
                    files[str(rel)] = skill_md
                # Include references/ and scripts/ subdirs
                for sub in ("references", "scripts"):
                    sub_dir = skill_dir / sub
                    if sub_dir.is_dir():
                        for f in sorted(sub_dir.rglob("*")):
                            if f.is_file():
                                rel = f.relative_to(root)
                                files[str(rel)] = f

    return files


def _resolve_shared_sandbox(
    org: str, project_root: Path | None = None,
) -> dict[str, Path]:
    """Shared sandbox helpers referenced by policy host_setup hooks.

    ``project_root`` defaults to the kit's LIVE resolver — per-call overridable."""
    root = project_root if project_root is not None else _default_project_root()
    files: dict[str, Path] = {}
    org_dir = _org_path(org)
    policy_path = org_dir / "policy.yaml"
    if not policy_path.is_file():
        return files

    try:
        policy = yaml.safe_load(policy_path.read_text()) or {}
    except yaml.YAMLError:
        return files

    if not isinstance(policy, dict):
        return files

    hooks = policy.get("host_setup") or []
    if not isinstance(hooks, list):
        return files

    shared_sandbox = _orgs_dir() / "_shared" / "sandbox"
    shared_clients = _orgs_dir() / "_shared" / "clients"

    for hook in hooks:
        if not isinstance(hook, dict):
            continue
        script = hook.get("helper_script", "")
        if not isinstance(script, str) or not script:
            continue
        # script is project-relative like "orgs/_shared/sandbox/extract_browser_cookies.py"
        if "_shared" not in script:
            continue
        script_path = root / script
        if script_path.is_file():
            files[str(Path(script).parent / script_path.name)] = script_path

    # Also include shared clients if any org-local sandbox script imports them
    org_sandbox = org_dir / "sandbox"
    if org_sandbox.is_dir() and shared_clients.is_dir():
        for py_file in sorted(org_sandbox.glob("*.py")):
            try:
                content = py_file.read_text()
            except Exception:
                continue
            if "from" in content and ("clients" in content or "_shared" in content):
                for client_file in sorted(shared_clients.glob("*.py")):
                    rel = client_file.relative_to(root)
                    files[str(rel)] = client_file
                break  # one check is enough to decide we need clients

    # Include shared sandbox helpers that org sandbox scripts might import
    if org_sandbox.is_dir() and shared_sandbox.is_dir():
        for py_file in sorted(org_sandbox.glob("*.py")):
            try:
                content = py_file.read_text()
            except Exception:
                continue
            if "from" in content and "sandbox" in content:
                for helper in sorted(shared_sandbox.glob("*.py")):
                    rel = helper.relative_to(root)
                    files[str(rel)] = helper
                break

    return files


def _resolve_tool_servers(org: str) -> dict[str, Path]:
    """Shared tool_servers.yaml if the org's policy declares tool_servers."""
    files: dict[str, Path] = {}
    org_dir = _org_path(org)
    policy_path = org_dir / "policy.yaml"
    if not policy_path.is_file():
        return files

    try:
        policy = yaml.safe_load(policy_path.read_text()) or {}
    except yaml.YAMLError:
        return files

    if not isinstance(policy, dict):
        return files

    tool_servers = policy.get("tool_servers")
    if not tool_servers:
        return files

    ts_path = _orgs_dir() / "_shared" / "tool_servers.yaml"
    if ts_path.is_file():
        files["orgs/_shared/tool_servers.yaml"] = ts_path

    return files


def _build_manifest(
    org: str,
    files: dict[str, Path],
) -> dict[str, Any]:
    """Machine-readable inventory of the exported archive."""
    org_slugs = org_agent_slugs(org)

    # Categorize files
    categories: dict[str, list[str]] = {
        "root_prompt": [],
        "org_core": [],
        "org_agents": [],
        "org_skills": [],
        "org_sandbox": [],
        "org_other": [],
        "shared_agents": [],
        "shared_skills": [],
        "shared_sandbox": [],
        "shared_clients": [],
        "shared_tool_servers": [],
    }

    for rel in sorted(files):
        if rel == "AGENTS.md":
            categories["root_prompt"].append(rel)
        elif rel.startswith(f"orgs/{org}/agents/"):
            categories["org_agents"].append(rel)
        elif rel.startswith(f"orgs/{org}/skills/"):
            categories["org_skills"].append(rel)
        elif rel.startswith(f"orgs/{org}/sandbox/"):
            categories["org_sandbox"].append(rel)
        elif rel.startswith(f"orgs/{org}/"):
            categories["org_core"].append(rel)
        elif rel.startswith("orgs/_shared/agents/"):
            categories["shared_agents"].append(rel)
        elif rel.startswith("orgs/_shared/skills/"):
            categories["shared_skills"].append(rel)
        elif rel.startswith("orgs/_shared/sandbox/"):
            categories["shared_sandbox"].append(rel)
        elif rel.startswith("orgs/_shared/clients/"):
            categories["shared_clients"].append(rel)
        elif rel.startswith("orgs/_shared/tool_servers"):
            categories["shared_tool_servers"].append(rel)
        else:
            categories["org_other"].append(rel)

    return {
        "org": org,
        "description": f"Standalone export of the {org} org",
        "agent_roster": org_slugs,
        "total_files": len(files),
        "categories": categories,
        "files": sorted(files.keys()),
    }


def export_org(
    org: str,
    output: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Export an org as a self-contained ``.tar.gz`` archive.

    ``project_root`` defaults to the kit's LIVE resolver (``$PUX_PROJECT_ROOT``
    or the CWD) — per-call overridable, mirroring ``kit.compile_org``. Returns
    the path to the written archive.
    """
    root = project_root if project_root is not None else _default_project_root()
    if org not in discover_orgs():
        raise FileNotFoundError(
            f"org {org!r} not found; discovered: {discover_orgs()}")

    org_dir = _org_path(org)

    # 1. Root AGENTS.md (base prompt)
    root_agents_md = root / "AGENTS.md"
    files: dict[str, Path] = {}
    if root_agents_md.is_file():
        files["AGENTS.md"] = root_agents_md

    # 2. Org-local files
    files.update(_collect_org_files(org_dir))

    # 3. Shared dependencies
    files.update(_resolve_shared_agents(org))
    files.update(_resolve_shared_skills(org, root))
    files.update(_resolve_shared_sandbox(org, root))
    files.update(_resolve_tool_servers(org))

    # 4. Build manifest
    manifest = _build_manifest(org, files)

    # 5. Write tar.gz
    if output is None:
        output = Path(f"{org}.tar.gz")

    with tarfile.open(output, "w:gz") as tar:
        # Add a top-level directory entry
        info = tarfile.TarInfo(name=f"{org}/")
        info.type = tarfile.DIRTYPE
        tar.addfile(info)

        for archive_path, host_path in sorted(files.items()):
            try:
                data = host_path.read_bytes()
            except (PermissionError, OSError):
                continue
            info = tarfile.TarInfo(name=f"{org}/{archive_path}")
            info.size = len(data)
            tar.addfile(info, BytesIO(data))

        # Add manifest
        manifest_json = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo(name=f"{org}/manifest.json")
        info.size = len(manifest_json)
        tar.addfile(info, BytesIO(manifest_json))

    return output
