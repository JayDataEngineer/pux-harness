"""Org + specialist-agent LOADING — backend-agnostic.

This is the portable core of the harness's ``agent/orgs.py``: it knows how to
read an org (``AGENTS.md`` + ``org.yaml`` + ``agents/<slug>.md`` + ``skills/``)
off the filesystem and turn it into data (prompt strings, agent specs, skills
source paths). It depends ONLY on the stdlib + ``yaml`` — no Docker sandbox, no
pux context layer, no model registry. That's what makes it reusable from a
different, standalone project.

Every function that needs to find files takes an explicit ``project_root`` (the
directory that contains ``orgs/`` and the root ``AGENTS.md``). There is no
module-level ``PROJECT_ROOT`` global — the caller (the kit compiler, the pux
harness shim, or a consumer app) supplies it.

System prompt shape (mirrors the harness): root ``AGENTS.md`` body +
``orgs/<name>/AGENTS.md`` body + an optional ``addendum``. The kit default
addendum is empty (the pux harness supplies its own ``/sandbox/workspace``
addendum via its shim — see ``harness/pux_harness/agent/orgs.py``).

An org is a self-contained bundle: ``orgs/<name>/agents/<slug>.md`` is ONE file
— YAML frontmatter (``name``/``description`` + optional ``tools``/``skills``/
``model``) + a markdown body that IS the system prompt (mirrors ``SKILL.md``).
The roster is ``orgs/<name>/org.yaml`` (``agents: [slug, ...]``); ``AGENTS.md``
is pure CTO-prompt prose. Cross-org agents live in ``orgs/_shared/agents/``;
resolution is org-local first, then ``_shared``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# --- pure helpers ----------------------------------------------------------

def _parse_list(raw: Any) -> list[str]:
    """A list value -> stripped non-empty items. Accepts either a YAML list
    (``[a, b]``) or a comma-separated scalar (``agents: a,b``)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def _scan_orgs(root: Path) -> list[str]:
    """Scan a single directory for org subdirs (dirs containing ``AGENTS.md``)."""
    out: list[str] = []
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "AGENTS.md").is_file():
            out.append(child.name)
    return out


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a ``.md`` file into ``(frontmatter, body)``.

    Frontmatter is the optional leading ``---``-delimited YAML block, parsed
    with ``yaml.safe_load``. Body is the markdown after the closing ``---``. No
    frontmatter -> ``({}, body)``. A non-mapping frontmatter block or a YAML
    syntax error raises ``ValueError`` (fail loud).
    """
    if not text.startswith("---"):
        return {}, text.strip()
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text.strip()
    _, head, body = parts
    try:
        fm = yaml.safe_load(head) or {}
    except yaml.YAMLError as e:  # pragma: no cover - covered by contract tests
        msg = f"invalid YAML frontmatter: {e}"
        raise ValueError(msg) from e
    if not isinstance(fm, dict):
        msg = f"frontmatter must be a YAML mapping, got {type(fm).__name__}"
        raise ValueError(msg)
    return fm, body.strip()


# --- path helpers (project_root-parameterized) -----------------------------

def _orgs_dir(project_root: Path) -> Path:
    return project_root / "orgs"


def _specialists_dir(project_root: Path) -> Path:
    return _orgs_dir(project_root) / "specialists"


def _org_path(name: str, project_root: Path) -> Path:
    """Resolve an org's directory — top-level ``orgs/`` first, then
    ``orgs/specialists/``. Raises ``FileNotFoundError`` if neither exists."""
    top = _orgs_dir(project_root) / name
    if top.is_dir():
        return top
    spec = _specialists_dir(project_root) / name
    if spec.is_dir():
        return spec
    raise FileNotFoundError(f"org {name!r} not found in orgs/ or orgs/specialists/")


def _agent_search_dirs(org: str, project_root: Path) -> list[Path]:
    """Directories searched for an agent ``<slug>.md``, org-local first then
    shared. An org specializes a shared agent by placing a same-named
    ``<slug>.md`` in its own ``agents/`` dir (first hit wins).

    Checks both ``orgs/<org>/agents`` and ``orgs/specialists/<org>/agents`` for
    the org-local dir (the latter holds orgs nested under ``specialists/``)."""
    orgs = _orgs_dir(project_root)
    local: list[Path] = []
    for candidate in [orgs / org / "agents", _specialists_dir(project_root) / org / "agents"]:
        if candidate.is_dir():
            local.append(candidate)
    return [*local, orgs / "_shared" / "agents"]


def _read(rel: str, project_root: Path) -> str:
    """Read a project-relative file (used for the root ``AGENTS.md``)."""
    return (project_root / rel).read_text()


# --- org discovery + roster ------------------------------------------------

def discover_orgs(project_root: Path) -> list[str]:
    """Sorted names of every org dir containing ``AGENTS.md``. Scans both
    ``orgs/`` (top-level orgs) and ``orgs/specialists/`` (nested orgs)."""
    return sorted(
        _scan_orgs(_orgs_dir(project_root))
        + _scan_orgs(_specialists_dir(project_root))
    )


def org_agent_slugs(name: str, project_root: Path) -> list[str]:
    """The specialist slugs this org delegates to, read from
    ``orgs/<name>/org.yaml``."""
    org_dir = _org_path(name, project_root)
    manifest = org_dir / "org.yaml"
    if not manifest.is_file():
        return []
    data = yaml.safe_load(manifest.read_text()) or {}
    if not isinstance(data, dict):
        msg = f"{name}/org.yaml: top level must be a mapping, got {type(data).__name__}"
        raise ValueError(msg)
    return _parse_list(data.get("agents"))


# --- prompt assembly -------------------------------------------------------

def load_root_prompt(project_root: Path) -> str:
    """Body of the root ``AGENTS.md`` (the base system prompt). Returns ``""``
    if there is no root ``AGENTS.md`` — a standalone consumer app may keep its
    base prompt entirely in the org overlay."""
    path = project_root / "AGENTS.md"
    if not path.is_file():
        return ""
    return _split_frontmatter(path.read_text())[1]


def load_org_prompt(name: str, project_root: Path) -> str:
    """Body of ``orgs/<name>/AGENTS.md`` (the per-org CTO overlay)."""
    return _split_frontmatter((_org_path(name, project_root) / "AGENTS.md").read_text())[1]


def build_system_prompt(org: str, *, project_root: Path, addendum: str = "") -> str:
    """root ``AGENTS.md`` + org overlay + ``addendum``. The kit default addendum
    is empty; the pux harness passes its own ``/sandbox/workspace`` addendum."""
    root = load_root_prompt(project_root)
    overlay = load_org_prompt(org, project_root)
    head = f"{root}\n\n{overlay}" if root else overlay
    return f"{head}{addendum}"


# --- agent specs + skills --------------------------------------------------

def _load_agent_spec(slug: str, org: str, project_root: Path) -> dict[str, Any] | None:
    """Read ``<slug>.md`` from the org-local then ``_shared`` agent dir and
    return a spec dict (``name``/``description`` + optional ``tools``/
    ``skills``/``model`` from frontmatter; ``system_prompt`` = the body).

    Returns ``None`` if no ``<slug>.md`` exists in either search dir."""
    for d in _agent_search_dirs(org, project_root):
        path = d / f"{slug}.md"
        if path.is_file():
            fm, body = _split_frontmatter(path.read_text())
            return {**fm, "system_prompt": body}
    return None


def _resolve_skills(
    raw: Any, slug: str, *, project_root: Path, workspace_root: str | None = None,
) -> list[str]:
    """``skills`` value -> skills-ROOT paths for deepagents' ``SkillsMiddleware``.

    deepagents' ``SkillsMiddleware`` resolves each source against the BACKEND and
    loads EVERY ``<source>/<skill>/SKILL.md`` beneath it — a source is a skills
    **root** directory, not an individual skill. So a value is a
    **project-relative** directory (e.g. ``orgs/_shared/skills`` or
    ``orgs/<org>/skills``); we validate it exists under ``project_root``.

    Path mapping (the one thing that differs between consumers):

    - ``workspace_root=None`` (the kit default, for a local ``FilesystemBackend``):
      returns ABSOLUTE local paths (``<project_root>/<p>``) so skills resolve on
      the host filesystem of a standalone app.
    - ``workspace_root="/sandbox/workspace"`` (the pux harness): returns
      container-absolute paths (``/sandbox/workspace/<p>``), because the project
      is bind-mounted 1:1 at that path inside the sandbox container.
    """
    out: list[str] = []
    for p in _parse_list(raw):
        if not isinstance(p, str) or not p:
            msg = f"{slug}: each skills source must be a non-empty path string"
            raise ValueError(msg)
        if p.startswith("/") or ".." in Path(p).parts:
            msg = f"{slug}: skills source must be project-relative (got {p!r})"
            raise ValueError(msg)
        if not (project_root / p).is_dir():
            raise KeyError(
                f"{slug}: skills source {p!r} -> no such directory under the project root"
            )
        out.append(
            f"{workspace_root}/{p}" if workspace_root else str(project_root / p)
        )
    return out
