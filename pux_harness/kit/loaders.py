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

from . import _paths


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
    """Resolve an org's directory across ALL roots — the ONE org-directory
    resolver. Delegates to ``_paths.search_org_dir`` so a ``pux:<base>`` name
    resolves against the shipped library bases and a bare name searches the
    project's ``orgs/`` (top-level, then ``specialists/``) plus every
    ``$PUX_ORG_PATHS`` root. Raises ``FileNotFoundError`` if no root has it."""
    return _paths.search_org_dir(name, project_root)


# --- org inheritance (``org.yaml extends:``) ---------------------
#
# An org may declare ``extends: <parent-org>`` in its ``org.yaml`` to inherit
# the parent's ROSTER (``agents:``), AGENTS.md overlay, and profile.yaml. This
# is the org-level analogue of an agent's ``extends:``: the parent is
# the BASE, the child SPECIALIZES. Three things compose root→child:
#
# * roster — parent ``agents:`` ∪ own (``org_agent_slugs``); an inherited slug
#   resolves through the child's agent dirs FIRST (``_agent_search_dirs`` is
#   chain-aware), so a child specializes an inherited agent by dropping a
#   same-named ``<slug>.md`` in its own ``agents/``.
# * AGENTS.md overlay — parent + own concatenated own-last (``_chain_overlay``).
# * profile.yaml — deep-merged root→child (``profile._resolved_profile_yaml``).
#
# ``policy.yaml`` is NEVER inherited (security — each org owns its egress); the
# contract warns on a policy-less child (Safeguard S6).
#
# Cycle-safety: ``org_extends_chain`` RAISES on a cycle / unresolvable parent
# (mirrors ``_load_agent_spec``'s agent-extends recursion); the runtime loaders
# use the cycle-safe ``_resolved_org_chain`` (falls back to ``[name]``), and the
# contract walks RAW (``contract._org_extends_chain_violations``) for precise
# ``org-extends-resolvable`` / ``org-extends-acyclic`` messages. Two walkers, one
# pattern — exactly the agent-extends split (``_load_agent_spec`` raises vs
# ``_agent_extends_chain_violations`` reports).


def org_extends(name: str, project_root: Path) -> str | None:
    """This org's single-hop ``extends:`` parent (a raw read of ``org.yaml``), or
    ``None``. ``None`` when the org ships no ``org.yaml``, no ``extends`` key, or
    a non-string ``extends``. The RAW reader the contract's chain walker +
    ``org_extends_chain`` build on (no recursion, no merge — single hop only)."""
    manifest = _org_path(name, project_root) / "org.yaml"
    if not manifest.is_file():
        return None
    data = yaml.safe_load(manifest.read_text()) or {}
    if not isinstance(data, dict):
        return None
    parent = data.get("extends")
    if not isinstance(parent, str) or not parent.strip():
        return None
    return parent.strip()


def org_extends_chain(name: str, project_root: Path) -> list[str]:
    """The org's inheritance chain, ROOT→CHILD (``[grandparent, parent, child]``),
    walking ``extends:`` recursively. RAISES on a broken chain so the fault
    surfaces loudly:

    * ``ValueError`` — a cycle (``a extends b extends a``).
    * ``FileNotFoundError`` — an unresolvable parent (no such org dir, or the
      parent dir has no ``AGENTS.md`` — an org without one is not a valid base).

    Mirrors ``_load_agent_spec``'s agent-extends recursion. Runtime loaders use
    the cycle-safe ``_resolved_org_chain``; the contract walks RAW via
    ``_org_extends_chain_violations`` for precise, actionable messages."""
    upward: list[str] = []  # built child→root, reversed before return
    visited: set[str] = set()
    cur = name
    while True:
        if cur in visited:
            cycle = " -> ".join([*upward, cur])
            msg = f"org {name!r}: extends cycle ({cycle})"
            raise ValueError(msg)
        visited.add(cur)
        upward.append(cur)
        parent = org_extends(cur, project_root)
        if parent is None:
            break  # chain terminates cleanly
        try:
            pdir = _org_path(parent, project_root)
        except FileNotFoundError as exc:
            msg = f"org {name!r}: extends {parent!r} -> no such org"
            raise FileNotFoundError(msg) from exc
        if not (pdir / "AGENTS.md").is_file():
            msg = f"org {name!r}: extends {parent!r} -> no AGENTS.md (not a valid base org)"
            raise FileNotFoundError(msg)
        cur = parent
    upward.reverse()  # root→child
    return upward


def _resolved_org_chain(name: str, project_root: Path) -> list[str]:
    """Cycle-safe inheritance chain, ROOT→CHILD. Falls back to ``[name]`` on a
    broken chain (cycle / unresolvable parent) so runtime loaders NEVER crash —
    the contract's ``org-extends-*`` rules report the real fault offline. For an
    org with no ``extends:`` this is just ``[name]`` (byte-identical to today)."""
    try:
        return org_extends_chain(name, project_root)
    except (ValueError, FileNotFoundError):
        return [name]


def _agent_search_dirs(org: str, project_root: Path) -> list[Path]:
    """Directories searched for an agent ``<slug>.md``, child-local first then
    each ancestor's, then shared. First hit wins, so an org specializes an
    INHERITED agent (one it got from a parent's roster via ``extends:``) by
    placing a same-named ``<slug>.md`` in its own ``agents/`` dir; it specializes
    a SHARED agent the same way against ``orgs/_shared/agents``.

    Chain-aware: walks the inheritance chain child→root
    (``_resolved_org_chain`` reversed), appending each ancestor's ``agents/``
    dir, so an inherited slug whose ``<slug>.md`` lives in a parent's
    ``agents/`` resolves. Cycle-safe (falls back to ``[org]``).

    Each ancestor resolves through ``_org_path`` (the ONE resolver),
    so a ``pux:`` ancestor (a library base) contributes the BASE's own
    ``agents/`` dir, and a ``$PUX_ORG_PATHS`` org contributes its own. The
    ``_shared`` fallback stays PROJECT-LOCAL (the consumer app's shared agents).
    For a non-extending local org the chain is ``[org]`` → byte-identical to
    the previous list."""
    chain = _resolved_org_chain(org, project_root)  # root→child
    local: list[Path] = []
    for ancestor in reversed(chain):  # child→root (child's agents win)
        try:
            adir = _org_path(ancestor, project_root) / "agents"
        except FileNotFoundError:
            continue  # ancestor doesn't resolve (minimal fixture / broken chain)
        if adir.is_dir():
            local.append(adir)
    return [*local, _orgs_dir(project_root) / "_shared" / "agents"]


def _read(rel: str, project_root: Path) -> str:
    """Read a project-relative file (used for the root ``AGENTS.md``)."""
    return (project_root / rel).read_text()


# --- org discovery + roster ------------------------------------------------

def discover_orgs(project_root: Path) -> list[str]:
    """Sorted names of every org dir containing ``AGENTS.md``. Scans the
    project's ``orgs/`` (top-level) + ``orgs/specialists/`` (nested), then each
    ``$PUX_ORG_PATHS`` root (top-level + its ``specialists/``). Library bases
    (``pux:``) are NOT auto-discovered — they're opt-in via the namespace, so a
    consumer app's org list stays its own. De-duped + sorted."""
    names: list[str] = []
    for root in [_orgs_dir(project_root), *_paths.extra_org_roots()]:
        names.extend(_scan_orgs(root))
        names.extend(_scan_orgs(root / "specialists"))
    return sorted(set(names))


def _own_org_agent_slugs(name: str, project_root: Path) -> list[str]:
    """This org's OWN roster (``org.yaml agents:``) — NO inheritance. Raises
    ``ValueError`` on a malformed ``org.yaml`` (non-mapping top level). Returns
    ``[]`` when the org ships no ``org.yaml`` (valid for a CTO-only org).

    Factored out of ``org_agent_slugs`` so the chain-aware reader can call it per
    ancestor without re-implementing the parse."""
    org_dir = _org_path(name, project_root)
    manifest = org_dir / "org.yaml"
    if not manifest.is_file():
        return []
    data = yaml.safe_load(manifest.read_text()) or {}
    if not isinstance(data, dict):
        msg = f"{name}/org.yaml: top level must be a mapping, got {type(data).__name__}"
        raise ValueError(msg)
    return _parse_list(data.get("agents"))


def org_agent_slugs(name: str, project_root: Path) -> list[str]:
    """The specialist slugs this org delegates to — the chain-INHERITED roster
    (parent ``agents:`` ∪ own, walked root→child; a slug in both appears once at
    the parent's position, own redeclarations specialized via the child's local
    ``agents/`` dir). Cycle-safe: a broken ``extends:`` chain falls back to
    ``[name]`` (the contract's ``org-extends-*`` rules report the fault).

    For a non-extending org the chain is ``[name]`` → byte-identical to reading
    just its own ``org.yaml``."""
    seen: set[str] = set()
    roster: list[str] = []
    for org in _resolved_org_chain(name, project_root):  # root→child
        for slug in _own_org_agent_slugs(org, project_root):
            if slug not in seen:
                seen.add(slug)
                roster.append(slug)
    return roster


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


def _chain_overlay(org: str, project_root: Path) -> str:
    """Concatenated ``AGENTS.md`` overlays across the inheritance chain,
    root→child (own LAST). Cycle-safe. Each ancestor's overlay is read by
    ``load_org_prompt`` (frontmatter stripped, body only). A child extends a
    parent's CTO prose the same way an agent extends a base prompt — by
    APPENDING. For a non-extending org this is just its own overlay
    (byte-identical)."""
    parts: list[str] = []
    for ancestor in _resolved_org_chain(org, project_root):  # root→child
        body = load_org_prompt(ancestor, project_root)
        if body:
            parts.append(body)
    return "\n\n".join(parts)


def build_system_prompt(org: str, *, project_root: Path, addendum: str = "") -> str:
    """root ``AGENTS.md`` + the chain-inherited org overlay + ``addendum``. The
    overlay is the parent's + own AGENTS.md concatenated (own last) when the org
    ``extends:`` a parent; otherwise just the org's own. The kit default
    addendum is empty; the pux harness passes its own ``/sandbox/workspace``
    addendum."""
    root = load_root_prompt(project_root)
    overlay = _chain_overlay(org, project_root)
    head = f"{root}\n\n{overlay}" if root else overlay
    return f"{head}{addendum}"


# --- agent specs + skills --------------------------------------------------

def _merge_extends(base: dict[str, Any], delta_fm: dict[str, Any], body: str) -> dict[str, Any]:
    """Merge a delta agent (one that declares ``extends:``) onto its resolved
    base. The base is a FULLY RESOLVED spec (already merged up its own chain);
    the delta is this agent's OWN frontmatter (``extends:`` already popped) +
    body. Returns a spec dict in the same ``{**fm, "system_prompt": body}``
    shape ``_load_agent_spec`` returns, so ``_build_sub`` consumes it unchanged.

    Merge rules (the universal per-agent override vocabulary — the SAME fields
    that work at the org level via ``profile.yaml`` work here via frontmatter):

    * ``name`` / ``description``: delta wins. ``description_append`` is
      concatenated onto the effective description (a child can ADD context
      without restating the parent's).
    * ``model``: delta wins.
    * ``tools``: an EXPLICIT ``tools:`` in the delta is a FULL-REPLACE (opt into
      a fixed whitelist — matches ``_resolve_tools`` semantics). Otherwise the
      base list is modified additively: ∪ ``tools_add`` − ``tools_remove`` (set
      semantics on the tool SUFFIX after the last ``/``; base order preserved,
      adds appended in order, dups + removes dropped).
    * ``skills``: explicit ``skills:`` full-replace; otherwise ∪ ``skills_add``
      (additive only — no remove use-case today).
    * ``tool_description_overrides``: per-key merge, delta wins (Safeguard S5 —
      keeps the legacy-block fold from leaking a second surface).
    * ``system_prompt``: base body + delta body joined with ``\\n\\n`` (the
      delta body IS ``prompt_append`` — least-specific suffix; the org-wide
      ``system_prompt_suffix`` still layers on AFTER, in ``load_subagents``).
    """
    merged: dict[str, Any] = dict(base)

    # name — delta wins.
    if "name" in delta_fm:
        merged["name"] = delta_fm["name"]

    # description — delta wins; description_append concatenates onto the
    # effective description (delta if given, else base).
    desc = merged.get("description")
    if "description" in delta_fm:
        desc = delta_fm["description"]
    if delta_fm.get("description_append"):
        desc = f"{desc or ''} {delta_fm['description_append']}".strip()
    if desc is not None:
        merged["description"] = desc

    # model — delta wins.
    if "model" in delta_fm:
        merged["model"] = delta_fm["model"]

    # tools — explicit full-replace, else additive set union/diff on suffixes.
    if "tools" in delta_fm:
        merged["tools"] = delta_fm["tools"]
    else:
        base_tools = _parse_list(base.get("tools"))
        add = _parse_list(delta_fm.get("tools_add"))
        rem = {t.rsplit("/", 1)[-1] for t in _parse_list(delta_fm.get("tools_remove"))}
        if add or rem:
            seen: set[str] = set()
            out: list[str] = []
            for entry in [*base_tools, *add]:
                suffix = entry.rsplit("/", 1)[-1]
                if suffix in rem or suffix in seen:
                    continue
                seen.add(suffix)
                out.append(entry)
            merged["tools"] = out

    # skills — explicit full-replace, else additive (dedup, order preserved).
    if "skills" in delta_fm:
        merged["skills"] = delta_fm["skills"]
    elif delta_fm.get("skills_add"):
        base_skills = _parse_list(base.get("skills"))
        add = _parse_list(delta_fm.get("skills_add"))
        seen_s: set[str] = set()
        out_s: list[str] = []
        for entry in [*base_skills, *add]:
            if entry in seen_s:
                continue
            seen_s.add(entry)
            out_s.append(entry)
        merged["skills"] = out_s

    # tool_description_overrides — per-key merge, delta wins (Safeguard S5).
    tdo = dict(base.get("tool_description_overrides") or {})
    if delta_fm.get("tool_description_overrides"):
        tdo.update(delta_fm["tool_description_overrides"])
    if tdo:
        merged["tool_description_overrides"] = tdo

    # system_prompt — base body + delta body (the delta IS prompt_append).
    base_body = base.get("system_prompt", "")
    merged["system_prompt"] = f"{base_body}\n\n{body}".strip() if body else base_body

    return merged


def _load_agent_spec(
    slug: str,
    org: str,
    project_root: Path,
    _chain: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Read ``<slug>.md`` from the org-local then ``_shared`` agent dir and
    return a spec dict (``name``/``description`` + optional ``tools``/
    ``skills``/``model`` from frontmatter; ``system_prompt`` = the body).

    Returns ``None`` if no ``<slug>.md`` exists in either search dir.

    ``extends: <base-slug>`` (recursive, cycle-detected). When the
    frontmatter carries ``extends:``, the base resolves from the SAME search
    dirs (org-local then ``_shared``) and this slug's frontmatter + body merge
    ON TOP via ``_merge_extends`` (delta wins; ``tools_add`` / ``tools_remove``
    / ``skills_add`` / ``description_append`` / ``tool_description_overrides``
    are the delta vocabulary). A cycle raises ``ValueError``; an unresolvable
    base raises ``FileNotFoundError`` — both fail loud (the contract surfaces
    them as ``agent-extends-acyclic`` / ``agent-extends-resolvable``).

    A ``pux:``-namespaced slug (roster entry or ``extends:``) resolves
    ONLY against the library bases' ``agents/`` dirs (``_paths.library_base_agent_dirs``),
    so a consumer app pulls a shipped agent without vendoring it. The cycle guard
    uses the namespaced slug as-is, so a ``pux:`` base cycle raises loud too.
    ``_chain`` is the ordered recursion guard (internal — not for callers)."""
    pux = _paths.is_pux_namespace(slug)
    look = _paths.strip_namespace(slug) if pux else slug
    search_dirs = (
        _paths.library_base_agent_dirs() if pux
        else _agent_search_dirs(org, project_root)
    )
    for d in search_dirs:
        path = d / f"{look}.md"
        if path.is_file():
            fm, body = _split_frontmatter(path.read_text())
            extends = fm.pop("extends", None)
            if extends is not None:
                if not isinstance(extends, str) or not extends.strip():
                    msg = (
                        f"agent {slug!r}: extends must be a non-empty agent slug, "
                        f"got {extends!r}"
                    )
                    raise ValueError(msg)
                extends = extends.strip()
                if slug in _chain:
                    chain = " -> ".join([*_chain, slug])
                    msg = f"agent {slug!r}: extends cycle ({chain})"
                    raise ValueError(msg)
                base = _load_agent_spec(extends, org, project_root, (*_chain, slug))
                if base is None:
                    base_dirs = (
                        _paths.library_base_agent_dirs()
                        if _paths.is_pux_namespace(extends)
                        else _agent_search_dirs(org, project_root)
                    )
                    searched = [str(p / f"{_paths.strip_namespace(extends)}.md")
                                for p in base_dirs]
                    msg = (
                        f"agent {slug!r}: extends {extends!r} -> no such agent "
                        f"(searched {searched})"
                    )
                    raise FileNotFoundError(msg)
                return _merge_extends(base, fm, body)
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


def supervisor_skills_roots(
    org: str, project_root: Path, workspace_root: str | None = None,
) -> list[str]:
    """Skills-ROOT paths for the SUPERVISOR's ``SkillsMiddleware`` — the focused
    set (``orgs/_shared/skills`` + THIS org's own ``skills/``), existing dirs
    only, mapped per ``workspace_root`` (host-absolute by default;
    container-absolute when the harness pins ``/sandbox/workspace``).

    Wires native progressive disclosure on the CTO: ``SkillsMiddleware``
    injects each root's skill METADATA (name + description) into the supervisor
    prompt, and the agent then peeks a body via the native ``read_file`` (the
    canonical path — the ``pux_sandbox_load_skill`` specialist is GONE, killed by
    the ``skills-peek-via-read-file`` contract tripwire). This is a FOCUSED set —
    the org's own + shared — NOT the broad every-org catalog
    ``pux_sandbox_list_skills`` exposes; that tool is a discovery aid that
    COMPLEMENTS the middleware, it does not duplicate it.

    Reuses ``_resolve_skills``'s validate+map (the candidate dirs are
    pre-filtered to existing, so the ``KeyError`` never fires). Returns ``[]``
    when neither root exists — a no-skills org gets ``skills=None`` at the
    binding and is byte-identical to today (no SkillsMiddleware mounted)."""
    candidates = [
        "orgs/_shared/skills",
        f"orgs/{org}/skills",
        f"orgs/specialists/{org}/skills",
    ]
    existing = [c for c in candidates if (project_root / c).is_dir()]
    return _resolve_skills(
        existing, "supervisor", project_root=project_root, workspace_root=workspace_root,
    )
