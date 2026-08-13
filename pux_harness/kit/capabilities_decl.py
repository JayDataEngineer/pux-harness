"""The ``capabilities:`` declaration sugar — the desugarer (CU-3).

ONE opt-in unified key, accepted in TWO homes, routed by ``kind`` to the
existing kind-specific declaration the leaf resolvers already read. This is
PURE SUGAR: the leaf resolvers (``classify_slug`` / ``resolve_tool_servers`` /
``supervisor_skills_roots``) are UNCHANGED — CU-1's zero-behavior contract
holds. An org written in the new form resolves bit-identically to its old form
(the CU-3 parity proof in ``tests/harness/test_capabilities_decl.py``).

Homes (kind-routed). ``tool`` / ``skill`` each have ONE natural home (per-agent
frontmatter); ``mcp`` is the exception — it lives in BOTH, at TWO different layers:
- **Agent frontmatter** (``<slug>.md``): ``capabilities:`` with
  ``kind ∈ {tool, skill, mcp}`` desugars into that agent's ``tools:`` /
  ``skills:`` lists, or — for ``mcp`` — a new ``mcp:`` list
  (``desugar_agent_capabilities``). ``middleware`` / ``job`` are REJECTED here
  (no ``ref``-catalog).
- **``org.yaml``**: ``capabilities:`` with ``kind == mcp`` desugars into
  ``tool_server`` items (a bare catalog-ref string, or a
  ``{ref, tools}`` allowlist override) that ``resolve_tool_servers`` merges in
  (``org_mcp_items_from_dict``). The per-agent kinds (``tool``/``skill``) are
  REJECTED here.

The ``mcp`` two-level model: the **org** arms the server (boots its process,
owns its egress ACL) and the **agent** routes a *focused subset* of the org's
already-armed MCP tools into one subagent's whitelist. An agent ``mcp`` ref to
a server the org did NOT arm fails loud at build (the two-level grant gate —
see ``_resolve_mcp``). This is the one kind that legitimately spans both homes
(org = transport/egress, agent = routing) — unlike ``tool``/``skill``, which are
purely per-agent.

Scope (CU-3): the THREE model add-on channels the operator named — skills,
tools, mcp — all of which have a clean ``ref`` semantics (a skills-root path, a
registry/declared tool name, an mcp catalog name). ``middleware`` and ``job``
are NOT in the sugar surface: neither has a ``ref``-catalog today (middleware
toggles a registry name but is scope-dependent; jobs are inline script specs).
They stay declarable at their existing sites (``profile.yaml`` / ``policy.yaml``)
and are ALREADY unified as ``CapabilityIndex`` kinds (CU-2). Forcing them into
the sugar would INVENT a catalog — the opposite of unification.

Egress coupling: mcp is security-scoped. It stays operationally coupled to the
org's policy today because the mcp server PROCESS runs inside the org's sandbox
with the org's egress ACL applied. A static "host ∈ policy.egress.allow" contract
check is deferred — it is unreliable offline (``${VAR}``-placeholder URLs,
``stdio`` transports have no host) and would false-positive. The declaration
site (``policy.yaml tool_servers:``) and the ``org.yaml`` sugar BOTH route
through the same ``resolve_tool_servers`` (per-server isolation + credentials
unchanged), so the coupling is preserved by construction.

Both forms coexist in CU-3 (the old frontmatter keys ``tools:`` / ``skills:``
and ``policy.yaml tool_servers:`` remain valid); CU-4 makes the superseded forms
permanent contract failures. ``CapabilitiesSugarError`` is the loud failure for a
malformed ``capabilities:`` block (wrong kind for the home, bad shape, unknown
kind) — surfaced by the contract checker as a dedicated violation, never
silently skipped (no-legacy-left-behind / no-fallbacks-no-aliases).

This module is PURE DATA — it takes parsed dicts/frontmatter and returns
desugared data. File I/O stays in the callers (``kit.loaders`` for frontmatter,
``agent.tool_servers`` / ``agent.org_validation`` for ``org.yaml``), so it depends only
on the stdlib and stays cycle-free at the kit layer.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "CapabilitiesSugarError",
    "AGENT_CAPABILITY_KINDS",
    "ORG_CAPABILITY_KINDS",
    "desugar_agent_capabilities",
    "org_mcp_items_from_dict",
]


# The kinds each home accepts. ``tool`` / ``skill`` are per-agent ONLY;
# ``mcp`` is the one kind valid in BOTH homes (org arms the server; agent routes
# a focused subset — see the module docstring's two-level model). These are the
# CU-3 sugar surface: the three model add-on channels with a clean ``ref``
# semantics. ``middleware`` / ``job`` are deferred (no ref-catalog).
AGENT_CAPABILITY_KINDS = ("tool", "skill", "mcp")
ORG_CAPABILITY_KINDS = ("mcp",)


class CapabilitiesSugarError(ValueError):
    """A malformed ``capabilities:`` block — wrong kind for its home, bad shape,
    or an unknown kind. Raised loud (the contract checker surfaces it as a
    dedicated ``capabilities-sugar-*`` violation); never silently skipped."""


def _coerce_str_list(raw: Any, *, where: str) -> list[str]:
    """Coerce a frontmatter ``tools`` / ``skills`` value (list OR comma-string)
    to a ``list[str]`` — the SAME coercion ``_resolve_tools`` /
    ``_resolve_skills`` accept, so the desugared form composes with the legacy
    keys during the CU-3 transition."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    if isinstance(raw, list):
        return [str(s) for s in raw]
    raise CapabilitiesSugarError(
        f"{where}: tools/skills must be a list or comma-string, "
        f"got {type(raw).__name__}"
    )


def desugar_agent_capabilities(
    fm: dict[str, Any], slug: str
) -> dict[str, Any]:
    """Expand a frontmatter ``capabilities:`` block into ``tools:`` / ``skills:``
    / ``mcp:`` on ``fm`` (returns the merged dict). Accepts ``kind ∈ {tool,
    skill, mcp}``; ``middleware`` / ``job`` raise ``CapabilitiesSugarError``
    (not in the sugar surface — those stay in ``profile.yaml`` / ``policy.yaml``).

    ``kind == mcp`` is special: unlike tool/skill (a bare ``ref`` appended to a
    list), an mcp entry carries an OPTIONAL per-tool allowlist, so it desugars to
    a NEW ``mcp:`` key — a list in the SAME shape ``org_mcp_items_from_dict``
    produces (a bare catalog-ref string, or ``{ref, tools: [...]}``) — NOT into
    ``tools:``. The leaf resolver ``_resolve_mcp`` (harness ``agent/orgs.py`` +
    kit ``kit/compile.py``) matches the org's ARMED mcp tools by the
    ``mcp__<ref>__*`` prefix, narrowing to the allowlist's bare tool names, and
    fails loud if the ref matches no armed server (the two-level grant gate: the
    org must have armed it).

    ``capabilities:`` ADDS to any explicit ``tools:`` / ``skills:`` already on
    ``fm`` (both forms compose during the CU-3 transition): the legacy list is
    preserved first, then sugar entries appended in order, deduped. The
    ``capabilities:`` key is then POPPED — the leaf resolvers
    (``_resolve_tools`` / ``_resolve_skills`` / ``_resolve_mcp``) never see it;
    they see the merged ``tools:`` / ``skills:`` / ``mcp:`` and behave exactly as
    on an org written the old way.

    Runs in ``kit.loaders._load_agent_spec`` on EACH agent's OWN frontmatter,
    BEFORE the ``extends:`` merge — so a parent's ``capabilities:`` becomes the
    parent's ``tools:`` / ``skills:`` and the existing
    ``tools`` / ``tools_add`` / ``skills`` / ``skills_add`` inheritance machinery
    operates on the desugared form unchanged. NOTE: there is no ``mcp`` /
    ``mcp_add`` delta vocabulary yet, so agent-level ``extends:`` does NOT carry
    a ``mcp:`` list up the chain — declare ``mcp:`` on the leaf agent.
    """
    block = fm.get("capabilities")
    if block is None:
        return fm
    if not isinstance(block, list):
        raise CapabilitiesSugarError(
            f"agent {slug!r}: capabilities: must be a list of mappings, "
            f"got {type(block).__name__}"
        )
    where = f"agent {slug!r}"
    tools = _coerce_str_list(fm.get("tools"), where=where)
    skills = _coerce_str_list(fm.get("skills"), where=where)
    mcp_items: list[Any] = []
    for i, entry in enumerate(block):
        if not isinstance(entry, dict):
            raise CapabilitiesSugarError(
                f"agent {slug!r}: capabilities[{i}] must be a mapping, "
                f"got {type(entry).__name__}"
            )
        kind = entry.get("kind")
        ref = entry.get("ref")
        if kind not in AGENT_CAPABILITY_KINDS:
            raise CapabilitiesSugarError(
                f"agent {slug!r}: capabilities[{i}] kind={kind!r} is not in the "
                f"agent frontmatter sugar surface; accepts {list(AGENT_CAPABILITY_KINDS)} "
                f"(middleware/job stay in profile.yaml/policy.yaml)"
            )
        if not isinstance(ref, str) or not ref.strip():
            raise CapabilitiesSugarError(
                f"agent {slug!r}: capabilities[{i}] kind={kind!r} ref must be a "
                f"non-empty string"
            )
        ref = ref.strip()
        if kind == "mcp":
            # The one kind with an optional per-tool allowlist → ``mcp:`` key, in
            # the SAME shape ``org_mcp_items_from_dict`` emits (bare ref string,
            # or ``{ref, tools}``). Routed by ``_resolve_mcp``, NOT ``_resolve_tools``.
            allow = entry.get("allowlist", entry.get("tools"))
            if allow is None:
                mcp_items.append(ref)
            elif isinstance(allow, list):
                mcp_items.append({"ref": ref, "tools": [str(t) for t in allow]})
            else:
                raise CapabilitiesSugarError(
                    f"agent {slug!r}: capabilities[{i}] mcp allowlist must be a "
                    f"list, got {type(allow).__name__}"
                )
            continue
        target = tools if kind == "tool" else skills
        if ref not in target:
            target.append(ref)
    if tools:
        fm["tools"] = tools
    else:
        fm.pop("tools", None)
    if skills:
        fm["skills"] = skills
    else:
        fm.pop("skills", None)
    if mcp_items:
        fm["mcp"] = mcp_items
    else:
        fm.pop("mcp", None)
    fm.pop("capabilities", None)  # consumed — the leaves never see it
    return fm


def org_mcp_items_from_dict(
    data: dict[str, Any] | None, org: str
) -> list[Any]:
    """The ``mcp`` entries from an ``org.yaml`` ``capabilities:`` block, in the
    SAME shape ``resolve_tool_servers`` consumes (a bare catalog-ref string, or
    a ``{ref: name, tools: [...]}`` allowlist-override mapping). Empty when the
    org ships no ``org.yaml`` or no ``capabilities:`` / no ``mcp`` entries.

    Accepts ONLY ``kind == mcp`` in ``org.yaml``; ``tool`` / ``skill`` raise
    (wrong home — per-agent kinds go in frontmatter). Any other ``kind`` raises
    too — ``middleware`` / ``job`` are not in the sugar surface (deferred), so a
    typo or an attempted-unsupported kind fails loud rather than silently
    dropping.

    ``data`` is the parsed ``org.yaml`` mapping (or ``None`` when absent) — file
    I/O is the caller's job, keeping this module pure-data / cycle-free.
    """
    if data is None:
        return []
    block = data.get("capabilities")
    if not block:
        return []
    if not isinstance(block, list):
        raise CapabilitiesSugarError(
            f"{org}/org.yaml: capabilities: must be a list of mappings, "
            f"got {type(block).__name__}"
        )
    items: list[Any] = []
    for i, entry in enumerate(block):
        if not isinstance(entry, dict):
            raise CapabilitiesSugarError(
                f"{org}/org.yaml: capabilities[{i}] must be a mapping, "
                f"got {type(entry).__name__}"
            )
        kind = entry.get("kind")
        if kind == "mcp":
            ref = entry.get("ref")
            if not isinstance(ref, str) or not ref.strip():
                raise CapabilitiesSugarError(
                    f"{org}/org.yaml: capabilities[{i}] mcp ref must be a "
                    f"non-empty string"
                )
            allow = entry.get("allowlist", entry.get("tools"))
            if allow is None:
                items.append(ref.strip())
            else:
                if not isinstance(allow, list):
                    raise CapabilitiesSugarError(
                        f"{org}/org.yaml: capabilities[{i}] mcp allowlist "
                        f"must be a list, got {type(allow).__name__}"
                    )
                items.append({"ref": ref.strip(), "tools": [str(t) for t in allow]})
        elif kind in AGENT_CAPABILITY_KINDS:
            raise CapabilitiesSugarError(
                f"{org}/org.yaml: capabilities[{i}] kind={kind!r} is a per-agent "
                f"kind — declare it in agent frontmatter, not org.yaml"
            )
        else:
            raise CapabilitiesSugarError(
                f"{org}/org.yaml: capabilities[{i}] kind={kind!r} is not in the "
                f"sugar surface; org.yaml accepts {list(ORG_CAPABILITY_KINDS)} "
                f"(middleware/job stay in profile.yaml/policy.yaml)"
            )
    return items
