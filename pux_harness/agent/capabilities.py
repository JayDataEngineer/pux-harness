"""The capability facade — ONE resolution front-door for the model add-on
channels (tools / skills / MCP), the structural cleanup behind
``docs/capability-unification.md`` (CU-1).

The runtime is *already* unified: every callable — registry specialist, declared
``sandbox`` tool, dynamic ``lib/`` function, **and every MCP tool** — is a
``langchain_core.tools.BaseTool``, and skills are injected context. That
two-surface split (callable ≠ context) is locked by the
``skills-peek-via-read-file`` tripwire + the dynamic-tools "skills ≠ lib"
decision; this module does NOT touch it.

What this collapses is the declaration/resolution mess: ``build_stack`` used to
assemble the org-local tool + skill channels *inline*. Now it calls ONE
``CapabilityResolver.resolve(...)`` which dispatches by ``kind`` to the
**unchanged, contract-tested leaf resolvers** (``build_declared_tools`` /
``build_dynamic_tools`` / ``supervisor_skills_roots``) and records a unified
``Capability`` index row per channel — the audit surface (CU-2's ``pux
capabilities`` CLI) and the MDA export seam (CU-5) both read that index.

**CU-1 contract: zero behavior change.** ``build_stack`` composes the SAME
``tools_surface`` / ``supervisor_tools`` / ``supervisor_skills`` lists from
``ResolvedCapabilities`` that it used to build inline; the parity test
(``tests/harness/test_capability_resolver.py``) proves it bit-for-bit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import yaml
from langchain_core.tools import BaseTool

from pux_harness.agent import tool_servers as tool_servers_mod
from pux_harness.agent.orgs import _org_path, _orgs_dir, supervisor_skills_roots
from pux_harness.agent.profile import load_dynamic_tools_enabled
from pux_harness.kit._paths import project_root
from pux_harness.sandbox import policy as policy_mod
from pux_harness.sandbox.tools._shared import PUX_PREFIX
from pux_harness.sandbox.tools.declared import build_declared_tools, declared_tool_names
from pux_harness.sandbox.tools.dynamic import (
    PUX_DYN_PREFIX,
    build_dynamic_tools,
    load_dynamic_index,
)
from pux_harness.sandbox.tools.registry import REGISTRY

_log = logging.getLogger(__name__)

# The kind taxonomy — chosen to map 1:1 onto MDA's export fields
# (tool→tools/, skill→skills/, mcp→connectors/, middleware→middleware/,
# job→schedules/). ``kind`` is the model-facing surface.
Kind = Literal["tool", "skill", "mcp", "middleware", "job"]
# The (a)/(b)/(c) ownership ladder — ORTHOGONAL to ``kind``.
# registry=harness-authored, declared=operator-git-tracked (tools.yaml),
# dynamic=agent-authored (lib/), catalog=shared catalog (tool_servers.yaml),
# authored=unspecified/default. All three tool levels are ``kind=tool`` with a
# different ``provenance``; the levels compose UNDER ``kind``, not over it.
Provenance = Literal["registry", "declared", "dynamic", "catalog", "authored"]
ScopeLit = Literal["supervisor", "subagent", "org", "both"]


@dataclass(frozen=True)
class Capability:
    """ONE dispatch/audit row over the capability fleet.

    This is the **index/audit/export row** — it does NOT replace
    ``ToolSpec`` / ``ToolServerSpec`` / a skill root; it POINTS at one (via
    ``ref``) and routes resolution. ``kind`` and ``provenance`` are orthogonal
    axes: ``kind`` is the surface (callable tool vs context skill vs transport
    mcp vs loop-wrapping middleware vs schedule job); ``provenance`` is who
    authored it / where it lives. The dynamic-tools level model is composed
    under ``kind=tool`` (level (c) = ``kind=tool, provenance=dynamic``), not
    replaced.
    """

    kind: Kind
    ref: str  # tool name | skills-root path | mcp tool/server name | mw id | job id
    scope: ScopeLit = "both"
    provenance: Provenance = "authored"
    # MCP per-tool allowlist (None = take all). Reserved for the Layer-2
    # ``capabilities:`` sugar (CU-3); the leaf resolvers take all today.
    allowlist: tuple[str, ...] | None = None


@dataclass
class ResolvedCapabilities:
    """ONE org's resolved capability channels — what ``build_stack`` composes
    from.

    ``specialists`` + ``mcp`` are **fed in** (runtime-resolved OUTSIDE this
    layer: specialists need a model + backend built in ``graph.py``; mcp needs
    an async transport session). ``declared`` / ``dynamic`` / ``skill_roots``
    are org-local pure-data resolutions the resolver OWNS. Each channel
    dispatches to its unchanged leaf resolver, so the resolver adds no new
    runtime behavior — only one front-door.
    """

    specialists: list[BaseTool]  # kind=tool, provenance=registry (native specialists)
    declared: list[BaseTool]  # kind=tool, provenance=declared  (sandbox/tools.yaml)
    dynamic: list[BaseTool]  # kind=tool, provenance=dynamic  (lib/, opt-in)
    mcp: Sequence[BaseTool]  # kind=mcp                       (tool_servers)
    skill_roots: list[str]  # kind=skill   (SkillsMiddleware roots, container-absolute)
    # The unified audit/export index (one row per resolved channel member).
    # CU-2's CLI + CU-5's MDA exporter read this; runtime composition ignores it.
    capabilities: list[Capability] = field(default_factory=list)

    @property
    def tools_surface(self) -> list[BaseTool]:
        """The subagent-shared callable surface: specialists + declared +
        dynamic. NO mcp, NO retrieval — mcp is supervisor-only here, and the
        retrieval tools are appended per-scope by the context middleware
        (they are a side effect of middleware, not a capability channel).

        Order is LOAD-BEARING: a subagent's ``tools:`` allowlist resolves
        names against THIS list, so it must be byte-identical to the legacy
        ``[*specialists, *declared, *dynamic]`` composition.
        """
        return [*self.specialists, *self.declared, *self.dynamic]


def _index(
    specialists: Sequence[BaseTool],
    declared: Sequence[BaseTool],
    dynamic: Sequence[BaseTool],
    mcp_tools: Sequence[BaseTool],
    skill_roots: Sequence[str],
) -> list[Capability]:
    """Build the unified audit index from the resolved channels — one
    ``Capability`` row per member, tagged by ``kind`` + ``provenance``. Pure
    derivation from the runtime channels; never feeds back into resolution."""
    rows: list[Capability] = []
    for t in specialists:
        rows.append(Capability(kind="tool", ref=t.name, provenance="registry"))
    for t in declared:
        rows.append(Capability(kind="tool", ref=t.name, provenance="declared"))
    for t in dynamic:
        rows.append(Capability(kind="tool", ref=t.name, provenance="dynamic"))
    for t in mcp_tools:
        rows.append(Capability(kind="mcp", ref=t.name, provenance="catalog"))
    for root in skill_roots:
        rows.append(Capability(kind="skill", ref=root))
    return rows


class CapabilityResolver:
    """ONE front-door for resolving an org's model add-on channels.

    Stateless except for the ``exec_client`` the leaf tool builders
    (``build_declared_tools`` / ``build_dynamic_tools``) need to synthesize
    in-container-exec'ing ``StructuredTool``s. Dispatches by ``kind`` to the
    unchanged leaf resolvers; records the unified ``Capability`` index.

    This is the seam ``build_stack`` (runtime), the contract checker (CU-2),
    the ``capabilities:`` desugarer (CU-3), and the MDA exporter (CU-5) all
    call — "one way" to resolve an org's capabilities.
    """

    def __init__(self, exec_client: Any) -> None:
        self._exec_client = exec_client

    def resolve(
        self,
        org: str,
        *,
        specialists: Sequence[BaseTool] = (),
        mcp_tools: Sequence[BaseTool] = (),
    ) -> ResolvedCapabilities:
        declared = build_declared_tools(_org_path(org) / "sandbox", self._exec_client)
        # Dynamic (level c) tools — opt-in via ``sandbox.dynamic_tools: true``.
        # Byte-identical ([]) for orgs that do not opt in.
        lib_dir = _org_path(org) / "lib"
        if load_dynamic_tools_enabled(org):
            dynamic = build_dynamic_tools(lib_dir, self._exec_client)
        else:
            dynamic = []
            # No silent gap: an org may ship ``lib/*.py`` helper modules
            # (promoted dynamic tools, host-authored helpers) which are ONLY
            # callable through the ``pux_dyn_*`` surface — gated on the opt-in
            # flag. With it off those modules are INERT. Surface that rather
            # than let an operator wonder why a ``pux_dyn_call_function`` 404s.
            modules = sorted(lib_dir.glob("*.py")) if lib_dir.is_dir() else []
            if modules:
                _log.warning(
                    "%s: lib/ has %d Python module(s) but `sandbox.dynamic_tools` "
                    "is off — they are inert (not callable). Set "
                    "`sandbox.dynamic_tools: true` in policy.yaml to mount the "
                    "dynamic + promoted tooling surface.",
                    org, len(modules),
                )
        skill_roots = supervisor_skills_roots(org)
        specialists_ = list(specialists)
        return ResolvedCapabilities(
            specialists=specialists_,
            declared=declared,
            dynamic=dynamic,
            mcp=mcp_tools,
            skill_roots=skill_roots,
            capabilities=_index(
                specialists_, declared, dynamic, mcp_tools, skill_roots
            ),
        )


__all__ = [
    "Capability",
    "CapabilityIndex",
    "CapabilityResolver",
    "ResolvedCapabilities",
    "Kind",
    "Provenance",
]


# --- the discovery/audit catalog view (CU-2) -------------------------------
# ``CapabilityIndex`` is the fleet catalog — what ``pux capabilities list``
# reads. It MERGES the scattered sources into one in-memory ``Capability``
# list; the per-kind files stay the editable sources of truth (no mega-file).


class CapabilityIndex:
    """ONE discovery/audit view over the capability fleet — the CU-2 surface
    the ``pux capabilities`` CLI (and any fleet-wide audit) reads.

    MERGES the scattered catalogs/sources into one ``Capability`` list IN
    MEMORY; it does NOT create a new mega-file (the per-kind files —
    ``REGISTRY``, ``tool_servers.yaml``, ``skills.yaml`` — stay the editable
    sources of truth). The index is the *discovery/audit* view; the
    per-org ``CapabilityResolver`` is the *runtime resolution* view. Both
    speak the same ``Capability`` row + ``kind`` taxonomy.

    ``load(org=None)``:
    - Fleet-wide (``org is None``): the harness registries + shared catalogs —
      ``REGISTRY`` (tool/registry), ``MIDDLEWARE_REGISTRY`` (middleware),
      ``tool_servers.yaml`` (mcp/catalog), ``skills.yaml`` (skill/catalog).
    - Per-org (``org`` given): the fleet-wide rows PLUS the org's DECLARED
      channels — declared ``tools.yaml`` (tool/declared), authored ``lib/``
      functions (tool/dynamic, when opted in), the org's skill roots
      (skill/authored), and its ``jobs:`` schedule (job).

    Docker-free: every leaf is a spec/declaration loader (no exec_client, no
    transport session). Best-effort on the shared catalogs (a missing or
    malformed catalog yields no rows — the contract checker owns the
    structural error there); the org channels are read directly.

    ``ref`` naming matches the runtime resolver index (``_index``): declared
    tools are ``pux_sandbox_<name>`` (``PUX_PREFIX``), dynamic ``pux_dyn_<fn>``
    (``PUX_DYN_PREFIX``), so the catalog and the resolved surface agree.
    """

    @classmethod
    def load(cls, org: str | None = None) -> list[Capability]:
        rows: list[Capability] = []
        # --- fleet-wide: the harness registries + shared catalogs ---
        # tool/registry: the harness-authored specialist tools.
        for spec in REGISTRY:
            rows.append(Capability(kind="tool", ref=spec.slug, provenance="registry"))
        # middleware: the loop-wrapping middleware. MIDDLEWARE_REGISTRY lives
        # in stack.py, which imports THIS module (build_stack -> CapabilityResolver);
        # import it lazily inside the call to avoid the stack<->capabilities cycle.
        from pux_harness.agent.stack import MIDDLEWARE_REGISTRY  # noqa: PLC0415

        for spec in MIDDLEWARE_REGISTRY:
            rows.append(Capability(kind="middleware", ref=spec.name))
        # mcp/catalog: the shared tool_servers install catalog (.mcp.json-shaped).
        try:
            for name in tool_servers_mod.load_catalog():
                rows.append(Capability(kind="mcp", ref=name, provenance="catalog"))
        except (FileNotFoundError, ValueError):
            pass  # no catalog / malformed -> the contract owns the structural error
        # skill/catalog: the shared skills install catalog.
        rows.extend(cls._skill_catalog_rows())
        # --- per-org: the org's DECLARED channels ---
        if org is not None:
            rows.extend(cls._org_rows(org))
        return rows

    @staticmethod
    def _skill_catalog_rows() -> list[Capability]:
        """Names from the shared install catalog
        (``orgs/_shared/upstream_skills/skills.yaml``) — a flat
        ``{name: {repo, description, ...}}`` mapping. Best-effort."""
        path = _orgs_dir() / "_shared" / "upstream_skills" / "skills.yaml"
        if not path.is_file():
            return []
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            return []
        if not isinstance(data, dict):
            return []
        return [
            Capability(kind="skill", ref=name, provenance="catalog")
            for name in data
        ]

    @staticmethod
    def _org_rows(org: str) -> list[Capability]:
        """The org's declared channels — declared tools, authored lib/ functions,
        skill roots, and the jobs schedule. Spec/declaration level (docker-free)."""
        rows: list[Capability] = []
        org_path = _org_path(org)
        # declared sandbox tools (tools.yaml) -> tool/declared (prefixed, matches
        # the resolver's runtime index).
        for name in declared_tool_names(org_path / "sandbox"):
            rows.append(
                Capability(kind="tool", ref=PUX_PREFIX + name, provenance="declared")
            )
        # authored lib/ functions -> tool/dynamic (only when the org opts in).
        if load_dynamic_tools_enabled(org):
            for name in load_dynamic_index(org_path / "lib"):
                rows.append(
                    Capability(
                        kind="tool", ref=PUX_DYN_PREFIX + name, provenance="dynamic"
                    )
                )
        # the org's skill roots -> skill/authored (the SkillsMiddleware roots).
        for root in supervisor_skills_roots(org):
            rows.append(Capability(kind="skill", ref=root))
        # the org's jobs schedule -> job.
        try:
            pol = policy_mod.load(org, project_root())
        except policy_mod.NoPolicy:
            pol = None
        except Exception:
            # a malformed policy is the contract checker's job, not the index's.
            pol = None
        for spec in policy_mod.job_specs(pol):
            rows.append(Capability(kind="job", ref=spec.name or "<unnamed>"))
        return rows
