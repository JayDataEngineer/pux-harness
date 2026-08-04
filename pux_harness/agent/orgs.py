"""Org + specialist-agent loading for the deepagents harness.

System prompt = the chain-inherited org overlay (base org ``general`` + the
org's own, via ``extends:``) + harness addendum.
Each org is a self-contained bundle: ``orgs/<name>/agents/<slug>.md`` is ONE
file — YAML frontmatter (``name``/``description`` + optional ``tools``/
``skills``/``model``) + a markdown body that IS the system prompt (mirrors the
``SKILL.md`` convention). The org roster is ``orgs/<name>/org.yaml``
(``agents: [slug, ...]``); ``AGENTS.md`` is pure CTO-prompt prose (no
frontmatter). Cross-org agents (e.g. ``researcher``) live in
``orgs/_shared/agents/``; resolution is **org-local first, then _shared**, so
an org can specialize a shared agent by dropping a same-named ``<slug>.md`` in
its own ``agents/`` dir.

The harness addendum pins the deepagents delegation surface (the ``task``
tool with ``subagent_type=<name>``), bridging any legacy wording in inherited
org docs to the live tool.

Agent definitions are pure data (frontmatter + prose) — no executable module,
no ``importlib``. Tool + skills resolution stays CENTRAL (here, via
``_resolve_tools`` / ``_resolve_skills``) so the contract checker
(``--check-contract``, offline) reads the same files the runtime does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Container, Sequence, TYPE_CHECKING

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool
from deepagents import HarnessProfileConfig

from pux_harness.agent.model import get_model
from pux_harness.agent.prompt_parts import (
    SUBAGENT_PROMPT_PARTS,
    PromptCtx,
    PromptPartSpec,
    PromptScope,
    assemble_prompt,
)
from pux_harness.sandbox.tools import Category, classify_slug, prefixed

if TYPE_CHECKING:
    # Type-only: the GP subagent profile shape is referenced only in a string
    # annotation (``from __future__ import annotations``), so it stays lazy.
    from deepagents import GeneralPurposeSubagentProfile

# The PURE org/agent loaders live in ``pux_harness.kit`` (the slim, Docker-free
# core — the ONE source of truth for how an org dir / agent <slug>.md /
# SKILL.md frontmatter doc is parsed). The path helpers, org discovery, roster,
# org-overlay, agent-spec, and skills functions BELOW are THIN DELEGATES: each
# forwards to the matching ``_aloaders`` function, resolving ``project_root``
# from the injectable ``_orgs_dir()`` seam (``project_root = _orgs_dir().parent``).
# The kit's ``project_root`` parameter is NOT incompatible with the contract
# tests' monkeypatch of ``orgs._orgs_dir`` — it IS the seam's value, threaded
# through. The contract tripwire ``no-duplicate-loaders-in-orgs`` locks these as
# delegates so the verbatim duplication can't silently return.
#
# Harness-local (genuinely harness-specific, NOT delegated): the ``_orgs_dir()``
# / ``_specialists_dir()`` seams; ``build_system_prompt`` (the static base of the
# supervisor prompt — the chain-inherited org overlay only; the base org
# ``general`` IS the base prompt, flowing to specialists via ``extends:``; the
# addendum + every suffix is folded on top by ``prompt_parts.assemble_prompt``);
# ``_resolve_tools`` (strict registry); ``_build_sub`` + ``load_subagents``
# (profile/middleware/retrieval enrichment).
from pux_harness.kit import loaders as _aloaders
from pux_harness.kit._paths import project_root

# Re-exported for the contract surface: ``contract.py`` imports ``_parse_list``
# / ``_split_frontmatter`` from THIS module (and the contract tests reach them
# at ``contract._parse_list`` etc.). They are the SAME callables the delegates
# below route through ``_aloaders`` — one parser upstream, two reachable names.
_parse_list = _aloaders._parse_list
_split_frontmatter = _aloaders._split_frontmatter

# Container bind-mount target (container.py: ``<project>:/sandbox/workspace``).
# Skills sources are mapped to container-absolute paths under this root for
# deepagents' SkillsMiddleware (which resolves them against the backend).
_WORKSPACE_ROOT = "/sandbox/workspace"


# --- path helpers (injectable — tests monkeypatch these) -------------------
# Single source of truth for where orgs/agents live. ``contract.py``
# re-exports these; the contract tests monkeypatch them at both module sites.

def _orgs_dir() -> Path:
    return project_root() / "orgs"


def _specialists_dir() -> Path:
    return _orgs_dir() / "specialists"


def _org_path(name: str) -> Path:
    """Resolve an org's directory — checks top-level ``orgs/`` first, then
    ``orgs/specialists/``. Raises ``FileNotFoundError`` if neither exists.

    Thin delegate to ``pux_harness.kit.loaders`` (single source of truth); the
    project root comes from the injectable ``_orgs_dir()`` seam so the contract
    tests' monkeypatch reaches it unchanged."""
    return _aloaders._org_path(name, _orgs_dir().parent)


def org_extends(name: str) -> str | None:
    """This org's single-hop ``extends:`` parent (raw read of ``org.yaml``), or
    ``None``. Org inheritance. Thin delegate to ``pux_harness.kit.loaders``."""
    return _aloaders.org_extends(name, _orgs_dir().parent)


def org_extends_chain(name: str) -> list[str]:
    """The org's inheritance chain, root→child. RAISES on a cycle / unresolvable
    parent. Org inheritance. Thin delegate to ``pux_harness.kit.loaders``."""
    return _aloaders.org_extends_chain(name, _orgs_dir().parent)


def _agent_search_dirs(org: str) -> list[Path]:
    """Directories searched for an agent ``<slug>.md``, org-local first then
    shared. Single source of truth — ``contract.py`` re-exports / monkeypatches
    this at its call sites. An org specializes a shared agent by placing a
    same-named ``<slug>.md`` in its own ``agents/`` dir (first hit wins).

    Thin delegate to ``pux_harness.kit.loaders``; project root from ``_orgs_dir()``."""
    return _aloaders._agent_search_dirs(org, _orgs_dir().parent)


def discover_orgs() -> list[str]:
    """Sorted names of every org dir containing ``AGENTS.md``. Data-driven —
    no hardcoded manifest. An org's specialist roster lives in its
    ``org.yaml``. ``_shared`` and other bundles without an
    AGENTS.md are excluded by the presence rule.

    Thin delegate to ``pux_harness.kit.loaders``; project root from ``_orgs_dir()``."""
    return _aloaders.discover_orgs(_orgs_dir().parent)


def org_agent_slugs(name: str) -> list[str]:
    """The specialist slugs this org delegates to, read from
    ``orgs/<name>/org.yaml``.

    Thin delegate to ``pux_harness.kit.loaders``; project root from ``_orgs_dir()``."""
    return _aloaders.org_agent_slugs(name, _orgs_dir().parent)


def load_org_prompt(name: str) -> str:
    """Body of orgs/<name>/AGENTS.md (the per-org CTO overlay).

    Thin delegate to ``pux_harness.kit.loaders``; project root from ``_orgs_dir()``."""
    return _aloaders.load_org_prompt(name, _orgs_dir().parent)


def build_system_prompt(org: str) -> str:
    """The chain-inherited org overlay — the STATIC base of the supervisor
    prompt. The base org (``general``) is the root of the ``extends:`` chain; its
    overlay IS the base prompt, and a specialist that extends it layers its own
    overlay on top (additive — base first, specialization after). The harness
    addendum + every conditional suffix is folded on top by
    ``prompt_parts.assemble_prompt``; this returns ONLY the chain overlay (no
    addendum — the harness addendum is folded by the registry), so the no-gap
    registry owns the full assembly. Routes through the kit's canonical
    ``build_system_prompt`` (cycle-aware; falls back to ``[org]`` on a broken
    chain) — ONE entrypoint shared by harness + standalone consumers."""
    return _aloaders.build_system_prompt(org, project_root=_orgs_dir().parent)


def harness_addendum_text() -> str:
    """The supervisor addendum from ``orgs/_shared/harness_addendum.md`` (body +
    leading ``"\\n"`` seam, ready for ``ctx.harness_addendum``). CWD-relative
    like ``build_system_prompt``. Falls back to the embedded ``_ADDENDUM``
    constant when the file is absent (minimal fixtures / packed archives). This
    is the runtime-path reader; the introspection view (``prompt_show``) calls
    ``load_harness_addendum`` directly with an explicit ``project_root``."""
    from pux_harness.agent.prompt_parts import load_harness_addendum

    return load_harness_addendum(_orgs_dir().parent)


def ask_user_suffix_text() -> str:
    """The ask-user turn-ending suffix from ``orgs/_shared/ask_user_suffix.md``
    (ready for ``ctx.ask_user_text``). CWD-relative like
    ``build_system_prompt``. Falls back to the embedded
    ``ASK_USER_PROMPT_SUFFIX`` constant when the file is absent. This is the
    runtime-path reader; the introspection view calls
    ``load_ask_user_suffix`` directly with an explicit ``project_root``."""
    from pux_harness.agent.prompt_parts import load_ask_user_suffix

    return load_ask_user_suffix(_orgs_dir().parent)


def dynamic_dispatch_suffix_text() -> str:
    """The eval-tool dispatch strategy from
    ``orgs/_shared/dynamic_dispatch_suffix.md`` (ready for
    ``ctx.dynamic_dispatch_text``). CWD-relative like
    ``build_system_prompt``. Falls back to the embedded
    ``_DYNAMIC_DISPATCH_SUFFIX`` constant when the file is absent. This is the
    runtime-path reader; the introspection view calls
    ``load_dynamic_dispatch_suffix`` directly with an explicit
    ``project_root``."""
    from pux_harness.agent.prompt_parts import load_dynamic_dispatch_suffix

    return load_dynamic_dispatch_suffix(_orgs_dir().parent)


def _load_extra_parts_for_scope(
    org: str, scope: "PromptScope", project_root: "Path | None" = None,
) -> tuple["PromptPartSpec", ...]:
    """Read ``extra_prompt_parts:`` from the extends-chain and build always-on
    ``PromptPartSpec`` instances filtered to ``scope``. Walks the chain
    root→child; the CHILD-MOST org that declares ``extra_prompt_parts`` wins
    (lists are delta-wins in the deep-merge). File paths resolve relative to the
    DECLARING org's directory (so a parent's extras inherit with their own
    relative paths intact). Returns ``()`` when no org in the chain declares
    extras (the common case — opt-in). ``project_root`` defaults to the
    CWD-resolved orgs tree (runtime path); the introspection view passes it
    explicitly."""
    import yaml

    from pux_harness.kit._paths import search_org_dir
    from pux_harness.kit.loaders import _resolved_org_chain
    from pux_harness.agent.prompt_parts import build_extra_parts

    if project_root is None:
        project_root = _orgs_dir().parent
    entries = None
    declaring_dir = None
    for ancestor in _resolved_org_chain(org, project_root):  # root→child
        try:
            org_dir = search_org_dir(ancestor, project_root)
        except FileNotFoundError:
            continue
        profile_path = org_dir / "profile.yaml"
        if not profile_path.is_file():
            continue
        data = yaml.safe_load(profile_path.read_text())
        if data and data.get("extra_prompt_parts"):
            entries = data["extra_prompt_parts"]
            declaring_dir = org_dir
    if not entries or declaring_dir is None:
        return ()
    return build_extra_parts(entries, org, declaring_dir, scope)


def _resolve_tools(raw: Any, tool_map: dict[str, BaseTool]) -> list[BaseTool]:
    """Map an agent's ``tools`` list to specialist StructuredTools.

    Classification is shared with the offline contract (``classify_slug`` from
    the tool ``REGISTRY``). A native slug (``execute``/``read_file``/…) is
    SKIPPED — the backend's ``FilesystemMiddleware`` injects the fs/shell tools
    into every subagent regardless of its whitelist, so they have no entry in
    the specialist ``tool_map`` (previously the runtime would have raised
    KeyError on one while the contract accepted it). Any other slug resolves iff
    its ``pux_sandbox_*`` key is present in ``tool_map`` — that admits BOTH
    REGISTRY specialists AND org-declared sandbox tools (``sandbox/tools/tools.yaml``),
    which share the prefix; a stale/unknown reference still fails loud. Sharing
    the classifier with the contract means the runtime and offline paths can no
    longer disagree.
    """
    resolved: list[BaseTool] = []
    for entry in _aloaders._parse_list(raw):
        slug = entry.rsplit("/", 1)[-1]
        kind = classify_slug(slug)
        if kind is Category.NATIVE:
            continue
        key = prefixed(slug, Category.SPECIALIST)
        # Resolve iff the tool is actually present in the map. REGISTRY
        # specialists and declared sandbox tools BOTH key under ``pux_sandbox_*``
        # (declared tools share the specialist prefix), so a single membership
        # check admits both; a stale/unknown reference still fails loud.
        if key not in tool_map:
            raise KeyError(
                f"agent references unknown tool {entry!r} "
                f"(resolved {key!r}, not in the tool map)"
            )
        resolved.append(tool_map[key])
    return resolved


def _org_declared_mcp_servers(org: str) -> frozenset[str]:
    """The MCP server names this org DECLARES in its own ``org.yaml``
    ``capabilities: [{kind: mcp, ref: <server>}]`` block — the two-level
    grant gate's level-1 set (``_resolve_mcp``'s ``declared_servers``).

    This is the SAME source ``resolve_tool_servers`` arms from
    (``_org_yaml_mcp_items``), so the agent-level gate and the org-level arming
    agree by construction: an agent may route a subset of EXACTLY the servers
    the org declared — never a server the org didn't. Lazy import
    (``tool_servers`` imports ``orgs`` at module load -> a top-level import here
    would cycle). Offline-safe: a pure ``org.yaml`` read with no server
    reachability, so the declared set is stable whether or not any server is
    up. Returns ``frozenset()`` for an org that ships no mcp sugar (no
    ``org.yaml`` or no ``capabilities:`` block).
    """
    # Lazy: ``tool_servers`` line 29 does ``from .orgs import _org_path,
    # _orgs_dir`` at module load -> importing it at the top of THIS module
    # cycles. Importing inside the function breaks the cycle cleanly.
    from pux_harness.agent.tool_servers import _org_yaml_mcp_items
    names: set[str] = set()
    for item in _org_yaml_mcp_items(org):
        if isinstance(item, str):
            ref = item.strip()
        elif isinstance(item, dict):
            ref = str(item.get("ref", "")).strip()
        else:
            ref = ""
        if ref:
            names.add(ref)
    return frozenset(names)


def _resolve_mcp(
    raw_mcp: Any, mcp_tools: Sequence[BaseTool], slug: str,
    *, declared_servers: Container[str] = frozenset(),
) -> list[BaseTool]:
    """Map an agent's desugared ``mcp:`` list to the org's ARMED MCP tools.

    The focused-MCP path: ``kind: mcp`` in agent frontmatter (CU-3) desugars to a
    ``mcp:`` list (see ``kit.capabilities_decl.desugar_agent_capabilities``) of
    bare catalog-ref strings (take EVERY tool from that server) or
    ``{ref, tools: [...]}`` mappings (narrow to the named bare tools). This
    resolves that list against ``mcp_tools`` — the org's already-armed,
    namespaced MCP tools (``caps.mcp`` from the factory) — and returns the
    focused subset to splice into the subagent's ``tools`` whitelist.

    Matching is by the namespacing PREFIX ``mcp__<ref>__*`` — the ``<ref>`` is the
    catalog key, which is the ``<server>`` segment ``mcp_client._namespace_tools``
    stamps (``mcp__{server_name}__{tool}``). The trailing ``__`` makes it an
    EXACT-server match (``equibles`` never matches ``equibles-extra``), and it is
    robust to a bare tool name that itself contains ``__``. The bare tool name is
    the segment AFTER the prefix — the SAME pre-namespace name
    ``mcp_client._apply_allowlist`` filters on, so the agent's allowlist uses the
    names you'd see in ``tools/list``.

    LENIENT TWO-LEVEL GRANT GATE — ``declared_servers`` is the level-1 set (the
    MCP server names the org DECLARES in its own ``org.yaml``
    ``capabilities: [{kind: mcp, ref: <server>}]``, the SAME source
    ``resolve_tool_servers`` arms from — see ``_org_declared_mcp_servers``).
    An agent ``ref`` NOT in ``declared_servers`` is a CONFIG ERROR and fails
    loud (naming the agent, the ref, and the declared set), so a typo'd ref can
    never silently inherit the supervisor's whole MCP surface: the org MUST
    declare it first. But a ref that IS declared yet resolves to no armed tools
    this run (the server unreachable/offline, ``mcp_tools`` empty) contributes
    ZERO tools and the org still builds — mirrors the org layer's OWN leniency
    (``resolve_tool_servers`` tolerates an unreachable declared server). So an
    org builds offline with its mcp agents carrying an empty mcp subset; a real
    misconfig still fails loud at build time. A bare-name allowlist the server
    doesn't expose fails loud too (mirrors ``mcp_client._apply_allowlist``) —
    but only when the server is actually armed (unreachable -> skipped). Empty
    ``declared_servers`` (the default) treats EVERY ref as undeclared and fails
    loud — the safe misuse-guard for a caller that forgot to pass the real set;
    ``load_subagents`` always passes the org's declared set. ``raw_mcp``
    empty/None -> ``[]`` (the agent declared no mcp; its surface is whatever
    ``_resolve_tools`` produced, or deepagents inheritance when that's also
    empty)."""
    if not raw_mcp:
        return []
    if not isinstance(raw_mcp, list):  # the desugarer always emits a list
        raise KeyError(
            f"agent {slug!r}: mcp frontmatter must be a list, "
            f"got {type(raw_mcp).__name__}"
        )
    resolved: list[BaseTool] = []
    for entry in raw_mcp:
        if isinstance(entry, str):
            ref: str = entry.strip()
            allowlist: Any = None
        elif isinstance(entry, dict):
            ref = str(entry.get("ref", "")).strip()
            allowlist = entry.get("tools")
            if allowlist is not None and not isinstance(allowlist, list):
                raise KeyError(
                    f"agent {slug!r}: mcp {ref!r} allowlist must be a list, "
                    f"got {type(allowlist).__name__}"
                )
        else:
            raise KeyError(
                f"agent {slug!r}: mcp entry must be a ref string or a "
                f"{{ref, tools}} mapping, got {type(entry).__name__}"
            )
        if not ref:
            raise KeyError(f"agent {slug!r}: mcp ref must be a non-empty string")
        prefix = f"mcp__{ref}__"
        server_tools = [t for t in mcp_tools if t.name.startswith(prefix)]
        # Lenient two-level gate (see docstring). Level 1 = declared-by-org:
        # a ref the org never declared is a CONFIG ERROR -> fail loud. Level 2
        # = armed-this-run: a declared ref with zero armed tools (server
        # unreachable/offline) is lenient -> contributes nothing, build lives.
        if ref not in declared_servers:
            raise KeyError(
                f"agent {slug!r}: mcp ref {ref!r} is not declared by this org "
                f"(declared: {sorted(declared_servers) or '<none>'}). Declare "
                f"it in org.yaml capabilities: [{{kind: mcp, ref: {ref}}}] "
                f"first, then route it to this agent."
            )
        if not server_tools:
            continue  # declared but not armed this run -> lenient empty subset
        if allowlist is None:
            resolved.extend(server_tools)
        else:
            by_bare = {t.name[len(prefix):]: t for t in server_tools}
            missing = [str(n) for n in allowlist if str(n) not in by_bare]
            if missing:
                raise KeyError(
                    f"agent {slug!r}: mcp ref {ref!r} allowlist names {missing} "
                    f"not exposed by server (exposed: {sorted(by_bare)})"
                )
            resolved.extend(by_bare[str(n)] for n in allowlist)
    return resolved


def _resolve_skills(raw: Any, slug: str) -> list[str]:
    """``skills`` value -> container-absolute skills-ROOT paths.

    deepagents' ``SkillsMiddleware`` resolves each source against the BACKEND
    (the sandbox container) and loads EVERY ``<source>/<skill>/SKILL.md``
    beneath it — a source is a skills **root** directory, not an individual
    skill (passing an individual skill dir loads nothing: its only child is the
    SKILL.md *file*). So a value is a **project-relative** directory (e.g.
    ``orgs/_shared/skills`` or ``orgs/<org>/skills``); we validate it exists on
    the host (the project is bind-mounted 1:1 at ``/sandbox/workspace``, so
    host existence == container existence) and map it to a container-absolute
    path for deepagents.

    Thin delegate to ``pux_harness.kit.loaders``; ``project_root`` from
    ``_orgs_dir()`` and ``workspace_root`` pinned to the harness
    ``_WORKSPACE_ROOT`` (``/sandbox/workspace``) so the kit returns
    container-absolute paths, not its default local-absolute ones. E2E-proven:
    ``backend.ls('/sandbox/workspace/orgs/_shared/skills')`` lists
    ``source-citation``; the middleware then reads its ``SKILL.md``.
    """
    return _aloaders._resolve_skills(
        raw, slug, project_root=_orgs_dir().parent, workspace_root=_WORKSPACE_ROOT,
    )


def supervisor_skills_roots(org: str) -> list[str]:
    """Container-absolute skills-ROOT paths for the SUPERVISOR's
    ``SkillsMiddleware`` — the focused set (``orgs/_shared/skills`` + this org's
    own ``skills/``), existing dirs only, mapped to ``/sandbox/workspace/...``.

    Native progressive disclosure on the CTO: the middleware injects
    each root's skill metadata (name + description) into the supervisor prompt;
    the agent peeks a body via the native ``read_file`` (the canonical path —
    ``pux_sandbox_load_skill`` is gone). ``[]`` for a no-skills org -> the graph
    binds ``skills=None`` -> byte-identical to today (no SkillsMiddleware).

    Thin delegate to ``pux_harness.kit.loaders``; project root from
    ``_orgs_dir()``, workspace root pinned to the harness ``_WORKSPACE_ROOT``."""
    return _aloaders.supervisor_skills_roots(
        org, _orgs_dir().parent, _WORKSPACE_ROOT,
    )


def _load_agent_spec(slug: str, org: str) -> dict[str, Any] | None:
    """Read ``<slug>.md`` from the org-local then ``_shared`` agent dir and
    return a spec dict (``name``/``description`` + optional ``tools``/
    ``skills``/``model`` from frontmatter; ``system_prompt`` = the body).

    Returns ``None`` if no ``<slug>.md`` exists in either search dir — the
    caller (``load_subagents`` / the contract checker) raises. There is NO
    legacy ``.py`` fallback; the ``no-legacy-agent-py`` contract tripwire
    guarantees every roster slug resolves to a frontmatter ``.md``.

    Thin delegate to ``pux_harness.kit.loaders``; project root from ``_orgs_dir()``."""
    return _aloaders._load_agent_spec(slug, org, _orgs_dir().parent)


def _build_sub(
    slug: str, spec: dict[str, Any], tool_map: dict[str, BaseTool], system_prompt: str,
    org: str, *, middleware: list[AgentMiddleware],
    mcp_tools: Sequence[BaseTool] = (),
    declared_servers: Container[str] = frozenset(),
) -> dict[str, Any]:
    """Build a deepagents SubAgent dict from a spec mapping (the module's
    ``SUBAGENT`` dict). ``system_prompt`` is passed in explicitly.

    Omitted ``tools`` -> inherit the main agent's tools. Model resolution
    a frontmatter ``model:`` is the agent-level override (a
    literal id); otherwise the subagent runs on the ``worker`` role — resolved
    through models.yaml + the org profile + env, decoupled from the base/CTO
    model so an org can set base!=worker. (Worker defaults to the same id as
    base, so today's orgs are byte-identical; the seam is what's new.)

    ``middleware`` is the FULLY RESOLVED subagent middleware list
    handed down by the stack factory (``stack.build_stack``): the always-on
    context layer (capture + offload) PLUS any toggleable middleware the org
    added to the subagent scope via ``profile.yaml``'s ``middleware.subagent``
    block. ``SubAgentMiddleware`` forwards a spec's ``middleware`` key into the
    compiled subagent (verified against deepagents 0.6.12: ``graph.py:656`` +
    ``middleware/subagents.py:494`` both ``.extend(spec.get("middleware", []))``
    into ``create_agent``), so this makes every entry intercept THIS subagent's
    own tool calls. The retrieval tools (``ctx_recall``/``ctx_search``) are
    appended to the whitelist separately in ``load_subagents`` (after profile
    filtering) — see there.
    """
    sub: dict[str, Any] = {
        "name": spec.get("name", slug),
        "description": spec.get("description", slug),
        "system_prompt": system_prompt,
    }
    # Focused whitelist: ANY declared ``tools`` OR ``mcp`` flips the subagent out
    # of deepagents' "inherit the main agent's tools" default (omitted ``tools``
    # -> inherit). Composing the mcp subset (``_resolve_mcp``) into the SAME
    # ``tools`` key is what makes a focused mcp-only specialist possible — it
    # gets EXACTLY its declared servers (plus any ``tools``), not the inherited
    # kitchen sink. ``_resolve_tools([])`` is ``[]`` so an mcp-only decl still
    # works; the gate is ``tools or mcp`` so a present ``mcp`` alone flips it.
    raw_mcp = spec.get("mcp")
    if spec.get("tools") or raw_mcp:
        sub["tools"] = (
            _resolve_tools(spec.get("tools") or [], tool_map)
            + _resolve_mcp(raw_mcp, mcp_tools, slug, declared_servers=declared_servers)
        )
    if spec.get("model"):
        sub["model"] = get_model(model=spec["model"])
    else:
        sub["model"] = get_model(role="worker", org=org)
    if "skills" in spec:
        sub["skills"] = _resolve_skills(spec["skills"], slug)
    sub["middleware"] = list(middleware)
    return sub


# --- per-agent frontmatter overrides ---------------------------------------
#
# The universal override vocabulary: the SAME ``HarnessProfileConfig`` fields
# that work ORG-WIDE via ``profile.yaml`` work PER-AGENT via the agent's OWN
# frontmatter. This folded the legacy ``profile.yaml`` ``subagents:`` block (a
# SECOND partial-override surface with its own resolver) into the one
# frontmatter path — the ``extends:`` merge in ``kit.loaders._merge_extends``
# inherits + overrides these the same way it inherits tools/skills, so there is
# no second surface. The ``no-legacy-subagents-block`` contract tripwire keeps
# the old block from returning.
_AGENT_PROFILE_KEYS: tuple[str, ...] = (
    "system_prompt_suffix",
    "tool_description_overrides", "excluded_tools",
)


def _agent_profile_from_spec(spec: dict[str, Any]) -> HarnessProfileConfig | None:
    """Build a per-agent ``HarnessProfileConfig`` from an agent spec's OWN
    frontmatter override fields. ``None`` when the spec carries
    none of the four fields — the common case, byte-identical (no per-agent
    rewriting). The spec is already ``extends:``-merged, so a child agent
    inherits + overrides these via the merge the same way it inherits tools."""
    present = {k: spec[k] for k in _AGENT_PROFILE_KEYS if k in spec}
    if not present:
        return None
    return HarnessProfileConfig.from_dict(present)


# --- the general-purpose subagent (own the GP) -----------------------------
#
# deepagents auto-adds a HEAVY default ``general-purpose`` subagent to EVERY
# graph (deepagents/graph.py:716-717) unless ``gp_profile.enabled is False`` OR
# a spec named ``general-purpose`` is already in the inline subagents. pux
# passes no GP kwarg to ``create_deep_agent`` and stays off the model-keyed
# ``_HARNESS_PROFILES`` registry (two orgs on one model would merge-collide; the
# long-lived server builds many orgs per process; and there is no
# ``unregister``). So without an explicit spec the auto-add fires for EVERY pux
# org — including coder, the Claude-Code-equivalent coding org whose
# ``roster_deny: [general-purpose, ...]`` declaration (checked by the
# ``roster-deny-enforced`` contract rule) so NEVER sees the auto-added slot.
#
# The fix honors the NATIVE field (no parallel grammar): when an org's
# ``profile.yaml`` carries a ``general_purpose_subagent:`` block — surfaced
# straight through ``HarnessProfileConfig.from_dict`` — pux emits its OWN
# ``name="general-purpose"`` spec, and deepagents then skips the auto-add (the
# name already exists). Three cases:
#   * no block (``cfg.general_purpose_subagent is None``) -> pux emits NOTHING;
#     deepagents auto-adds its default (byte-identical to today — the parity path).
#   * ``enabled: false`` -> a NEUTERED spec (empty tools + an honest disabled
#     description/prompt). Occupies the task-menu slot but is dead — even a
#     stray delegation returns immediately. (Safeguard S1: full removal would
#     need the registry pux can't safely use; the neuter is strictly better than
#     the heavy auto-add leak.)
#   * ``enabled`` None/True (+ optional description/system_prompt) -> a
#     CUSTOMIZED spec on the full specialist + retrieval surface, with the org's
#     profile overrides applied the same way every roster subagent's are.

_GENERAL_PURPOSE_NAME = "general-purpose"

# The default + disabled GP description/prompt live in ``orgs/_shared/
# general_purpose.md`` — loaded once, cached. The file is the single source
# (no English constants in Python); the harness reads it when an org enables
# the GP subagent without supplying its own text. See
# ``_load_general_purpose_text``.
_GP_TEXT: dict[str, str] | None = None


def _load_general_purpose_text() -> dict[str, str]:
    """Load the 4 GP text fields from ``orgs/_shared/general_purpose.md``.

    Returns ``{default_description, default_prompt, disabled_description,
    disabled_prompt}``. Cached module-level (``_GP_TEXT``); the file is read
    once per process. Fails LOUD if the file is absent — the file is the
    single source of truth (no embedded fallback constant to drift from)."""
    global _GP_TEXT
    if _GP_TEXT is not None:
        return _GP_TEXT
    import yaml as _yaml
    from pux_harness.kit.loaders import load_shared_prompt_body

    # Load the frontmatter (the 4 fields) — body is documentation, unused here.
    path = _orgs_dir().parent / "_shared" / "general_purpose.md"
    raw = path.read_text(encoding="utf-8")
    # frontmatter split (same convention as every other _shared/*.md)
    if raw.startswith("---"):
        _, fm, _body = raw.split("---", 2)
        data = _yaml.safe_load(fm) or {}
    else:
        data = {}
    _GP_TEXT = {
        "default_description": data.get("default_description", ""),
        "default_prompt": data.get("default_prompt", ""),
        "disabled_description": data.get("disabled_description", ""),
        "disabled_prompt": data.get("disabled_prompt", ""),
    }
    # Fail loud if any field is empty — the file is the single source.
    missing = [k for k, v in _GP_TEXT.items() if not v]
    if missing:
        raise ValueError(
            f"orgs/_shared/general_purpose.md: missing frontmatter field(s) "
            f"{missing}; the file must supply all 4 GP text variants"
        )
    return _GP_TEXT


def _build_general_purpose_sub(
    gp: "GeneralPurposeSubagentProfile",
    org: str,
    *,
    tool_surface: list[BaseTool],
    middleware: list[AgentMiddleware],
    profile: "HarnessProfileConfig | None",
) -> dict[str, Any]:
    """Build the ``general-purpose`` subagent spec from a profile's GP config.

    Emitted by the factory (``stack.build_stack``) ONLY when the org's
    ``profile.yaml`` carries a ``general_purpose_subagent:`` block — so the
    no-block case stays byte-identical to deepagents' auto-add. See the module
    note above for the three-case shape (disabled / customized) + the
    no-registry rationale.

    ``tool_surface`` is the general pux surface (specialists + retrieval); it is
    profile-filtered here (``excluded_tools`` / ``tool_description_overrides``)
    the same way every roster subagent's whitelist is (``apply_profile_to_tools``
    — imported lazily because ``profile.py`` imports ``_orgs_dir`` from THIS
    module → a cycle), so an org-wide override reaches the GP too. The org-wide
    ``system_prompt_suffix`` layers on top of the GP prompt (custom or default),
    matching the precedence every other subagent follows in ``load_subagents``.
    Both applications route through the ``profile_apply`` seam (single owner of
    the three upstream-gap applications)."""
    # Lazy import: profile.py imports ``_orgs_dir`` from THIS module → cycle.
    from pux_harness.agent.profile_apply import (
        apply_profile_to_prompt,
        apply_profile_to_tools,
    )

    model = get_model(role="worker", org=org)
    gp_text = _load_general_purpose_text()
    if gp.enabled is False:
        return {
            "name": _GENERAL_PURPOSE_NAME,
            "description": gp_text["disabled_description"],
            "system_prompt": gp_text["disabled_prompt"],
            "tools": [],
            "model": model,
            "middleware": [],
        }
    description = gp.description or gp_text["default_description"]
    prompt = apply_profile_to_prompt(
        gp.system_prompt or gp_text["default_prompt"], profile
    )
    tools = list(tool_surface)
    if profile is not None:
        tools = apply_profile_to_tools(tools, profile)
    return {
        "name": _GENERAL_PURPOSE_NAME,
        "description": description,
        "system_prompt": prompt,
        "tools": tools,
        "model": model,
        "middleware": list(middleware),
    }


def load_subagents(
    org: str,
    all_tools: list[BaseTool],
    profile: Any = None,
    *,
    subagent_middleware: list[AgentMiddleware],
    retrieval_tools: list[BaseTool],
    mcp_tools: Sequence[BaseTool] = (),
    build_subagent_middleware: Callable[..., list[AgentMiddleware]] | None = None,
) -> list[dict[str, Any]]:
    """Build deepagents SubAgent dicts for ``org``'s specialists.

    For each slug in ``org.yaml``, load ``orgs/<org>/agents/<slug>.md`` (or
    ``orgs/_shared/agents/<slug>.md``) and build from its frontmatter + body.

    ONE WAY — the stack factory (``stack.build_stack``) owns the WHOLE agent
    tree's middleware + retrieval surface and hands the resolved SUBAGENT slice
    down here. Both are required: every caller (the factory at runtime, every
    direct test/contract call) builds the context layer itself and passes it.
    The loader never builds the layer — it only threads what it is given:

    - ``subagent_middleware``: forwarded into each spec via ``_build_sub`` →
      ``sub["middleware"]`` (the always-on context layer + any toggleable
      middleware the org added to the subagent scope).
    - ``retrieval_tools``: ``ctx_recall``/``ctx_search``, appended to each
      specialist's resolved whitelist AFTER profile filtering (so an org's
      ``excluded_tools`` can never strip retrieval).
    - ``build_subagent_middleware``: a per-subagent resolver (built by
      ``stack.make_subagent_middleware_builder``) that, when a subagent's
      frontmatter carries ``middleware: [name, ...]``, resolves the org's
      subagent baseline PLUS those per-agent names into a fresh, validated,
      registry-ordered list for THAT subagent only. ``None`` (a direct/test
      caller that doesn't need per-agent middleware) leaves the shared
      ``subagent_middleware`` on every subagent — the parity path.

    ``profile`` (optional ``HarnessProfileConfig`` from ``orgs/<org>/
    profile.yaml) applies the ORG-WIDE overrides to EACH
    specialist: ``system_prompt_suffix`` is appended to the body, and
    ``tool_description_overrides`` + ``excluded_tools`` are applied to the
    resolved tool whitelist (so an org-wide override reaches a shared subagent
    like the browser agent, not just the CTO). The helper is imported lazily to
    avoid a module cycle (``profile.py`` imports ``orgs._orgs_dir``).

    PER-AGENT overrides (the universal pattern): the agent's OWN
    frontmatter may carry the SAME three ``HarnessProfileConfig`` fields
    (``system_prompt_suffix`` / ``tool_description_overrides`` /
    ``excluded_tools``), honored per-agent via ``_agent_profile_from_spec``. The
    spec is already ``extends:``-merged (``kit.loaders._merge_extends``), so a
    child agent inherits + overrides these exactly as it inherits tools/skills —
    no second surface. This REPLACED the legacy ``profile.yaml`` ``subagents:``
    block (now a contract failure). A per-agent ``base_system_prompt`` is a
    PERMANENT contract failure (it was a nuclear replace that wiped the body).
    Precedence (most-specific = last word):

        .md body (or extends-merged body)
        → org-wide ``system_prompt_suffix``
        → per-agent ``system_prompt_suffix``

    Tools: resolved whitelist → org-wide prune/rewrite → per-agent
    prune/rewrite (per-agent wins; ``excluded_tools`` is additive,
    ``tool_description_overrides`` per-key wins) → retrieval surface appended.
    """
    if org not in discover_orgs():
        raise KeyError(f"unknown org {org!r}; discovered orgs: {discover_orgs()}")
    tool_map: dict[str, BaseTool] = {t.name: t for t in all_tools}
    apply_profile_to_tools = None
    if profile is not None:
        # Lazy: profile.py imports ``_orgs_dir`` from THIS module at load time.
        # profile_apply.py imports profile.py at load time → same cycle. The seam
        # owns the application; route through it.
        from pux_harness.agent.profile_apply import apply_profile_to_tools as _aptt
        apply_profile_to_tools = _aptt
    subs: list[dict[str, Any]] = []
    # Level-1 of the two-level grant gate: the servers this org DECLARES in its
    # own org.yaml (the same source resolve_tool_servers arms from). Computed
    # ONCE per org (offline-safe config read), so every agent's mcp ref is
    # checked against the same declared set — declared-but-unreachable -> empty
    # subset (build lives), undeclared -> fail loud (config error).
    declared_servers = _org_declared_mcp_servers(org)
    for slug in org_agent_slugs(org):
        spec = _load_agent_spec(slug, org)
        if spec is None:
            searched = [str(p / f"{slug}.md") for p in _agent_search_dirs(org)]
            raise FileNotFoundError(
                f"no agent {slug!r} for org {org!r} — searched {searched}")
        sub = _build_sub(
            slug, spec, tool_map, spec["system_prompt"], org,
            middleware=subagent_middleware,
            mcp_tools=mcp_tools,
            declared_servers=declared_servers,
        )
        # Per-subagent middleware: a frontmatter ``middleware: [name, ...]``
        # mounts those REGISTERED middleware on THIS subagent only (e.g.
        # ``audit`` on a read-only investigator). The resolver (from
        # ``stack.make_subagent_middleware_builder``) re-resolves the org's
        # subagent baseline + the per-agent names, validated + registry-ordered,
        # so ``audit`` lands outermost. A bare string is accepted (one name);
        # ``None`` frontmatter keeps the shared list.
        extra_mw = spec.get("middleware")
        if extra_mw:
            if build_subagent_middleware is None:
                raise ValueError(
                    f"{org}/{slug}: agent frontmatter declares `middleware:` "
                    f"{extra_mw!r} but no per-subagent resolver was supplied "
                    f"(the runtime factory always supplies one; a direct/test "
                    f"caller that arms per-agent middleware must pass "
                    f"build_subagent_middleware=...)."
                )
            mw_names = [extra_mw] if isinstance(extra_mw, str) else list(extra_mw)
            # Thread the agent's per-agent ``rubric:`` frontmatter field (if any)
            # so the middleware builder can prepend a ``_RubricOverride`` that
            # injects the agent's OWN rubric into state before RubricMiddleware.
            agent_rubric = spec.get("rubric")
            sub["middleware"] = build_subagent_middleware(
                mw_names, rubric_text=agent_rubric,
            )
        # Per-agent overrides from the spec's OWN frontmatter.
        agent_cfg = _agent_profile_from_spec(spec)
        # Subagent prompt = its OWN body + the org-wide suffix + its own
        # per-agent suffix — NO supervisor content (the user's hard rule:
        # subagents are SPECIALIZED for independent tasks). Assembled by the
        # no-gap registry (``prompt_parts``). A per-agent ``base_system_prompt``
        # (a nuclear replace that would wipe the body) is GONE — a stray one in
        # frontmatter must FAIL, not silently drop (else it's a gap).
        if "base_system_prompt" in spec:
            raise ValueError(
                f"{org}/{slug}: `base_system_prompt` is removed — it was a "
                f"per-agent global-REPLACE that wiped the agent's own body. "
                f"Use `system_prompt_suffix` (append) instead."
            )
        sub["system_prompt"] = assemble_prompt(
            (*SUBAGENT_PROMPT_PARTS, *_load_extra_parts_for_scope(org, PromptScope.SUBAGENT)),
            PromptCtx(
                agent_body=spec["system_prompt"],
                system_prompt_suffix=(
                    profile.system_prompt_suffix if profile is not None else None
                ),
                agent_system_prompt_suffix=(
                    agent_cfg.system_prompt_suffix
                    if agent_cfg is not None else None
                ),
            ),
            PromptScope.SUBAGENT,
        )
        # Tools: .md → org-wide prune/rewrite → per-agent prune/rewrite.
        if sub.get("tools"):
            if profile is not None:
                sub["tools"] = apply_profile_to_tools(sub["tools"], profile)
            if agent_cfg is not None and (
                agent_cfg.excluded_tools or agent_cfg.tool_description_overrides
            ):
                from pux_harness.agent.profile_apply import apply_profile_to_tools as _aptt_per
                sub["tools"] = _aptt_per(sub["tools"], agent_cfg)
        # Retrieval surface, appended AFTER profile filtering so an org-wide
        # ``excluded_tools`` can't strip it. Guarded by ``sub.get("tools")``: a
        # spec with no whitelist inherits the main agent's tools (already
        # including these via the factory), and synthesizing a list here would
        # silently break that inheritance.
        if sub.get("tools"):
            have = {t.name for t in sub["tools"]}
            sub["tools"] = [
                *sub["tools"], *(t for t in retrieval_tools if t.name not in have)
            ]
        subs.append(sub)
    return subs
