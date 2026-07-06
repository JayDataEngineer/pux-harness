"""Org + specialist-agent loading for the deepagents harness.

System prompt = root AGENTS.md + orgs/<name>/AGENTS.md + harness addendum.
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
from typing import Any, TYPE_CHECKING

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool
from deepagents import HarnessProfileConfig

from pux_harness.agent.model import get_model
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
# / ``_specialists_dir()`` seams; ``load_root_prompt`` (the root AGENTS.md is
# pinned to ``PROJECT_ROOT``, NOT the ``_orgs_dir()`` seam — a tempdir test
# patches ``_orgs_dir`` but the base prompt stays the real one); ``_read``
# (the PROJECT_ROOT-bound reader); ``build_system_prompt`` (owns ``_ADDENDUM``);
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


def _read(rel: str) -> str:
    """Read a project-relative file (used for the root ``AGENTS.md`` only —
    org/agent/skill reads go through the injectable helpers above so the loader
    is testable via monkeypatch)."""
    return (project_root() / rel).read_text()


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


_ADDENDUM = """\

## Harness addendum (deepagents) — authoritative

You are running under the Python deepagents harness. Where this addendum
conflicts with the org docs above, THIS ADDENDUM wins.

- **Delegation:** delegate with the `task` tool:
  `task(subagent_type="<name>", description="<what to do>")`. The subagents
  available to you are listed in the `task` tool's own description. The
  subagent sees only your `description`, not your conversation — give it
  enough context (relevant paths, the question, the expected output shape).
- **File/shell surface:** the file and shell tools are the NATIVE deepagents
  tools — `execute` (run a shell command), `read_file`, `write_file`,
  `edit_file`, `glob`, `grep`, `ls`. There is NO `pux_sandbox_bash` or
  `pux_sandbox_file_*`. Anywhere the org docs say `pux_sandbox_bash`, use
  `execute`; `pux_sandbox_file_read` -> `read_file`; `pux_sandbox_file_glob`
  -> `glob`; `pux_sandbox_file_grep` -> `grep`; and so on. Specialist
  capabilities remain under `pux_sandbox_*` (`pux_sandbox_python`,
  `pux_sandbox_browser_*`, `pux_sandbox_desktop_*`, `pux_sandbox_describe_image`,
  `pux_sandbox_list_skills`). Skill BODIES are peeked with the native
  `read_file` (the ``SkillsMiddleware`` advertises each skill's name +
  description in your prompt; `list_skills` is the host-side catalog) — there is
  no `pux_sandbox_load_skill`. The workspace is at `/sandbox/workspace/` inside
  the sandbox container — the project root, bind-mounted. You and every
  subagent share this same surface.
"""


def load_root_prompt() -> str:
    """Body of the root AGENTS.md (the base 'Pux' system prompt).

    Harness-local (NOT a delegate): the root prompt is read through ``_read``,
    which resolves ``project_root()`` LIVE — NOT the ``_orgs_dir()`` seam. A
    tempdir contract test patches ``_orgs_dir`` to a throwaway tree, but the base
    'Pux' prompt must stay the real shipped one — so this reads through
    ``project_root()``, deliberately outside the delegated seam (the kit's
    ``load_root_prompt`` would return ``""`` against a tempdir with no AGENTS.md,
    changing ``build_system_prompt``)."""
    return _aloaders._split_frontmatter(_read("AGENTS.md"))[1]


def load_org_prompt(name: str) -> str:
    """Body of orgs/<name>/AGENTS.md (the per-org CTO overlay).

    Thin delegate to ``pux_harness.kit.loaders``; project root from ``_orgs_dir()``."""
    return _aloaders.load_org_prompt(name, _orgs_dir().parent)


def build_system_prompt(org: str) -> str:
    """root AGENTS.md + the chain-inherited org overlay + harness addendum
    (mirrors pi-mono's append-org-to-root assembly, plus the deepagents
    terminology bridge). The overlay is the parent's + own AGENTS.md
    concatenated (own last) when the org ``extends:`` a parent — read via the
    kit's cycle-aware ``_chain_overlay``. For a non-extending org, byte-identical
    to ``root + own + addendum``."""
    overlay = _aloaders._chain_overlay(org, _orgs_dir().parent)
    return f"{load_root_prompt()}\n\n{overlay}{_ADDENDUM}"


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
    if spec.get("tools"):
        sub["tools"] = _resolve_tools(spec["tools"], tool_map)
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
    "base_system_prompt", "system_prompt_suffix",
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
# org — including dev-bot, the Claude-Code-equivalent coding org whose roster
# rule (``dev-bot-no-general-subagent``) checks ``org.yaml`` and so NEVER sees
# the auto-added slot.
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

# The pux-default description/system_prompt for a customized GP the org enables
# without supplying its own (the native field's description/system_prompt are
# both optional). Mirrors deepagents' GENERAL_PURPOSE_SUBAGENT intent: a
# generalist fallback for tasks no specialist covers.
_DEFAULT_GP_DESCRIPTION = (
    "General-purpose worker for tasks no specialist covers. Has the full "
    "specialist + retrieval tool surface."
)
_DEFAULT_GP_PROMPT = (
    "You are a general-purpose subagent. Complete the delegated task directly "
    "using the tools available; do not delegate further. Return the result, not "
    "a log of how you got there."
)

# Safeguard S1: the disabled slot must say so honestly + carry NO tools, so even
# a stray delegation returns immediately rather than silently doing generic work.
_DISABLED_GP_DESCRIPTION = "Disabled for this org — do not delegate here."
_DISABLED_GP_PROMPT = (
    "This subagent is disabled for this org. Do not attempt the task; return "
    "immediately with a one-line notice that this slot is disabled."
)


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
    matching the precedence every other subagent follows in ``load_subagents``."""
    # Lazy import: profile.py imports ``_orgs_dir`` from THIS module → cycle.
    from pux_harness.agent.profile import apply_profile_to_tools

    model = get_model(role="worker", org=org)
    if gp.enabled is False:
        return {
            "name": _GENERAL_PURPOSE_NAME,
            "description": _DISABLED_GP_DESCRIPTION,
            "system_prompt": _DISABLED_GP_PROMPT,
            "tools": [],
            "model": model,
            "middleware": [],
        }
    description = gp.description or _DEFAULT_GP_DESCRIPTION
    prompt = gp.system_prompt or _DEFAULT_GP_PROMPT
    if profile is not None and profile.system_prompt_suffix:
        prompt = f"{prompt}\n\n{profile.system_prompt_suffix}"
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

    ``profile`` (optional ``HarnessProfileConfig`` from ``orgs/<org>/
    profile.yaml) applies the ORG-WIDE overrides to EACH
    specialist: ``system_prompt_suffix`` is appended to the body, and
    ``tool_description_overrides`` + ``excluded_tools`` are applied to the
    resolved tool whitelist (so an org-wide override reaches a shared subagent
    like the browser agent, not just the CTO). The helper is imported lazily to
    avoid a module cycle (``profile.py`` imports ``orgs._orgs_dir``).

    PER-AGENT overrides (the universal pattern): the agent's OWN
    frontmatter may carry the SAME four ``HarnessProfileConfig`` fields
    (``base_system_prompt`` / ``system_prompt_suffix`` /
    ``tool_description_overrides`` / ``excluded_tools``), honored per-agent via
    ``_agent_profile_from_spec``. The spec is already ``extends:``-merged
    (``kit.loaders._merge_extends``), so a child agent inherits + overrides these
    exactly as it inherits tools/skills — no second surface. This REPLACED the
    legacy ``profile.yaml`` ``subagents:`` block (now a contract failure).
    Precedence (most-specific = last word):

        .md body (or extends-merged body)
        → per-agent ``base_system_prompt`` (replace)
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
        from pux_harness.agent.profile import apply_profile_to_tools as _aptt
        apply_profile_to_tools = _aptt
    subs: list[dict[str, Any]] = []
    for slug in org_agent_slugs(org):
        spec = _load_agent_spec(slug, org)
        if spec is None:
            searched = [str(p / f"{slug}.md") for p in _agent_search_dirs(org)]
            raise FileNotFoundError(
                f"no agent {slug!r} for org {org!r} — searched {searched}")
        sub = _build_sub(
            slug, spec, tool_map, spec["system_prompt"], org,
            middleware=subagent_middleware,
        )
        # Per-agent overrides from the spec's OWN frontmatter.
        agent_cfg = _agent_profile_from_spec(spec)
        # Prompt precedence: body → per-agent base (replace) → org-wide suffix
        # → per-agent suffix (most-specific = last word).
        if agent_cfg is not None and agent_cfg.base_system_prompt:
            sub["system_prompt"] = agent_cfg.base_system_prompt
        if profile is not None and profile.system_prompt_suffix:
            sub["system_prompt"] = (
                f"{sub['system_prompt']}\n\n{profile.system_prompt_suffix}"
            )
        if agent_cfg is not None and agent_cfg.system_prompt_suffix:
            sub["system_prompt"] = (
                f"{sub['system_prompt']}\n\n{agent_cfg.system_prompt_suffix}"
            )
        # Tools: .md → org-wide prune/rewrite → per-agent prune/rewrite.
        if sub.get("tools"):
            if profile is not None:
                sub["tools"] = apply_profile_to_tools(sub["tools"], profile)
            if agent_cfg is not None and (
                agent_cfg.excluded_tools or agent_cfg.tool_description_overrides
            ):
                from pux_harness.agent.profile import apply_profile_to_tools as _aptt_per
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
