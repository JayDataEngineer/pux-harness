"""Declarative org contract — the portable org <-> harness interface.

An org is a directory ``orgs/<name>/`` containing ``AGENTS.md`` (CTO prompt
prose) and optionally ``org.yaml`` (the specialist roster). This module
enforces that every org is a self-contained, portable bundle with **no
harness-level per-org code coupling** — pillar (a) of the deepagents pivot:
orgs declare what they need, the harness treats them generically.

Two validation tiers:

* **Structural** (always checked, no server, no model tokens):
  1. ``AGENTS.md`` present.
  2. ``AGENTS.md`` carries no frontmatter (prose-only); roster lives in
     ``org.yaml``.
  3. Every ``org.yaml`` slug resolves to a ``orgs/<org>/agents/<slug>.md`` (or
     ``orgs/_shared/agents/<slug>.md``) whose frontmatter carries ``name`` +
     ``description`` and whose body (the system prompt) is non-empty.
  5. Optional ``policy.yaml`` parses and uses known sections.

* **Tool-resolution** (rule 4 — always on, no server): every entry in an
  agent's frontmatter ``tools`` list resolves through ``classify_slug`` (from
  the tool ``REGISTRY``) to a native fs tool OR a ``pux_sandbox_*`` specialist.
  The classifier is pure Python, so this runs offline in pytest and in
  ``--check-contract`` with no container or Go server — and the runtime
  resolver (``orgs._resolve_tools``) shares it, so the two paths agree.

* **Harness-level** (rule 6) and **global** (rules 7-8):
  6. No hardcoded org->agent manifest in the harness source.
  7. No orphan agents (every specialist owned by >=1 org).
  8. Skill hygiene: every ``SKILL.md`` is Agent-Spec well-formed, and no ``.md``
     sits loose directly under a skills root (``check_skill_roots``).

* **Permanent legacy tripwires** (``no-legacy-agent-py``,
  ``no-legacy-org-roster``, ``no-legacy-sandbox-artifacts``,
  ``no-legacy-middleware-in-graph``, ``no-legacy-memory-saver``,
  ``no-legacy-subagents-block``): the legacy
  ``.pi/agents/<slug>.py`` SUBAGENT-dict module form, the
  ``agents:``-key-on-AGENTS.md org form, the frozen bash/compose sandbox
  lifecycle, a hand-assembled middleware list in ``graph.py``, an ephemeral
  ``MemorySaver`` in ``acp.py``/``main.py``, and the ``profile.yaml``
  ``subagents:`` block (a second partial-override surface) are structurally
  forbidden — ``--check-contract`` blocks any commit that reintroduces them.
  Agents are now
  one frontmatter+body ``<slug>.md`` per org (org-local first, then
  ``orgs/_shared/agents/``); the sandbox lifecycle is harness-owned; the
  middleware stack is built by the single ``stack.build_stack`` factory; the
  server-side runtimes share one persistent ``AsyncSqliteSaver``; and per-agent
  overrides go in the agent's own frontmatter (+ ``extends:`` for inheritance)
  (``threads.open_thread_store``). A sixth tripwire
  ``no-duplicate-loaders-in-orgs`` keeps the 7 pure org/agent loaders in
  ``orgs.py`` as thin delegates to ``pux_harness.kit.loaders``, so the verbatim duplication the user flagged can't return. A
  seventh tripwire ``kit-import-isolation`` (Stage 2 import hygiene) keeps the
  slim kit core (``pux_harness/kit/**`` + ``pux_harness/__init__.py``) free of
  heavy runtime imports (``docker``/``fastapi``/…) and sibling-subsystem
  reaches — the precondition for Stage 3 splitting the heavy deps into optional
  extras so a bare ``pip install pux-harness`` is truly slim. An eighth
  tripwire ``no-harness-profile-registration`` (registry parity) keeps
  pux OFF the model-keyed ``_HARNESS_PROFILES`` registry — pux applies
  ``HarnessProfileConfig`` fields itself in ``build_stack`` /
  ``load_subagents``, so its middleware is never stripped by deepagents' own
  ``_apply_excluded_middleware`` (which only fires through a registered
  profile).

Rule 4 resolves through ``classify_slug`` (the single source of truth shared
with ``orgs._resolve_tools`` and ``graph.py`` via the ``REGISTRY``), so a
stale ``tools:`` reference fails loud here without any process to probe.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pux_harness.sandbox import policy as policy_mod
from pux_harness.kit import _paths
from pux_harness.agent import profile as profile_mod
from pux_harness.agent import model as model_mod
from pux_harness.agent import stack as stack_mod
from pux_harness.agent import tool_servers as tool_servers_mod
from pux_harness.sandbox.tools import (
    NATIVE_FS_TOOLS,
    SPECIALIST_TOOL_NAMES,
    Category,
    classify_slug,
    prefixed,
)
from pux_harness.agent.orgs import (
    PROJECT_ROOT,
    _agent_search_dirs,
    _load_agent_spec,
    _org_path,
    _orgs_dir,
    _parse_list,
    _split_frontmatter,
    discover_orgs,
    org_agent_slugs,
    org_extends,
)
# ``_orgs_dir`` / ``_agent_search_dirs`` / ``_load_agent_spec`` are re-exported
# here (bound into THIS module's namespace by the import) so the contract tests
# can monkeypatch them at the existing call sites.

# --- the contract vocabulary ----------------------------------------------

# Every agent ``<slug>.md`` must carry these (name + description from
# frontmatter; system_prompt = the body).
_REQUIRED_AGENT_KEYS: frozenset[str] = frozenset({
    "name", "description", "system_prompt",
})

# Optional ``orgs/<name>/policy.yaml`` top-level sections. ``host_setup`` is
# harness-added (no Go equivalent) — the host-side prep-hook list.
# ``build`` is a sub-key under ``sandbox``, NOT a top-level section.
KNOWN_POLICY_SECTIONS: frozenset[str] = frozenset({
    "workspace", "egress", "credentials", "sandbox", "browser", "host_setup",
    "jobs", "tool_servers",
})

# ``NATIVE_FS_TOOLS`` is imported from ``pux_harness.sandbox.tools`` (derived
# from the single ``REGISTRY``) — see the import above. Re-exported through
# this module's namespace so the contract tests can keep importing it from
# ``contract`` (``from pux_harness.agent.contract import NATIVE_FS_TOOLS``).

# Agent-Skills spec: a skill dir name (and its ``SKILL.md`` ``name``) must be
# kebab-case — lowercase letters/digits joined by single hyphens.
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Violation:
    """One contract failure. ``severity`` is "error" (fails green) or "warn"
    (SHOULD). The green gate treats only errors as blocking."""

    severity: str  # "error" | "warn"
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.rule}: {self.message}"


# --- discovery (orgs + agent-slugs live in the low-level orgs module) ----
# ``_orgs_dir`` / ``_agent_search_dirs`` / ``_load_agent_spec`` / ``_parse_list``
# / ``_split_frontmatter`` are all imported from ``orgs`` (re-exported above) —
# single source of truth. The contract tests monkeypatch them at
# ``contract._orgs_dir`` etc., which still works because the import binds the
# names into THIS module's namespace.


# --- per-org checks (rules 1-5) ------------------------------------------


def _load_agent_subagent(slug: str, org: str) -> dict[str, Any] | None:
    """Read ``<slug>.md`` (org-local then ``_shared``) -> spec dict, or ``None``.

    Delegates to ``orgs._load_agent_spec`` (single source of truth — the runtime
    loader and the contract read the SAME file). Returns ``None`` if no
    ``<slug>.md`` exists in either search dir; the caller (``check_org``)
    reports an ``agent-resolves`` violation. A malformed frontmatter raises
    ``ValueError`` (caught + reported by the caller)."""
    return _load_agent_spec(slug, org)


def _first_agent_md(slug: str, org: str) -> Path | None:
    """The first ``<slug>.md`` that exists in the agent search dirs (org-local
    then ``_shared``), or ``None``. Used by the extends-chain walker to read RAW
    frontmatter without triggering the recursive merge in ``_load_agent_spec``."""
    for d in _agent_search_dirs(org):
        path = d / f"{slug}.md"
        if path.is_file():
            return path
    return None


def _agent_extends_chain_violations(slug: str, org: str) -> list[Violation]:
    """Validate an agent's ``extends:`` chain resolves + is acyclic.

    Reads RAW frontmatter (NO merge) and walks the chain manually so the two
    dedicated rules fire with precise, actionable messages — independent of
    ``_load_agent_spec``'s own recursion (which raises + would surface as a
    generic ``agent-resolves``). A chain that references a non-existent agent
    fires ``agent-extends-resolvable``; a cycle fires ``agent-extends-acyclic``.

    Returns ``[]`` for an agent with no ``extends:`` (the common case), a
    missing roster slug (``agent-resolves`` owns that — ``cur == slug`` + path
    ``None``), or unreadable frontmatter (``agent-resolves`` /
    ``no-legacy-agent-py`` own that). Callers skip the merge when this returns
    non-empty so the same fault isn't reported twice."""
    v: list[Violation] = []
    chain: list[str] = [slug]
    visited: set[str] = {slug}
    cur = slug
    while True:
        path = _first_agent_md(cur, org)
        if path is None:
            if cur != slug:
                v.append(Violation(
                    "error", "agent-extends-resolvable",
                    f"{org}/{slug}: extends chain references unknown agent "
                    f"{cur!r} (chain: {' -> '.join(chain)})"))
            return v  # cur == slug + missing -> agent-resolves owns it
        try:
            fm, _ = _split_frontmatter(path.read_text())
        except ValueError:
            return v  # bad frontmatter -> agent-resolves / no-legacy-agent-py own it
        parent = fm.get("extends")
        if not isinstance(parent, str) or not parent.strip():
            return v  # chain terminates cleanly (no extends); bad-type -> merge raises
        parent = parent.strip()
        if parent in visited:
            v.append(Violation(
                "error", "agent-extends-acyclic",
                f"{org}/{slug}: extends cycle detected "
                f"({' -> '.join(chain)} -> {parent})"))
            return v
        chain.append(parent)
        visited.add(parent)
        cur = parent


def _org_extends_chain_violations(name: str) -> list[Violation]:
    """Validate an org's ``extends:`` chain resolves + is acyclic.

    Mirrors ``_agent_extends_chain_violations``: walks the chain via the RAW
    single-hop reader (``org_extends`` — reads ``org.yaml``'s ``extends:`` with
    NO recursion, NO merge) so the two dedicated rules fire with precise,
    actionable messages, independent of ``org_extends_chain``'s own recursion
    (which raises + would surface as a generic crash in the cycle-safe runtime
    loaders). A parent that is no org, or an org without ``AGENTS.md`` (not a
    valid base), fires ``org-extends-resolvable``; a cycle fires
    ``org-extends-acyclic``. Returns ``[]`` for an org with no ``extends:``
    (the common case). Callers skip the inherited-roster read when this returns
    non-empty so a broken chain doesn't double-report through ``org_agent_slugs``
    (which falls back to ``[name]`` anyway)."""
    v: list[Violation] = []
    chain: list[str] = [name]
    visited: set[str] = {name}
    cur = name
    while True:
        parent = org_extends(cur)
        if parent is None:
            return v  # chain terminates cleanly (no extends)
        if parent in visited:
            v.append(Violation(
                "error", "org-extends-acyclic",
                f"{name}: extends cycle detected ({' -> '.join(chain)} -> {parent})"))
            return v
        try:
            pdir = _org_path(parent)
        except FileNotFoundError:
            v.append(Violation(
                "error", "org-extends-resolvable",
                f"{name}: extends {parent!r} -> no such org "
                f"(chain: {' -> '.join(chain)} -> {parent})"))
            return v
        if not (pdir / "AGENTS.md").is_file():
            v.append(Violation(
                "error", "org-extends-resolvable",
                f"{name}: extends {parent!r} -> {parent}/AGENTS.md missing "
                f"(not a valid base org; chain: {' -> '.join(chain)} -> {parent})"))
            return v
        chain.append(parent)
        visited.add(parent)
        cur = parent


def check_org(name: str) -> list[Violation]:
    """Validate one org's bundle — fully offline (no server, no tokens).

    Rules 1,2,3,5 are structural. Rule 4 (tool-resolution) classifies every
    agent ``SUBAGENT["tools"]`` entry via ``classify_slug`` (native ∪
    specialist surface derived from the ``REGISTRY``) — pure Python, so it
    runs in pytest and ``--check-contract`` with nothing live.
    """
    v: list[Violation] = []
    # Specialists-aware: orgs/<name> (e.g. ``general``) then
    # orgs/specialists/<name> (orgs moved under specialists/ in the reorg).
    org_dir = _org_path(name)
    agents_md = org_dir / "AGENTS.md"

    # Rule 1.
    if not agents_md.is_file():
        return [Violation("error", "org-agents-md",
                          f"{name}: orgs/{name}/AGENTS.md missing")]

    # Rule 2: AGENTS.md is prose-only (no frontmatter). The roster lives in
    # org.yaml. Permanent tripwire against reintroduction.
    fm, _ = _split_frontmatter(agents_md.read_text())
    if fm:
        v.append(Violation(
            "error", "no-legacy-org-roster",
            f"{name}: AGENTS.md carries YAML frontmatter — the roster must "
            f"live in orgs/{name}/org.yaml and AGENTS.md must be prose-only"))

    # Validate the ``extends:`` chain FIRST (specific rules), before
    # the inherited-roster read below. A broken chain reports
    # ``org-extends-resolvable`` / ``org-extends-acyclic`` here; the runtime
    # ``org_agent_slugs`` is cycle-safe (falls back to ``[name]``) so it never
    # crashes — this walker surfaces the real fault with a precise message.
    v.extend(_org_extends_chain_violations(name))

    # Safeguard S6 — policy.yaml is NEVER inherited (each org owns its egress;
    # security boundary). A child that ``extends:`` a parent but ships no OWN
    # policy.yaml runs with NO egress ACL — surprising + a likely footgun. Warn
    # (not error): an org legitimately MAY run wide-open by choice; this just
    # forces the choice to be explicit. Skipped when the child has no parent.
    if org_extends(name) is not None and not (org_dir / "policy.yaml").is_file():
        v.append(Violation(
            "warn", "org-extends-policy",
            f"{name}: extends a parent but ships no own policy.yaml — policy "
            f"is NOT inherited (each org owns its egress). Add a policy.yaml, "
            f"or confirm this org should run with no egress ACL."))

    # Read slugs from org.yaml (the only valid roster source).
    org_yaml = org_dir / "org.yaml"
    shape_ok = True
    if org_yaml.is_file():
        data = yaml.safe_load(org_yaml.read_text()) or {}
        if not isinstance(data, dict):
            v.append(Violation(
                "error", "org-yaml-shape",
                f"{name}: org.yaml top-level must be a mapping, "
                f"got {type(data).__name__}"))
            shape_ok = False
            slugs: list[str] = []
        else:
            slugs = _parse_list(data.get("agents"))
    elif not fm:
        # No org.yaml AND AGENTS.md has no frontmatter → empty roster (valid
        # for a CTO-only org with no specialists).
        slugs = []
    else:
        # No org.yaml but AGENTS.md has frontmatter — already reported above.
        slugs = _parse_list(fm.get("agents", ""))

    # The EFFECTIVE roster is the chain-inherited union
    # (parent ``agents:`` ∪ own) — what the runtime actually delegates to. Used
    # for agent-resolution (Rule 3) + tool-resolution (Rule 4) so an INHERITED
    # slug is validated the same as an own one. Cycle-safe: falls back to own
    # ``slugs`` on a broken chain or malformed org.yaml (both already reported).
    roster: list[str] = slugs
    if shape_ok:
        try:
            roster = org_agent_slugs(name)
        except Exception:
            roster = slugs

    # Permanent tripwire (no-legacy-left-behind): dev-bot is the
    # Claude-Code-equivalent CODING org — the CTO does all the thinking and
    # delegates only narrow execution (code-worker) / recon (dev-bot-explorer)
    # / e2e verification (web-agent). A generic catch-all subagent
    # (``general`` / ``general-purpose`` / the shared ``researcher``) would let
    # the CTO delegate the DESIGN itself, which is exactly the anti-pattern the
    # roster exists to prevent. A future re-add is a HARD contract failure, not
    # a silent drift.
    if name == "dev-bot":
        forbidden = {"general", "general-purpose", "researcher"}
        bad = sorted(set(slugs) & forbidden)
        if bad:
            v.append(Violation(
                "error", "dev-bot-no-general-subagent",
                f"dev-bot: roster must not include a generic subagent "
                f"({bad}); the CTO does the thinking — delegate only to "
                f"dev-bot-explorer / code-worker / web-agent"))

    # Defense in depth via a SECOND code path. The roster rule above
    # (``dev-bot-no-general-subagent``) reads org.yaml, so it NEVER sees the
    # ``general-purpose`` slot deepagents auto-adds to EVERY graph
    # (deepagents/graph.py:716-717) when no inline spec owns that name. This
    # sibling rule reads profile.yaml and asserts dev-bot OPTS OUT of that
    # auto-add via the NATIVE ``general_purpose_subagent.enabled: false`` field
    # — which the harness turns into a neutered spec (``orgs.
    # _build_general_purpose_sub``, Safeguard S1). Two layers, two files, one
    # intent: dev-bot must not ship a generic catch-all worker. A re-enable OR a
    # dropped field is a HARD contract failure, not a silent drift back to the
    # heavy auto-add. (Skipped on a malformed profile — the ``profile-schema``
    # rule below already reports that, no double-noise.)
    if name == "dev-bot":
        gp_cfg: Any = None
        gp_ok = True
        try:
            gp_cfg = profile_mod.load_profile(name)
        except (TypeError, ValueError, yaml.YAMLError):
            gp_ok = False
        if gp_ok:
            gp = gp_cfg.general_purpose_subagent if gp_cfg is not None else None
            if gp is None or gp.enabled is not False:
                v.append(Violation(
                    "error", "dev-bot-disables-general-purpose",
                    "dev-bot: profile.yaml must declare "
                    "'general_purpose_subagent: {enabled: false}' — deepagents "
                    "otherwise auto-adds a heavy generic worker the roster rule "
                    "(dev-bot-no-general-subagent) cannot see"))

    # Rule 3: every slug resolves to a valid agent .md (org-local or _shared)
    # with required frontmatter keys + a non-empty body (system_prompt).
    # Iterates the INHERITED roster (``roster``), so an agent inherited from a
    # parent via ``extends:`` is resolved through the chain-aware search dirs
    # (child-local → each ancestor → _shared) and validated here too.
    agent_subagents: dict[str, dict[str, Any]] = {}
    for slug in roster:
        # Validate the ``extends:`` chain FIRST (specific rules), before
        # the recursive merge in ``_load_agent_subagent`` raises a generic
        # ``agent-resolves``. A broken chain skips the merge so the same fault
        # isn't reported twice.
        extends_vs = _agent_extends_chain_violations(slug, name)
        if extends_vs:
            v.extend(extends_vs)
            continue
        try:
            sub = _load_agent_subagent(slug, name)
        except Exception as exc:
            v.append(Violation(
                "error", "agent-resolves",
                f"{name}: agents: {slug!r} -> failed to read agent .md: {exc}"))
            continue
        if sub is None:
            looked = ", ".join(
                str(d / f"{slug}.md") for d in _agent_search_dirs(name))
            v.append(Violation(
                "error", "agent-resolves",
                f"{name}: agents: {slug!r} -> no agent .md found "
                f"(searched: {looked})"))
            continue
        agent_subagents[slug] = sub
        missing = sorted(_REQUIRED_AGENT_KEYS - sub.keys())
        if missing:
            v.append(Violation(
                "error", "agent-missing-keys",
                f"{name}/{slug}: agent .md frontmatter missing required "
                f"keys: {missing}"))

    # Rule 4: tool whitelist resolves through the shared classifier
    # (``classify_slug`` from the tool registry). A native slug is accepted
    # (FilesystemMiddleware injects the fs/shell tools for every subagent), a
    # specialist slug resolves to a ``pux_sandbox_*`` tool, anything else is
    # unknown. The SAME classifier drives the runtime ``_resolve_tools`` in
    # ``orgs.py``, so the offline check and the runtime can no longer disagree
    # (the old paths diverged on native slugs). No server probe — pure Python,
    # runs identically in pytest and ``--check-contract``.
    for slug, sub in agent_subagents.items():
        for raw in _parse_list(sub.get("tools", [])):
            tool = raw.rsplit("/", 1)[-1]
            if classify_slug(tool) is None:
                v.append(Violation("error", "tool-resolves",
                                   f"{name}/{slug}: tool {raw!r} -> "
                                   f"{prefixed(tool, Category.SPECIALIST)!r} "
                                   f"not a native fs tool or a "
                                   f"pux_sandbox_* specialist"))

    # Rule 5: policy.yaml parses + valid schema + known sections.
    policy_path = org_dir / "policy.yaml"
    if policy_path.is_file():
        try:
            parsed = yaml.safe_load(policy_path.read_text())
        except yaml.YAMLError as e:
            v.append(Violation("error", "policy-parse",
                               f"{name}: policy.yaml is not valid YAML: {e}"))
            parsed = None
        if isinstance(parsed, dict):
            bad = sorted(k for k in parsed if k not in KNOWN_POLICY_SECTIONS)
            if bad:
                v.append(Violation("error", "policy-sections",
                                   f"{name}: policy.yaml unknown sections "
                                   f"{bad}; allowed: "
                                   f"{sorted(KNOWN_POLICY_SECTIONS)}"))
            pol = None
            try:
                pol = policy_mod.load(name, _orgs_dir().parent)
                policy_mod.resolve_mounts(pol)
            except policy_mod.PolicyError as e:
                v.append(Violation("error", "policy-schema",
                                   f"{name}: policy.yaml schema error: {e}"))
            except policy_mod.NoPolicy:
                pass
            if pol is not None:
                v.extend(_validate_host_setup(name, pol))
                v.extend(_validate_build_spec(name, pol))
                v.extend(_validate_jobs(name, pol))
                v.extend(_validate_tool_servers(name, pol))
        elif parsed is not None:
            v.append(Violation("error", "policy-shape",
                               f"{name}: policy.yaml top-level must be a "
                               f"mapping, got {type(parsed).__name__}"))

    # Optional per-org harness profile. Off by default — most orgs
    # ship none. If present, it must parse into HarnessProfileConfig (unknown
    # keys -> TypeError; bad shapes -> TypeError; bad excluded_middleware
    # grammar -> ValueError). Offline; no model/Docker.
    profile_path = org_dir / "profile.yaml"
    if profile_path.is_file():
        # Permanent no-legacy gate (no-legacy-left-behind). The
        # ``profile.yaml`` ``subagents:`` block was a SECOND partial-override
        # surface with its own resolver; it was folded into per-agent
        # ``extends:`` + delta frontmatter fields. A stale block is a HARD
        # contract failure pointing at the replacement. ``load_profile`` /
        # ``HarnessProfileConfig.from_dict`` reject the same key at BUILD time
        # too (unknown key → ``profile-schema`` below) — two layers over one
        # fault, mirroring the dev-bot GP defense-in-depth (Safeguard S2).
        # Scanned RAW (not via validate_profile) so the message names the fix.
        try:
            _profile_top = yaml.safe_load(profile_path.read_text())
        except yaml.YAMLError:
            _profile_top = None  # validate_profile below owns the parse error
        if isinstance(_profile_top, dict) and "subagents" in _profile_top:
            v.append(Violation(
                "error", "no-legacy-subagents-block",
                f"{name}: profile.yaml: the top-level `subagents:` block was "
                f"removed — folded into per-agent `extends:` + delta frontmatter "
                f"fields (tools_add / tools_remove / skills_add / "
                f"description_append / tool_description_overrides / "
                f"base_system_prompt / system_prompt_suffix / excluded_tools). "
                f"Move each subagent's override into its own "
                f"`orgs/{name}/agents/<slug>.md` (or a shared base + `extends:`)."
            ))
        try:
            profile_mod.validate_profile(name)
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            v.append(Violation("error", "profile-schema",
                               f"{name}: profile.yaml invalid: {exc}"))
            # Skip slug validation too — a malformed profile re-reads
            # below and would raise again.
        else:
            # Validate the ``middleware:`` override block's name + scope
            # — every add/remove name must be a registered middleware
            # (``stack.MIDDLEWARE_REGISTRY``) mounted on the scope it's added to.
            # ``profile.validate_profile`` only SHAPE-checks; the registry lives in
            # ``stack.py``, so the name/scope check fires here (offline). A typo'd
            # override fails --check-contract, not the first build.
            for err in stack_mod.validate_overrides(name):
                v.append(Violation("error", "middleware-overrides", err))

    return v


# --- host_setup + sandbox.build validators (offline) ---------------

# Allowed values in a host_setup hook's ``exports`` mapping. Mirrors the runtime
# check in ``sandbox/host_setup.py`` — kept here (not imported) so the contract
# stays decoupled from the runner module.
_EXPORT_SOURCES = frozenset({"stdout"})


def _validate_host_setup(name: str, pol: policy_mod.Policy) -> list[Violation]:
    """Offline validation of every host_setup hook: each has a name +
    helper_script; helper_script resolves under the project root and exists;
    export sources ⊆ {stdout}; python_deps is a list of strings. Mirrors the
    runner's own checks so a broken hook fails --check-contract before Docker."""
    v: list[Violation] = []
    hooks = policy_mod.host_setup_hooks(pol)
    if not hooks:
        return v
    project_root = _orgs_dir().parent
    for hook in hooks:
        hname = hook.name or "<unnamed>"
        if not hook.name:
            v.append(Violation("error", "host-setup-shape",
                               f"{name}: host_setup hook missing 'name'"))
        if not hook.helper_script:
            v.append(Violation("error", "host-setup-shape",
                               f"{name}/{hname}: host_setup hook missing 'helper_script'"))
            continue
        script = Path(hook.helper_script)
        if not script.is_absolute():
            script = project_root / hook.helper_script
        if not script.is_file():
            v.append(Violation("error", "host-setup-helper-missing",
                               f"{name}/{hname}: helper_script "
                               f"{hook.helper_script!r} not found at {script}"))
        bad = sorted({s for s in hook.exports.values() if s not in _EXPORT_SOURCES})
        if bad:
            v.append(Violation("error", "host-setup-shape",
                               f"{name}/{hname}: unsupported export source(s) {bad}; "
                               f"allowed: {sorted(_EXPORT_SOURCES)}"))
        if not isinstance(hook.python_deps, list) or not all(
            isinstance(d, str) for d in hook.python_deps
        ):
            v.append(Violation("error", "host-setup-shape",
                               f"{name}/{hname}: python_deps must be a list of strings"))
    return v


def _validate_build_spec(name: str, pol: policy_mod.Policy) -> list[Violation]:
    """Offline validation of sandbox.build: dockerfile resolves under the
    project root and exists; context (if set) is a dir. Mirrors the runner's
    build path so a broken build spec fails --check-contract before Docker."""
    v: list[Violation] = []
    spec = policy_mod.build_spec(pol)
    if spec is None:
        return v
    project_root = _orgs_dir().parent
    dockerfile = Path(spec.dockerfile)
    if not dockerfile.is_absolute():
        dockerfile = project_root / spec.dockerfile
    if not dockerfile.is_file():
        v.append(Violation("error", "sandbox-build-shape",
                           f"{name}: sandbox.build dockerfile "
                           f"{spec.dockerfile!r} not found at {dockerfile}"))
    if spec.context:
        context = Path(spec.context)
        if not context.is_absolute():
            context = project_root / spec.context
        if not context.is_dir():
            v.append(Violation("error", "sandbox-build-shape",
                               f"{name}: sandbox.build context "
                               f"{spec.context!r} not found at {context}"))
    return v


def _validate_jobs(name: str, pol: policy_mod.Policy) -> list[Violation]:
    """Offline validation of jobs: each has a name + script; script resolves
    under the project root and exists; timeout is a non-negative integer;
    names are unique. Mirrors the runner's checks so a broken job spec fails
    --check-contract before Docker."""
    v: list[Violation] = []
    specs = policy_mod.job_specs(pol)
    if not specs:
        return v
    project_root = _orgs_dir().parent
    seen_names: set[str] = set()
    for spec in specs:
        jname = spec.name or "<unnamed>"
        if not spec.name:
            v.append(Violation("error", "jobs-shape",
                               f"{name}: job entry missing 'name'"))
        if spec.name in seen_names:
            v.append(Violation("error", "jobs-shape",
                               f"{name}: duplicate job name {spec.name!r}"))
        seen_names.add(spec.name)
        if not spec.script:
            v.append(Violation("error", "jobs-shape",
                               f"{name}/{jname}: job entry missing 'script'"))
            continue
        script = Path(spec.script)
        if not script.is_absolute():
            script = project_root / spec.script
        if not script.is_file():
            v.append(Violation("error", "jobs-script-missing",
                               f"{name}/{jname}: script "
                               f"{spec.script!r} not found at {script}"))
        if spec.timeout < 0:
            v.append(Violation("error", "jobs-shape",
                               f"{name}/{jname}: timeout must be >= 0, "
                               f"got {spec.timeout}"))
    return v


def _validate_tool_servers(name: str, pol: policy_mod.Policy) -> list[Violation]:
    """Offline validation of the ``tool_servers`` declaration in policy.yaml.
    Mirrors ``tool_servers.validate_tool_servers`` but returns Violation objects
    instead of plain strings."""
    v: list[Violation] = []
    for err in tool_servers_mod.validate_tool_servers(name):
        v.append(Violation("error", "tool-servers", err))
    return v


# --- global checks (rules 6-7) -------------------------------------------

_MANIFEST_RE = re.compile(r"^\s*ORG_AGENTS\s*[:=]", re.MULTILINE)


def _agent_dirs() -> list[Path]:
    """Every directory that ships agent definitions: each org's ``agents/``
    plus ``orgs/_shared/agents/`` (shared agents — ``_shared`` is not itself an
    org, so it is not returned by ``discover_orgs``)."""
    orgs = _orgs_dir()
    dirs = [orgs / "_shared" / "agents"]
    dirs += [orgs / org / "agents" for org in discover_orgs()]
    return [d for d in dirs if d.is_dir()]


def orphan_agents() -> list[str]:
    """Agent slugs owned by no org (not listed in any ``org.yaml``).
    Rule 7 — SHOULD (warn), not blocking. Scans every org's ``agents/`` dir
    plus ``orgs/_shared/agents/`` — a shared agent is "owned" if >=1 org lists
    it in its roster."""
    owned: set[str] = set()
    for org in discover_orgs():
        owned.update(org_agent_slugs(org))
    all_agents = {p.stem for d in _agent_dirs() for p in d.glob("*.md")}
    return sorted(all_agents - owned)


def _no_legacy_agent_py() -> list[Violation]:
    """No ``.py`` agent may ship, and every agent ``.md`` must be well-formed.

    Permanent tripwire (flipped the legacy two-phase form). The legacy
    ``.pi/agents/<slug>.py`` SUBAGENT-dict module was replaced by ONE
    frontmatter+body ``<slug>.md`` per org (``orgs/<org>/agents/`` +
    ``orgs/_shared/agents/``). A re-introduced ``.py`` agent, an agent ``.md``
    whose frontmatter is missing a required key (``name``/``description``),
    or an empty body (no system prompt) is a HARD contract failure — not a
    silent dual-read.
    """
    v: list[Violation] = []
    for d in _agent_dirs():
        for path in sorted(d.iterdir()):
            if path.suffix == ".py":
                v.append(Violation(
                    "error", "no-legacy-agent-py",
                    f"{path}: .py agents are the forbidden legacy form — use a "
                    f"frontmatter+body .md (see "
                    f"orgs/_shared/agents/researcher.md for the shape)"))
            elif path.suffix == ".md":
                try:
                    fm, body = _split_frontmatter(path.read_text())
                except ValueError as exc:
                    v.append(Violation(
                        "error", "no-legacy-agent-py",
                        f"{path}: frontmatter does not parse: {exc}"))
                    continue
                spec = {**fm, "system_prompt": body}
                missing = sorted(_REQUIRED_AGENT_KEYS - spec.keys())
                if missing:
                    v.append(Violation(
                        "error", "no-legacy-agent-py",
                        f"{path}: missing required frontmatter keys: {missing}"))
                if not body.strip():
                    v.append(Violation(
                        "error", "no-legacy-agent-py",
                        f"{path}: empty body — the agent has no system prompt"))
    return v


def _no_legacy_middleware_in_graph() -> list[Violation]:
    """Permanent tripwire (no-legacy-left-behind): ``graph.py`` must
    NOT import the deepagents middleware classes directly.

    The factory ``stack.build_stack`` is the SINGLE place the per-org middleware
    stack is resolved (the user's "one place to adjust defaults" goal). Before
    the factory, ``graph.py`` hand-assembled the middleware list — importing
    ``RoutingMiddleware`` / ``SessionGuideMiddleware`` / ``RubricMiddleware``
    and building them inline. That dual-read (a second, hand-maintained
    middleware list) is exactly the drift the factory killed: an override in
    ``stack.MIDDLEWARE_REGISTRY`` or ``DEFAULT_SUPERVISOR`` would silently NOT
    reach the graph. A future re-introduction (someone wires a middleware in
    ``graph.py`` because it "feels simpler") is a HARD contract failure, not a
    silent regression — mirroring ``no-legacy-agent-py`` /
    ``no-legacy-sandbox-artifacts``.

    Mechanism: AST-scan ``graph.py``'s IMPORT nodes for the three banned names.
    AST (not a regex) so a commented-out line or a string literal doesn't
    trip a false positive, and a re-import under a renamed alias DOES trip.
    """
    v: list[Violation] = []
    graph_src = Path(__file__).with_name("graph.py")
    if not graph_src.is_file():
        return v  # the tripwire is about graph.py; nothing to check if it's gone
    banned = {"RoutingMiddleware", "SessionGuideMiddleware", "RubricMiddleware"}
    try:
        tree = ast.parse(graph_src.read_text())
    except SyntaxError as exc:  # pragma: no cover - graph.py is imported, so valid
        v.append(Violation(
            "error", "no-legacy-middleware-in-graph",
            f"{graph_src}: does not parse: {exc}"))
        return v
    for node in ast.walk(tree):
        names: set[str] = set()
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[-1] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names = {alias.name.split(".")[-1] for alias in node.names}
        hits = names & banned
        if hits:
            v.append(Violation(
                "error", "no-legacy-middleware-in-graph",
                f"{graph_src}: imports {sorted(hits)} — middleware assembly "
                f"belongs in stack.build_stack (the single factory); graph.py "
                f"is the thin deps+binding caller"))
    return v


def _no_legacy_sandbox_artifacts() -> list[Violation]:
    """No ``orgs/<name>/{bootstrap.sh,docker-compose.yml,docker-compose.override.yml}``
    may ship — the harness owns the full sandbox lifecycle now (the
    frozen bash/compose shadow lifecycle was deleted). Permanent tripwire
    (no-legacy-left-behind): a future re-introduction is a HARD failure, not a
    silent regression — mirroring ``no-legacy-agent-py`` /
    ``no-legacy-org-roster``."""
    v: list[Violation] = []
    orgs = _orgs_dir()
    if not orgs.is_dir():
        return v
    for org_dir in sorted(orgs.iterdir()):
        if not org_dir.is_dir():
            continue
        for art in ("bootstrap.sh", "docker-compose.yml", "docker-compose.override.yml"):
            if (org_dir / art).is_file():
                v.append(Violation(
                    "error", "no-legacy-sandbox-artifacts",
                    f"orgs/{org_dir.name}/{art}: the harness owns the sandbox "
                    f"lifecycle now (policy.yaml host_setup + sandbox.build); "
                    f"this bash/compose artifact must be deleted"))
    return v


def _no_legacy_subagents_block() -> list[Violation]:
    """Permanent tripwire (no-legacy-left-behind): no
    ``profile.yaml`` may ship a top-level ``subagents:`` block.

    The per-subagent override surface was folded INTO agent frontmatter —
    ``extends:`` + the delta fields (``tools_add`` / ``tools_remove`` /
    ``skills_add`` / ``description_append``) and the native HarnessProfileConfig
    fields (``base_system_prompt`` / ``system_prompt_suffix`` /
    ``tool_description_overrides`` / ``excluded_tools``), honored per-agent via
    ``orgs._agent_profile_from_spec``. The old ``profile.yaml`` ``subagents:``
    block was a SECOND partial-override surface with its own resolver — exactly
    the dual-read the fold killed. A future re-introduction (someone re-adds the
    block because it "feels simpler") is a HARD contract failure, not a silent
    regression — mirroring ``no-legacy-middleware-in-graph`` /
    ``no-legacy-org-roster``.

    (A malformed profile is skipped here — ``profile-schema`` already reports
    it, no double-noise. And since the block is no longer peeled in
    ``profile.load_profile``, ``HarnessProfileConfig.from_dict`` ALSO rejects
    the unknown key, so the block fails twice over.)"""
    v: list[Violation] = []
    for org in discover_orgs():
        try:
            data = profile_mod._read_profile_yaml(org)
        except (TypeError, ValueError, yaml.YAMLError):
            continue  # profile-schema reports a malformed profile; don't double-noise
        if data and "subagents" in data:
            v.append(Violation(
                "error", "no-legacy-subagents-block",
                f"{org}/profile.yaml: the top-level 'subagents:' block was "
                f"removed — per-agent overrides now live "
                f"in the agent's OWN frontmatter (extends: + tools_add / "
                f"tools_remove / system_prompt_suffix / ...). Delete the block "
                f"and move the overrides into the agent .md frontmatter."))
    return v


def _scan_runtime_for_memory_saver(src: Path) -> list[Violation]:
    """AST-scan ONE runtime file for an imported or instantiated ``MemorySaver``.

    Returns one ``Violation`` per offending node so a file with both an import
    AND a call reports both. Pure + path-parameterised so the tripwire's
    provocation test can drive it against a temp file without touching the real
    ``acp.py`` / ``main.py``."""
    v: list[Violation] = []
    try:
        tree = ast.parse(src.read_text())
    except SyntaxError as exc:  # pragma: no cover - module is imported, so valid
        v.append(Violation(
            "error", "no-legacy-memory-saver",
            f"{src}: does not parse: {exc}"))
        return v
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {alias.name.split(".")[-1] for alias in node.names}
            if "MemorySaver" in names:
                v.append(Violation(
                    "error", "no-legacy-memory-saver",
                    f"{src}: imports MemorySaver — the server-side runtimes "
                    f"share threads.open_thread_store's persistent "
                    f"AsyncSqliteSaver; an ephemeral MemorySaver "
                    f"makes threads invisible across processes"))
        elif isinstance(node, ast.Call):
            func = node.func
            called = None
            if isinstance(func, ast.Name):
                called = func.id
            elif isinstance(func, ast.Attribute):
                called = func.attr
            if called == "MemorySaver":
                v.append(Violation(
                    "error", "no-legacy-memory-saver",
                    f"{src}: instantiates MemorySaver() — the server-side "
                    f"runtimes share threads.open_thread_store's persistent "
                    f"AsyncSqliteSaver; an ephemeral MemorySaver "
                    f"loses checkpoints when the process exits"))
    return v


def _no_legacy_memory_saver_in_runtimes() -> list[Violation]:
    """Permanent tripwire (no-legacy-left-behind): ``acp.py`` +
    ``main.py`` must NOT import or instantiate ``MemorySaver``.

    The server-side runtimes were unified onto ONE persistent
    ``AsyncSqliteSaver`` (``threads.open_thread_store``). Before the unification, ``acp.py``
    and ``main.py`` each minted an ephemeral ``MemorySaver()`` — checkpoints
    died with the process, threads were invisible to ``pux show``/``pux resume``,
    and the runtimes silently diverged. A future re-introduction (someone wires
    ``MemorySaver`` back because it "feels simpler for a quick test") is a HARD
    contract failure, not a silent regression — mirroring
    ``no-legacy-middleware-in-graph`` / ``no-legacy-agent-py``.

    Scope: ``acp.py`` + ``main.py`` ONLY — NOT ``tui.py`` (dcode stays
    self-contained, separate store), NOT ``server.py`` (already on the shared
    store), NOT ``graph.py`` (factory pattern, middleware tripwire's domain),
    and NOT the test tree (tests legitimately construct ``MemorySaver``).

    Mechanism: AST-scan (via ``_scan_runtime_for_memory_saver``) the two files
    for (a) any import whose bound name is ``MemorySaver`` (covers ``import
    langgraph...MemorySaver`` and ``from langgraph... import MemorySaver``, even
    aliased) and (b) any call to ``MemorySaver(...)`` (``Name`` or ``Attribute``
    form). AST (not regex) so a commented-out line or a docstring mention
    doesn't trip a false positive."""
    runtimes_dir = Path(__file__).resolve().parent.parent  # harness/pux_harness/
    v: list[Violation] = []
    for name in ("acp.py", "main.py"):
        src = runtimes_dir / name
        if src.is_file():
            v.extend(_scan_runtime_for_memory_saver(src))
    return v


# Consolidation — the 7 ``orgs.py`` functions that must be thin
# delegates to ``pux_harness.kit.loaders``. ``load_root_prompt`` is INTENTIONALLY
# exempt (PROJECT_ROOT-pinned root prompt, NOT the ``_orgs_dir()`` seam — see
# its docstring). Re-implementing any of these here re-creates the verbatim
# drift the consolidation removed.
_DELEGATED_ORGS_LOADERS: frozenset[str] = frozenset({
    "_org_path",
    "_agent_search_dirs",
    "discover_orgs",
    "org_agent_slugs",
    "load_org_prompt",
    "_resolve_skills",
    "_load_agent_spec",
})


def _scan_orgs_for_duplicate_loaders(src: Path) -> list[Violation]:
    """AST-scan ONE ``orgs.py``-shaped source for the 7 delegated loaders being
    real implementations instead of thin delegates to ``pux_harness.kit.loaders``.

    Pure + path-parameterised so the tripwire's provocation test can drive it
    against a temp file without touching the real ``orgs.py`` — mirroring
    ``_scan_runtime_for_memory_saver``. See ``_no_duplicate_loaders_in_orgs``
    for the policy."""
    v: list[Violation] = []
    try:
        tree = ast.parse(src.read_text())
    except SyntaxError as exc:  # pragma: no cover - module is imported, so valid
        v.append(Violation("error", "no-duplicate-loaders-in-orgs",
                           f"{src}: does not parse: {exc}"))
        return v

    # The local alias bound to the delegated loaders (``pux_harness.kit`` — the
    # slim in-tree core; the kit lives in-package, no separate package). The old
    # ``pux_agentkit`` package was FOLDED in (Stage 1) and deleted, so importing
    # it is no longer possible — the tripwire enforces the ONE remaining path.
    # ImportFrom form: ``from pux_harness.kit import loaders as <alias>``;
    # Import form: ``import pux_harness.kit.loaders as <alias>``.
    kit_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pux_harness.kit":
            for alias in node.names:
                if alias.name == "loaders":
                    kit_aliases.add(alias.asname or "loaders")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pux_harness.kit.loaders" and alias.asname:
                    kit_aliases.add(alias.asname)
    if not kit_aliases:
        v.append(Violation(
            "error", "no-duplicate-loaders-in-orgs",
            f"{src}: no longer imports loaders from pux_harness.kit — "
            f"the pure org/agent loaders must be delegated to it; "
            f"re-importing them as a verbatim copy re-creates the drift"))
        return v

    defined: dict[str, ast.FunctionDef] = {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in _DELEGATED_ORGS_LOADERS:
        fn = defined.get(name)
        if fn is None:
            v.append(Violation(
                "error", "no-duplicate-loaders-in-orgs",
                f"{src}: delegated loader {name!r} is missing — it must exist "
                f"as a thin delegate to pux_harness.kit.loaders"))
            continue
        delegates = False
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call):
                func = sub.value.func
                if (isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id in kit_aliases):
                    delegates = True
                    break
        if not delegates:
            v.append(Violation(
                "error", "no-duplicate-loaders-in-orgs",
                f"{src}: {name!r} must be a thin delegate to "
                f"pux_harness.kit.loaders (a ``return <alias>.<method>(...)``); "
                f"re-implementing it here re-creates the duplicated-loader "
                f"drift that was removed"))
    return v


def _no_duplicate_loaders_in_orgs() -> list[Violation]:
    """Permanent tripwire (consolidation; no-legacy-left-behind): the 7
    pure org/agent loaders in ``orgs.py`` must stay THIN DELEGATES to
    ``pux_harness.kit.loaders``.

    The consolidation finished (and Stage 1 folded the kit in-tree):
    the parse/discovery/roster/prompt/spec/skills logic lives ONCE in
    ``pux_harness.kit.loaders`` (the slim in-package core), and ``orgs.py``
    forwards to it through the injectable ``_orgs_dir()`` seam
    (``project_root = _orgs_dir().parent``). Before the consolidation
    the bodies were duplicated verbatim. A future re-paste of the old logic into ``orgs.py``
    (someone "simplifies" a delegate back into a real implementation) is a HARD
    contract failure, not a silent regression — mirroring
    ``no-legacy-middleware-in-graph`` / ``no-legacy-agent-py``.

    Mechanism: AST-scan ``orgs.py`` (via ``_scan_orgs_for_duplicate_loaders``).
    Each delegated-name ``FunctionDef`` must ``return`` an
    ``<alias>.<method>(...)`` call, where ``<alias>`` is the local name bound to
    ``pux_harness.kit.loaders``. AST (not regex) so a docstring or comment mention
    doesn't false-trip, and a delegate that silently stopped delegating (no kit
    call) DOES trip. ``load_root_prompt`` is exempt — it deliberately reads the
    root AGENTS.md from ``PROJECT_ROOT`` (not the seam), so it is NOT a delegate
    (delegating it would change ``build_system_prompt`` under a tempdir-patched
    ``_orgs_dir``)."""
    src = Path(__file__).with_name("orgs.py")
    if not src.is_file():
        return []  # pragma: no cover - orgs.py is imported, so present
    return _scan_orgs_for_duplicate_loaders(src)


# Stage 2 (import hygiene) — the slim kit core (``pux_harness/kit/**`` + the
# top-level ``pux_harness/__init__.py``) must NOT import any heavy runtime
# module. The kit is the Docker-free portable core; these deps attach to
# ``pux-harness`` as optional extras (Stage 3: ``[sandbox]``, ``[browser]``,
# ``[server]``, ``[mcp]``). A bare ``pip install pux-harness`` must import the
# kit clean, and ``from pux_harness import compile_org`` must pull neither the
# sandbox/server/browser stack nor any sibling ``pux_harness`` subsystem.
#
# Roots grouped by the Stage-3 extra they will live behind:
_HEAVY_MODULE_ROOTS: frozenset[str] = frozenset({
    "docker",                                  # [sandbox]
    "selenium", "seleniumbase",                # [browser]
    "fastapi", "uvicorn", "starlette",         # [server] HTTP runtime
    "ag_ui_langgraph",                         # [server] AG-UI transport
    "fastmcp", "langchain_mcp_adapters",       # [mcp] MCP servers/adapters
})


def _resolve_relative_import(level: int, module: str | None,
                              pkg_parts: list[str]) -> str:
    """Resolve a relative ``from . import ...`` / ``from .x import ...`` to its
    absolute dotted name. ``pkg_parts`` is the dotted package the SOURCE FILE
    lives in (so for ``kit/compile.py`` → ``["pux_harness", "kit"]``; for
    ``kit/__init__.py`` → also ``["pux_harness", "kit"]`` — the init IS the
    package). ``level=1`` == current package, ``level=2`` == parent (drop 1)."""
    base = pkg_parts[: len(pkg_parts) - (level - 1)] if level >= 1 else list(pkg_parts)
    if module:
        base = base + module.split(".")
    return ".".join(base)


def _scan_for_heavy_imports(src: Path, pkg_parts: list[str]) -> list[Violation]:
    """AST-scan ONE kit-core source file for imports the slim core must not make.

    Denied (both eager and lazy — ``ast.walk`` descends into function bodies;
    the kit must not reference these AT ALL, not even deferred):
      * any ``_HEAVY_MODULE_ROOTS`` root (docker, fastapi, …);
      * any ``pux_harness.<sub>`` import where ``<sub> != "kit"`` — the kit must
        not reach into a sibling subsystem (sandbox/agent/context/server/...).
    Relative imports are resolved to absolute first so a within-kit ``from
    .loaders import ...`` (resolves to ``pux_harness.kit.loaders``) is NOT
    mistaken for a leak."""
    v: list[Violation] = []
    try:
        tree = ast.parse(src.read_text())
    except SyntaxError as exc:  # pragma: no cover - kit is imported, so valid
        v.append(Violation("error", "kit-import-isolation",
                           f"{src}: does not parse: {exc}"))
        return v

    def _check(name: str) -> None:
        root = name.split(".")[0]
        if root in _HEAVY_MODULE_ROOTS:
            v.append(Violation(
                "error", "kit-import-isolation",
                f"{src}: imports heavy module {name!r} — the slim kit core must "
                f"not depend on it (it will be a Stage-3 optional extra); "
                f"lazy-import it from the heavy subsystem instead"))
        if name.startswith("pux_harness.") and name.split(".")[1] != "kit":
            v.append(Violation(
                "error", "kit-import-isolation",
                f"{src}: imports sibling subsystem {name!r} — the kit must not "
                f"reach outside pux_harness.kit (keeps `from pux_harness import "
                f"compile_org` slim)"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                resolved = _resolve_relative_import(node.level, node.module, pkg_parts)
            else:
                resolved = node.module or ""
            if resolved and resolved != "__future__":
                _check(resolved)
    return v


def _kit_import_isolation() -> list[Violation]:
    """Permanent tripwire (Stage 2 import hygiene; no-legacy-left-behind): the
    slim kit core — ``pux_harness/__init__.py`` + every ``pux_harness/kit/*.py``
    — must import NEITHER a heavy runtime module (``docker``/``fastapi``/…) NOR
    a sibling ``pux_harness`` subsystem (``sandbox``/``agent``/``context``/...).

    Why permanent: the kit is the portable, Docker-free core a DIFFERENT project
    imports via ``from pux_harness import compile_org``. A future edit that
    adds ``import docker`` (or ``from pux_harness.sandbox import ...``) to the
    kit would silently couple the slim core back to the heavy run-time and
    break a bare ``pip install pux-harness``. The runtime isolation test
    (``tests/test_kit_compile.py::test_import_isolation_no_docker_no_heavy_subsystem``)
    proves the import GRAPH is clean today; this tripwire keeps the SOURCE from
    re-introducing a heavy import — mirroring ``no-legacy-middleware-in-graph``
    / ``no-duplicate-loaders-in-orgs``.

    Mechanism: AST-scan each kit-core file (via ``_scan_for_heavy_imports``).
    The allowed surface is stdlib + the declared core deps (``yaml``,
    ``deepagents``, ``langchain_core``, ``langgraph``) + within-kit relatives —
    everything that resolves to ``pux_harness.kit.*``. Stage 3 will split the
    heavy roots into extras; this tripwire is the precondition that proves the
    split is safe."""
    pkg_root = Path(__file__).resolve().parent.parent  # harness/pux_harness/
    targets: list[tuple[Path, list[str]]] = [
        (pkg_root / "__init__.py", ["pux_harness"]),
    ]
    kit_dir = pkg_root / "kit"
    if kit_dir.is_dir():
        for py in sorted(kit_dir.glob("*.py")):
            targets.append((py, ["pux_harness", "kit"]))
    v: list[Violation] = []
    for src, pkg_parts in targets:
        if src.is_file():
            v.extend(_scan_for_heavy_imports(src, pkg_parts))
    return v


def _scan_for_profile_registration(src: Path) -> list[Violation]:
    """AST-scan ONE source file for a CALL to ``register_harness_profile`` /
    ``register_provider_profile``. Pure + path-parameterised so the tripwire's
    provocation test can drive it against a temp file without touching the real
    package. Returns one ``Violation`` per offending call node."""
    v: list[Violation] = []
    banned = {"register_harness_profile", "register_provider_profile"}
    try:
        tree = ast.parse(src.read_text())
    except SyntaxError as exc:  # pragma: no cover - package modules parse
        v.append(Violation(
            "error", "no-harness-profile-registration",
            f"{src}: does not parse: {exc}"))
        return v
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else None)
        if name in banned:
            v.append(Violation(
                "error", "no-harness-profile-registration",
                f"{src}: calls {name}() — pux stays off the model-keyed "
                f"_HARNESS_PROFILES registry (multi-org server collision; no "
                f"unregister). Apply HarnessProfileConfig fields directly in "
                f"stack.build_stack / orgs.load_subagents instead."))
    return v


def _no_harness_profile_registration() -> list[Violation]:
    """Permanent tripwire (registry parity; no-legacy-left-behind): no
    file under ``pux_harness/`` may CALL deepagents' ``register_harness_profile``
    or ``register_provider_profile``.

    pux deliberately stays OFF the model-keyed ``_HARNESS_PROFILES`` registry:
    two orgs resolved under one model would merge-collide, the long-lived server
    builds many orgs per process, and there is no ``unregister``. pux applies
    ``HarnessProfileConfig`` fields ITSELF (``build_stack`` for the supervisor,
    ``load_subagents`` per specialist). The parity guarantee — pux
    middleware is never stripped by deepagents' own ``_apply_excluded_middleware``
    (which only fires through a REGISTERED profile) — depends on pux NEVER
    registering. A future re-introduction (someone "fixes" a profile gap by
    registering) is a HARD contract failure, not a silent regression — mirroring
    ``no-legacy-middleware-in-graph`` / ``no-duplicate-loaders-in-orgs``.

    Mechanism: AST-scan (via ``_scan_for_profile_registration``) every
    ``pux_harness/**/*.py`` for a CALL whose function name (bare or attribute) is
    one of the banned names. AST (not regex) so a docstring/comment mention
    doesn't trip a false positive, and an aliased import
    (``from x import register_harness_profile as reg``) still trips because the
    CALL node's name is what's matched."""
    pkg_root = Path(__file__).resolve().parent.parent  # .../pux_harness/
    v: list[Violation] = []
    for src in sorted(pkg_root.rglob("*.py")):
        v.extend(_scan_for_profile_registration(src))
    return v


def _no_load_skill_tool() -> list[Violation]:
    """Permanent tripwire (skills-peeking unification;
    no-legacy-left-behind): the ``pux_sandbox_load_skill`` specialist is GONE.
    Skill bodies are peeked via the native ``read_file`` — the canonical
    deepagents path, now that native ``SkillsMiddleware`` on the supervisor
    advertises each skill's name + description (progressive disclosure). The
    host-side ``pux_sandbox_list_skills`` stays as a CTO discovery catalog; it
    COMPLEMENTS the middleware, it does not duplicate the body-load.

    A re-introduction (someone re-adds a ``ToolSpec("load_skill", ...)`` to the
    ``REGISTRY``) is a HARD contract failure, not a silent regression. The check
    reads ``SPECIALIST_TOOL_NAMES`` — the frozenset of PREFIXED names
    (``pux_sandbox_*``) DERIVED from the registry — so it fires the instant
    ``load_skill`` re-enters the registry without a source scan or AST (the
    registry IS the single source of tool truth; a tool not in it is not a
    specialist surface)."""
    if "pux_sandbox_load_skill" in SPECIALIST_TOOL_NAMES:
        return [Violation(
            "error", "skills-peek-via-read-file",
            "pux_sandbox_load_skill was re-added to the tool REGISTRY — skill "
            "bodies are peeked via the native read_file now (SkillsMiddleware "
            "advertises name+description; list_skills is the catalog). Remove "
            "the load_skill ToolSpec.",
        )]
    return []


def _pux_namespace_resolvable() -> list[Violation]:
    """Every ``pux:`` reference resolves against the shipped library bases or
    ``$PUX_ORG_PATHS`` (the cross-project reuse contract). A dangling
    ``pux:`` is a HARD error: the namespace is the only way to pull a shipped
    org/agent without vendoring, so an unresolved one would silently fall back
    to nothing (the local search never matches a ``pux:`` token).

    Scans three surfaces for ``pux:`` tokens:

    1. each org's ``org.yaml extends:`` — a ``pux:<base>`` library base parent
       (must resolve to ``kit/bases/<base>/``);
    2. each org's effective roster ``agents:`` — a ``pux:<slug>`` library agent
       (must resolve to some base's ``agents/<slug>.md``);
    3. each agent ``.md`` frontmatter ``extends:`` — a ``pux:<slug>`` base agent
       (same resolution).

    The org-extends case (1) is ALSO caught by ``org-extends-resolvable`` with a
    generic message; this rule adds the precise namespace diagnosis + covers the
    agent-slug cases (2/3) no other rule reaches."""
    v: list[Violation] = []
    bases = _paths.library_bases_dir()
    # (1) + (2): org extends + roster agent slugs.
    for org in discover_orgs():
        parent = org_extends(org)
        if _paths.is_pux_namespace(parent) and not (bases / _paths.strip_namespace(parent)).is_dir():
            v.append(Violation(
                "error", "pux-namespace-resolvable",
                f"{org}: extends {parent!r} -> no such library base "
                f"(kit/bases/{_paths.strip_namespace(parent)}/ not found)"))
        for slug in org_agent_slugs(org):
            if _paths.is_pux_namespace(slug) and _paths.resolve_library_agent(slug) is None:
                v.append(Violation(
                    "error", "pux-namespace-resolvable",
                    f"{org}: roster agent {slug!r} -> no such library agent "
                    f"(no kit/bases/*/agents/{_paths.strip_namespace(slug)}.md)"))
    # (3): agent frontmatter extends — scan every agent source file.
    agent_dirs = [_orgs_dir() / "_shared" / "agents"]
    for org in discover_orgs():
        try:
            agent_dirs.append(_org_path(org) / "agents")
        except FileNotFoundError:
            continue
    for adir in agent_dirs:
        if not adir.is_dir():
            continue
        for md in sorted(adir.glob("*.md")):
            fm, _ = _split_frontmatter(md.read_text())
            ext = fm.get("extends")
            if _paths.is_pux_namespace(ext) and _paths.resolve_library_agent(ext) is None:
                v.append(Violation(
                    "error", "pux-namespace-resolvable",
                    f"{md.name}: extends {ext!r} -> no such library agent "
                    f"(no kit/bases/*/agents/{_paths.strip_namespace(ext)}.md)"))
    return v


def check_harness() -> list[Violation]:
    """Rule 6 (no hardcoded org->agent manifest) + rule 7 (no orphan agents)
    + permanent legacy tripwires. Global — not per-org."""
    v: list[Violation] = []
    src = Path(__file__).with_name("orgs.py")
    if src.is_file() and _MANIFEST_RE.search(src.read_text()):
        v.append(Violation("error", "no-hardcoded-manifest",
                           "pux_harness/orgs.py: a hardcoded ORG_AGENTS "
                           "manifest re-couples the harness to orgs; use the "
                           "`agents:` frontmatter + discover_orgs() instead"))
    for orphan in orphan_agents():
        v.append(Violation("warn", "no-orphan-agents",
                           f"agent {orphan!r} is owned by no org (not in any "
                           f"`agents:` frontmatter)"))
    # The shipped ``models.yaml`` (the role-spec single source of
    # truth) must be present + well-formed — every model consumer resolves
    # through it, so a missing/malformed spec breaks every org at once.
    try:
        model_mod.validate_models_spec()
    except RuntimeError as exc:
        v.append(Violation("error", "models-spec",
                           f"models.yaml invalid: {exc}"))
    v.extend(_no_legacy_agent_py())
    v.extend(_no_legacy_sandbox_artifacts())
    v.extend(_no_legacy_middleware_in_graph())
    v.extend(_no_legacy_memory_saver_in_runtimes())
    v.extend(_no_legacy_subagents_block())
    v.extend(_no_duplicate_loaders_in_orgs())
    v.extend(_kit_import_isolation())
    v.extend(_no_harness_profile_registration())
    v.extend(_no_load_skill_tool())
    v.extend(_pux_namespace_resolvable())
    return v


# --- rule 8 — global skill hygiene ---------------------------------------

def _check_skill_dir(skill_dir: Path) -> list[Violation]:
    """Agent-Spec well-formedness of one ``<source>/<name>/`` skill dir.

    Well-formed == it contains a ``SKILL.md`` whose YAML frontmatter parses,
    whose ``name`` equals the dir name (kebab-case), and whose ``description``
    is non-empty (the spec requires both, and SkillsMiddleware needs the
    ``description`` for level-1 metadata discovery). Returns one
    ``skill-well-formed`` error per failure; an empty list means the skill is
    well-formed.
    """
    name = skill_dir.name
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return [Violation("error", "skill-well-formed",
                          f"skill {name!r}: missing SKILL.md "
                          f"(expected <source>/<name>/SKILL.md)")]
    try:
        fm, _ = _split_frontmatter(skill_md.read_text())
    except ValueError as e:
        return [Violation("error", "skill-well-formed",
                          f"skill {name!r}: SKILL.md frontmatter does not "
                          f"parse: {e}")]
    out: list[Violation] = []
    if fm.get("name") != name:
        out.append(Violation("error", "skill-well-formed",
                             f"skill {name!r}: frontmatter name "
                             f"{fm.get('name')!r} must equal the dir name "
                             f"{name!r}"))
    if not _SKILL_NAME_RE.match(name):
        out.append(Violation("error", "skill-well-formed",
                             f"skill {name!r}: dir name must be kebab-case "
                             f"(lowercase letters/digits joined by '-')"))
    if not fm.get("description"):
        out.append(Violation("error", "skill-well-formed",
                             f"skill {name!r}: SKILL.md missing a non-empty "
                             f"'description' (required for skills-middleware "
                             f"discovery)"))
    return out


def _well_formed_skill_dirs(source: Path) -> list[Path]:
    """Skill dirs directly under ``source`` that pass well-formedness.

    Used by ``skill-source-resolves`` to require a declared source carry at
    least one real skill (a source of only malformed/empty dirs silently loads
    nothing)."""
    if not source.is_dir():
        return []
    return [c for c in sorted(source.iterdir())
            if c.is_dir() and not _check_skill_dir(c)]


def _skill_roots() -> list[Path]:
    """Every skills-ROOT directory in the project: each ``orgs/<name>/skills``
    (the ``*/skills`` glob matches ``_shared`` too, so ``orgs/_shared/skills``
    is covered). Scanned regardless of whether any agent declares the root — a
    loose playbook or malformed skill is a regression even if undeclared."""
    orgs = _orgs_dir()
    return [p for p in sorted(orgs.glob("*/skills")) if p.is_dir()]


def check_skill_roots() -> list[Violation]:
    """Global skill hygiene (rule 8). Scans EVERY skills root in the project
    whether or not an agent declares it:

    * each ``<root>/<name>/SKILL.md`` is Agent-Spec well-formed
      (``skill-well-formed`` error);
    * no ``.md`` sits loose directly under a root (``skill-dir-not-loose``
      warn) — a loose playbook is invisible to SkillsMiddleware, the exact
      regression that stranded the org playbooks before this rule.

    The contract CLI runs this alongside ``check_harness``; the per-org pass
    (``skill-source-resolves``) guards declared sources, this one guards the
    filesystem as a whole.
    """
    v: list[Violation] = []
    for root in _skill_roots():
        rel_root = root.relative_to(PROJECT_ROOT)
        for child in sorted(root.iterdir()):
            if child.is_dir():
                v.extend(_check_skill_dir(child))
            elif child.suffix == ".md":
                v.append(Violation(
                    "warn", "skill-dir-not-loose",
                    f"{rel_root}/{child.name}: loose .md under a skills root "
                    f"is invisible to SkillsMiddleware — move it into a "
                    f"<skill-name>/ dir (or its references/)."))
    return v


def check_all() -> dict[str, list[Violation]]:
    """Per-org violations for every discovered org. Global checks live in
    ``check_harness()``; the CLI runs both."""
    return {org: check_org(org) for org in discover_orgs()}


def has_errors(violations: list[Violation]) -> bool:
    return any(x.severity == "error" for x in violations)
