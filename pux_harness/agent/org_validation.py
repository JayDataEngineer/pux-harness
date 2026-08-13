"""Declarative org validation — the pydantic config layer for ``orgs/<name>/``.

An org is a directory ``orgs/<name>/`` containing ``AGENTS.md`` (CTO prompt
prose), ``org.yaml`` (roster + extends), and optionally ``policy.yaml`` /
``profile.yaml`` / ``skills/`` / ``agents/<slug>.md``. This module is the
fail-loud validation boundary: loading an org validates it.

Two entry points, one walk — and the same strict/audit contract applied to the
profile surface:

* ``load_org(name)`` / ``load_profile(name)`` — STRICT. Validate and raise
  ``OrgContractError`` (carrying every error) on the first error. This is the
  enforcement mode — used by the ``pux check-contract`` gate and available to
  the server for fail-fast boot.
* ``audit_org(name)`` / ``audit_profile(name)`` — AUDIT. Run the identical
  walk, return the problems instead of raising. This is the diagnostic mode
  for the CLI report.

The validation is fully offline (no server, no model tokens): it reads the
same files through the same loaders the runtime uses (``kit.loaders`` /
``orgs.py`` / ``policy`` / ``profile``), so the offline check and the runtime
cannot disagree.

The config SURFACES (org.yaml, agent frontmatter, SKILL.md) are pydantic
models; the assembly (``load_org`` / ``audit_org``) validates the
cross-file contract (extends chains, roster resolution, tool whitelists,
policy/profile schema). Repo-wide fleet checks — legacy tripwires, import
hygiene, orphan agents, skill-root hygiene — live in the test suite
(``tests/harness/tripwire_checks.py``), never in shipped code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pux_harness.sandbox import policy as policy_mod
from pux_harness.agent import profile as profile_mod
from pux_harness.agent import stack as stack_mod
from pux_harness.agent import tool_servers as tool_servers_mod
from pux_harness.sandbox.tools import declared as declared_mod
from pux_harness.sandbox.tools import dynamic as dynamic_mod
from pux_harness.sandbox.tools import (
    classify_slug,
    prefixed,
    Category,
)
from pux_harness.kit.capabilities_decl import (
    CapabilitiesSugarError,
    desugar_agent_capabilities,
)
from pux_harness.agent.orgs import (
    _agent_search_dirs,
    _load_agent_spec,
    _org_path,
    _orgs_dir,
    _parse_list,
    _split_frontmatter,
    org_agent_slugs,
    org_extends,
)

# --- the problem vocabulary -------------------------------------------------

# Every agent ``<slug>.md`` must carry these (name + description from
# frontmatter; system_prompt = the body).
_REQUIRED_AGENT_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "description",
        "system_prompt",
    }
)

# Optional ``orgs/<name>/policy.yaml`` top-level sections. ``host_setup`` is
# harness-added (no Go equivalent) — the host-side prep-hook list.
# ``build`` is a sub-key under ``sandbox``, NOT a top-level section.
KNOWN_POLICY_SECTIONS: frozenset[str] = frozenset(
    {
        "workspace",
        "egress",
        "credentials",
        "sandbox",
        "browser",
        "host_setup",
        "jobs",
        "tool_servers",
        "protocols",
        "tool_surface",
    }
)

# Agent-Skills spec: a skill dir name (and its ``SKILL.md`` ``name``) must be
# kebab-case — lowercase letters/digits joined by single hyphens.
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Problem:
    """One validation finding. ``severity`` is "error" (blocks) or "warn".

    ``kind`` (optional) tags the problem with the unified capability taxonomy
    (``tool`` / ``skill`` / ``mcp`` / ``middleware`` / ``job``) so the
    capability-channel violations report under ONE taxonomy. ``None`` = not a
    capability-channel violation (org-structural, policy-section, etc.)."""

    severity: str  # "error" | "warn"
    rule: str
    message: str
    kind: str | None = None

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.rule}: {self.message}"


class OrgContractError(Exception):
    """Raised by ``load_org`` (strict mode) when the org bundle has errors.

    Carries every problem found in one walk so a caller can report them all —
    ``problems`` contains errors AND warns; only errors trigger the raise."""

    def __init__(self, org: str, problems: list[Problem]) -> None:
        super().__init__(
            f"org {org!r} failed validation with "
            f"{sum(1 for p in problems if p.severity == 'error')} error(s)"
        )
        self.org = org
        self.problems = problems


# --- declarative config surfaces (pydantic) ---------------------------------


class OrgConfig(BaseModel):
    """``org.yaml`` — the org's declarative declaration. Shape-validates the
    roster, extends, deny-list and capability declarations. Cross-file checks
    (extends resolves, roster slugs resolve to real agents) live in the
    assembly. ``extra="allow"`` by design: org.yaml carries fields this
    validator doesn't enforce (``inherit_roster``, ``extends``, …) and the
    org's own data is not a hostile client boundary."""

    model_config = ConfigDict(extra="allow", frozen=True)

    extends: str | None = None
    agents: list[str] = Field(default_factory=list)
    roster_deny: list[str] = Field(default_factory=list)
    capabilities: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("extends", mode="before")
    @classmethod
    def _extends_str(cls, v: Any) -> Any:
        if v is None:
            return None
        if not isinstance(v, str):
            raise TypeError(f"extends must be an org slug, got {type(v).__name__}")
        return v.strip() or None

    @field_validator("agents", "roster_deny", mode="before")
    @classmethod
    def _slug_list(cls, v: Any) -> Any:
        if v is None:
            return []
        if not isinstance(v, list):
            raise TypeError(f"expected a list of slugs, got {type(v).__name__}")
        return v


class AgentMeta(BaseModel):
    """The RAW frontmatter of one ``agents/<slug>.md`` — shape-validated.

    ``name``/``description``/``system_prompt`` presence is enforced by the
    assembly (``agent-missing-keys``) so the exact missing keys surface in one
    walk; the model here pins the types of every field the loader reads."""

    model_config = ConfigDict(extra="allow", frozen=True)

    name: str | None = None
    description: str | None = None
    extends: str | None = None
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    middleware: list[str] = Field(default_factory=list)
    capabilities: list[dict[str, Any]] | None = None

    @field_validator("tools", "skills", "middleware", mode="before")
    @classmethod
    def _str_list(cls, v: Any) -> Any:
        if v is None:
            return []
        if not isinstance(v, list):
            raise TypeError(f"expected a list of strings, got {type(v).__name__}")
        return v


class SkillMeta(BaseModel):
    """One skill's ``SKILL.md`` frontmatter — Agent-Spec well-formedness.
    ``name`` must equal the directory name (kebab-case); ``description`` is
    required (SkillsMiddleware needs it for level-1 metadata discovery)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str

    @field_validator("name")
    @classmethod
    def _kebab(cls, v: str) -> str:
        if not _SKILL_NAME_RE.match(v):
            raise ValueError(
                f"skill name {v!r} must be kebab-case (lowercase letters/digits "
                f"joined by single hyphens)"
            )
        return v

    @field_validator("description")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("skill description must be non-empty")
        return v


# --- pure leaf cores ---------------------------------------------------------


def _declared_exec_guard_violation(
    declares_tools: bool,
    routing_removed: bool,
    org: str,
) -> list[Problem]:
    """Pure core of the exec-guard rule (no I/O — directly testable).

    Fires a WARN iff the org declares sandbox tools AND has removed the
    ``routing`` middleware from its supervisor. The exec-guard — a declared
    script is taken out of ``execute`` and reached via its typed
    ``pux_sandbox_*`` tool — is wired through ``RoutingMiddleware`` (default-on
    in ``DEFAULT_SUPERVISOR``). Removing routing silently turns the guard OFF
    and re-opens the dual-path (a declared script callable BOTH via the typed
    tool AND raw ``execute``), defeating the declared-tool invariant. A WARN,
    not an error: an org may legitimately shed routing (e.g. a test profile),
    but when it also declares tools that choice deserves to be explicit."""
    if declares_tools and routing_removed:
        return [
            Problem(
                "warn",
                "declared-exec-guard",
                f"{org}: declares sandbox tools but removes the 'routing' middleware "
                f"from the supervisor — the exec-guard (a declared script is taken "
                f"out of execute and reached via its typed tool) is wired through "
                f"RoutingMiddleware, so this re-opens the dual-path. Keep routing, "
                f"or confirm this org should allow raw exec of its declared scripts.",
            )
        ]
    return []


def check_skill_dir(skill_dir: Path) -> list[Problem]:
    """Agent-Spec well-formedness of one ``<source>/<name>/`` skill dir via the
    ``SkillMeta`` pydantic model.

    Well-formed == it contains a ``SKILL.md`` whose YAML frontmatter parses
    into ``SkillMeta`` (name == dir name, kebab-case, description non-empty).
    Returns one ``skill-well-formed`` error per failure; an empty list means
    the skill is well-formed."""
    name = skill_dir.name
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return [
            Problem(
                "error",
                "skill-well-formed",
                f"skill {name!r}: missing SKILL.md (expected <source>/<name>/SKILL.md)",
            )
        ]
    try:
        fm, _ = _split_frontmatter(skill_md.read_text())
    except ValueError as e:
        return [
            Problem(
                "error",
                "skill-well-formed",
                f"skill {name!r}: SKILL.md frontmatter does not parse: {e}",
            )
        ]
    if fm.get("name") != name:
        return [
            Problem(
                "error",
                "skill-well-formed",
                f"skill {name!r}: frontmatter name {fm.get('name')!r} must equal "
                f"the dir name {name!r}",
            )
        ]
    try:
        SkillMeta.model_validate(fm)
    except Exception as exc:  # pydantic ValidationError (name/description rules)
        return [Problem("error", "skill-well-formed", f"skill {name!r}: SKILL.md invalid: {exc}")]
    return []


# --- per-org validation walk -------------------------------------------------


def _load_agent_subagent(slug: str, org: str) -> dict[str, Any] | None:
    """Read ``<slug>.md`` (org-local then ``_shared``) -> spec dict, or ``None``.

    Delegates to ``orgs._load_agent_spec`` (single source of truth — the runtime
    loader and the validation read the SAME file). Returns ``None`` if no
    ``<slug>.md`` exists; a malformed frontmatter raises ``ValueError`` (caught
    + reported by the walker)."""
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


def _agent_extends_chain_problems(slug: str, org: str) -> list[Problem]:
    """Validate an agent's ``extends:`` chain resolves + is acyclic.

    Reads RAW frontmatter (NO merge) and walks the chain manually so the two
    dedicated rules fire with precise, actionable messages, independent of
    ``_load_agent_spec``'s own recursion (which raises + would surface as a
    generic ``agent-resolves``). A chain that references a non-existent agent
    fires ``agent-extends-resolvable``; a cycle fires ``agent-extends-acyclic``.

    Returns ``[]`` for an agent with no ``extends:`` (the common case), a
    missing roster slug (``agent-resolves`` owns that), or unreadable
    frontmatter (``agent-resolves`` owns that)."""
    v: list[Problem] = []
    chain: list[str] = [slug]
    visited: set[str] = {slug}
    cur = slug
    while True:
        path = _first_agent_md(cur, org)
        if path is None:
            if cur != slug:
                v.append(
                    Problem(
                        "error",
                        "agent-extends-resolvable",
                        f"{org}/{slug}: extends chain references unknown agent "
                        f"{cur!r} (chain: {' -> '.join(chain)})",
                    )
                )
            return v
        try:
            fm, _ = _split_frontmatter(path.read_text())
        except ValueError:
            return v
        parent = fm.get("extends")
        if not isinstance(parent, str) or not parent.strip():
            return v
        parent = parent.strip()
        if parent in visited:
            v.append(
                Problem(
                    "error",
                    "agent-extends-acyclic",
                    f"{org}/{slug}: extends cycle detected ({' -> '.join(chain)} -> {parent})",
                )
            )
            return v
        chain.append(parent)
        visited.add(parent)
        cur = parent


def _capabilities_sugar_agent_problems(slug: str, org: str) -> list[Problem]:
    """Validate an agent's opt-in frontmatter ``capabilities:`` block — a
    dedicated rule so a malformed block fires the precise
    ``capabilities-sugar-agent`` message, independent of ``_load_agent_spec``'s
    own desugar (which raises + would surface as a generic ``agent-resolves``).
    Reads RAW frontmatter (NO merge)."""
    path = _first_agent_md(slug, org)
    if path is None:
        return []
    try:
        fm, _ = _split_frontmatter(path.read_text())
    except ValueError:
        return []
    if "capabilities" not in fm:
        return []
    try:
        desugar_agent_capabilities(dict(fm), slug)
    except CapabilitiesSugarError as exc:
        return [
            Problem(
                "error",
                "capabilities-sugar-agent",
                f"{org}/{slug}: malformed capabilities: block — {exc}",
            )
        ]
    return []


def _org_extends_chain_problems(name: str) -> list[Problem]:
    """Validate an org's ``extends:`` chain resolves + is acyclic.

    A parent that is no org, or an org without ``AGENTS.md`` (not a valid
    base), fires ``org-extends-resolvable``; a cycle fires
    ``org-extends-acyclic``. Returns ``[]`` for an org with no ``extends:``."""
    v: list[Problem] = []
    chain: list[str] = [name]
    visited: set[str] = {name}
    cur = name
    while True:
        parent = org_extends(cur)
        if parent is None:
            return v
        if parent in visited:
            v.append(
                Problem(
                    "error",
                    "org-extends-acyclic",
                    f"{name}: extends cycle detected ({' -> '.join(chain)} -> {parent})",
                )
            )
            return v
        try:
            pdir = _org_path(parent)
        except FileNotFoundError:
            v.append(
                Problem(
                    "error",
                    "org-extends-resolvable",
                    f"{name}: extends {parent!r} -> no such org "
                    f"(chain: {' -> '.join(chain)} -> {parent})",
                )
            )
            return v
        if not (pdir / "AGENTS.md").is_file():
            v.append(
                Problem(
                    "error",
                    "org-extends-resolvable",
                    f"{name}: extends {parent!r} -> {parent}/AGENTS.md missing "
                    f"(not a valid base org; chain: {' -> '.join(chain)} -> {parent})",
                )
            )
            return v
        chain.append(parent)
        visited.add(parent)
        cur = parent


def _validate_host_setup(name: str, pol: policy_mod.Policy) -> list[Problem]:
    """Offline validation of every host_setup hook: each has a name +
    helper_script; helper_script resolves under the project root and exists;
    export sources subset of {stdout}; python_deps is a list of strings."""
    v: list[Problem] = []
    hooks = policy_mod.host_setup_hooks(pol)
    if not hooks:
        return v
    project_root = _orgs_dir().parent
    for hook in hooks:
        hname = hook.name or "<unnamed>"
        if not hook.name:
            v.append(
                Problem("error", "host-setup-shape", f"{name}: host_setup hook missing 'name'")
            )
        if not hook.helper_script:
            v.append(
                Problem(
                    "error",
                    "host-setup-shape",
                    f"{name}/{hname}: host_setup hook missing 'helper_script'",
                )
            )
            continue
        script = Path(hook.helper_script)
        if not script.is_absolute():
            script = project_root / hook.helper_script
        if not script.is_file():
            v.append(
                Problem(
                    "error",
                    "host-setup-helper-missing",
                    f"{name}/{hname}: helper_script {hook.helper_script!r} not found at {script}",
                )
            )
        bad = sorted({s for s in hook.exports.values() if s not in {"stdout"}})
        if bad:
            v.append(
                Problem(
                    "error",
                    "host-setup-shape",
                    f"{name}/{hname}: unsupported export source(s) {bad}; allowed: ['stdout']",
                )
            )
        if not isinstance(hook.python_deps, list) or not all(
            isinstance(d, str) for d in hook.python_deps
        ):
            v.append(
                Problem(
                    "error",
                    "host-setup-shape",
                    f"{name}/{hname}: python_deps must be a list of strings",
                )
            )
    return v


def _validate_build_spec(name: str, pol: policy_mod.Policy) -> list[Problem]:
    """Offline validation of sandbox.build: dockerfile resolves under the
    project root and exists; context (if set) is a dir."""
    v: list[Problem] = []
    spec = policy_mod.build_spec(pol)
    if spec is None:
        return v
    project_root = _orgs_dir().parent
    dockerfile = Path(spec.dockerfile)
    if not dockerfile.is_absolute():
        dockerfile = project_root / spec.dockerfile
    if not dockerfile.is_file():
        v.append(
            Problem(
                "error",
                "sandbox-build-shape",
                f"{name}: sandbox.build dockerfile {spec.dockerfile!r} not found at {dockerfile}",
            )
        )
    if spec.context:
        context = Path(spec.context)
        if not context.is_absolute():
            context = project_root / spec.context
        if not context.is_dir():
            v.append(
                Problem(
                    "error",
                    "sandbox-build-shape",
                    f"{name}: sandbox.build context {spec.context!r} not found at {context}",
                )
            )
    return v


def _validate_jobs(name: str, pol: policy_mod.Policy) -> list[Problem]:
    """Offline validation of jobs: each has a name + script; script resolves
    under the project root and exists; timeout is a non-negative integer;
    names are unique."""
    v: list[Problem] = []
    specs = policy_mod.job_specs(pol)
    if not specs:
        return v
    project_root = _orgs_dir().parent
    seen_names: set[str] = set()
    for spec in specs:
        jname = spec.name or "<unnamed>"
        if not spec.name:
            v.append(Problem("error", "jobs-shape", f"{name}: job entry missing 'name'"))
        if spec.name in seen_names:
            v.append(Problem("error", "jobs-shape", f"{name}: duplicate job name {spec.name!r}"))
        seen_names.add(spec.name)
        if not spec.script:
            v.append(Problem("error", "jobs-shape", f"{name}/{jname}: job entry missing 'script'"))
            continue
        script = Path(spec.script)
        if not script.is_absolute():
            script = project_root / spec.script
        if not script.is_file():
            v.append(
                Problem(
                    "error",
                    "jobs-script-missing",
                    f"{name}/{jname}: script {spec.script!r} not found at {script}",
                )
            )
        if spec.timeout < 0:
            v.append(
                Problem(
                    "error",
                    "jobs-shape",
                    f"{name}/{jname}: timeout must be >= 0, got {spec.timeout}",
                )
            )
        if spec.when and not isinstance(spec.when, str):
            v.append(
                Problem(
                    "error",
                    "jobs-shape",
                    f"{name}/{jname}: 'when' must be a string "
                    f"(a shell predicate), got {type(spec.when).__name__}",
                )
            )
    return v


def _validate_tool_servers(name: str, pol: policy_mod.Policy) -> list[Problem]:
    """Offline validation of the ``tool_servers`` declaration in policy.yaml."""
    del pol
    v: list[Problem] = []
    for err in tool_servers_mod.validate_tool_servers(name):
        v.append(Problem("error", "tool-servers", err))
    return v


def _validate_protocols(name: str, pol: policy_mod.Policy) -> list[Problem]:
    """Offline validation of the ``protocols`` declaration. Each entry must be
    a known surface (``policy.KNOWN_PROTOCOLS``); an unknown entry is a typo."""
    v: list[Problem] = []
    allowed = sorted(policy_mod.KNOWN_PROTOCOLS)
    for proto in pol.protocols:
        if proto not in policy_mod.KNOWN_PROTOCOLS:
            v.append(
                Problem(
                    "error",
                    "protocols",
                    f"{name}: policy.yaml protocols entry {proto!r} is not a known "
                    f"surface; allowed: {allowed}",
                )
            )
    return v


def _validate_sandbox_tools(name: str) -> list[Problem]:
    """Offline validation of the org's declared sandbox tools
    (``sandbox/tools/tools.yaml``). No-op when the org declares no tools.yaml."""
    v: list[Problem] = []
    for err in declared_mod.validate_declared_tools(_org_path(name) / "sandbox"):
        v.append(Problem("error", "sandbox-tools", err))
    return v


def _validate_declared_exec_guard(name: str) -> list[Problem]:
    """Warn if the org declares sandbox tools but its profile removes
    ``routing`` from the supervisor stack (the exec-guard is silently off).
    Gathers BOTH removal paths — the harness ``middleware.supervisor.remove``
    block AND the deepagents ``excluded_middleware`` field."""
    if not declared_mod.declared_tool_names(_org_path(name) / "sandbox"):
        return []
    removed: set[str] = set()
    try:
        removed |= set(profile_mod.load_middleware_overrides(name).supervisor_remove)
    except (TypeError, ValueError):
        pass
    try:
        prof = profile_mod.load_profile(name)
    except (TypeError, ValueError, yaml.YAMLError):
        prof = None
    if prof is not None and prof.excluded_middleware:
        removed |= set(prof.excluded_middleware)
    return _declared_exec_guard_violation(True, "routing" in removed, name)


def _validate_capabilities(
    name: str,
    pol: policy_mod.Policy | None,
) -> list[Problem]:
    """ONE pass over the org's capability channels — the unified front-door.
    Delegates to the leaf validators, then tags every resulting ``Problem``
    with the unified ``kind`` taxonomy (``tool | skill | mcp | middleware |
    job``) so the report speaks one vocabulary.

    Wraps the model add-on channels only:
    - ``_validate_sandbox_tools`` (kind=tool) + ``_validate_declared_exec_guard``
      (kind=tool): run for every org.
    - ``_validate_jobs`` (kind=job) + ``_validate_tool_servers`` (kind=mcp):
      policy-gated — only when ``pol`` resolved."""
    out: list[Problem] = []
    for leaf_kind, leaf in (
        ("tool", lambda: _validate_sandbox_tools(name)),
        ("tool", lambda: _validate_declared_exec_guard(name)),
        ("job", lambda: _validate_jobs(name, pol) if pol is not None else []),
        ("mcp", lambda: _validate_tool_servers(name, pol) if pol is not None else []),
    ):
        for problem in leaf():
            out.append(replace(problem, kind=leaf_kind))
    return out


# --- the walk ----------------------------------------------------------------


def _audit_org(name: str) -> list[Problem]:
    """Validate one org's bundle — fully offline (no server, no tokens).

    Every per-org contract rule in one walk. Returns ALL problems (errors +
    warns); the caller decides strict vs audit (see ``load_org`` /
    ``audit_org``).
    """
    v: list[Problem] = []
    org_dir = _org_path(name)
    agents_md = org_dir / "AGENTS.md"

    if not agents_md.is_file():
        return [Problem("error", "org-agents-md", f"{name}: orgs/{name}/AGENTS.md missing")]

    fm, _ = _split_frontmatter(agents_md.read_text())
    if fm:
        v.append(
            Problem(
                "error",
                "no-legacy-org-roster",
                f"{name}: AGENTS.md carries YAML frontmatter — the roster must "
                f"live in orgs/{name}/org.yaml and AGENTS.md must be prose-only",
            )
        )

    v.extend(_org_extends_chain_problems(name))

    if org_extends(name) is not None and not (org_dir / "policy.yaml").is_file():
        v.append(
            Problem(
                "warn",
                "org-extends-policy",
                f"{name}: extends a parent but ships no own policy.yaml — policy "
                f"is NOT inherited (each org owns its egress). Add a policy.yaml, "
                f"or confirm this org should run with no egress ACL.",
            )
        )

    org_yaml = org_dir / "org.yaml"
    shape_ok = True
    slugs: list[str] = []
    roster_deny: list[str] = []
    if org_yaml.is_file():
        data = yaml.safe_load(org_yaml.read_text()) or {}
        if not isinstance(data, dict):
            v.append(
                Problem(
                    "error",
                    "org-yaml-shape",
                    f"{name}: org.yaml top-level must be a mapping, got {type(data).__name__}",
                )
            )
            shape_ok = False
        else:
            try:
                cfg = OrgConfig.model_validate(data)
            except Exception as exc:  # pydantic ValidationError
                v.append(
                    Problem("error", "org-yaml-shape", f"{name}: org.yaml schema error: {exc}")
                )
                shape_ok = False
            else:
                slugs = cfg.agents
                roster_deny = cfg.roster_deny
    elif not fm:
        slugs = []
    else:
        slugs = _parse_list(fm.get("agents", ""))

    roster: list[str] = slugs
    if shape_ok:
        try:
            roster = org_agent_slugs(name)
        except Exception:
            roster = slugs

    bad = sorted(set(roster) & set(roster_deny))
    if bad:
        v.append(
            Problem(
                "error",
                "roster-deny-enforced",
                f"{name}: roster must not include any slug from org.yaml "
                f"``roster_deny:`` ({bad}); denied={sorted(set(roster_deny))}. "
                f"This list is the org's own focus-CTO declaration — the CTO does "
                f"the thinking, delegates only to narrow specialists.",
            )
        )

    if any(s in {"general-purpose", "general"} for s in roster_deny):
        gp_cfg: Any = None
        gp_ok = True
        try:
            gp_cfg = profile_mod.load_profile(name)
        except (TypeError, ValueError, yaml.YAMLError):
            gp_ok = False
        if gp_ok:
            gp = gp_cfg.general_purpose_subagent if gp_cfg is not None else None
            if gp is None or gp.enabled is not False:
                v.append(
                    Problem(
                        "error",
                        "roster-deny-disables-general-purpose",
                        f"{name}: profile.yaml must declare "
                        "'general_purpose_subagent: {enabled: false}' — the roster "
                        "denies general-purpose but deepagents otherwise auto-adds "
                        "a heavy generic worker the roster-deny rule cannot see",
                    )
                )

    agent_subagents: dict[str, dict[str, Any]] = {}
    for slug in roster:
        extends_vs = _agent_extends_chain_problems(slug, name)
        if extends_vs:
            v.extend(extends_vs)
            continue
        sugar_vs = _capabilities_sugar_agent_problems(slug, name)
        if sugar_vs:
            v.extend(sugar_vs)
            continue
        try:
            sub = _load_agent_subagent(slug, name)
        except Exception as exc:
            v.append(
                Problem(
                    "error",
                    "agent-resolves",
                    f"{name}: agents: {slug!r} -> failed to read agent .md: {exc}",
                )
            )
            continue
        if sub is None:
            looked = ", ".join(str(d / f"{slug}.md") for d in _agent_search_dirs(name))
            v.append(
                Problem(
                    "error",
                    "agent-resolves",
                    f"{name}: agents: {slug!r} -> no agent .md found (searched: {looked})",
                )
            )
            continue
        agent_subagents[slug] = sub
        missing = sorted(_REQUIRED_AGENT_KEYS - sub.keys())
        if missing:
            v.append(
                Problem(
                    "error",
                    "agent-missing-keys",
                    f"{name}/{slug}: agent .md frontmatter missing required keys: {missing}",
                )
            )

    declared_names = declared_mod.declared_tool_names(org_dir / "sandbox")
    dyn_names = (
        dynamic_mod.DYNAMIC_TOOL_NAMES
        if profile_mod.load_dynamic_tools_enabled(name)
        else frozenset()
    )
    mw_by_name = {s.name: s for s in stack_mod.MIDDLEWARE_REGISTRY}
    for slug, sub in agent_subagents.items():
        for raw in _parse_list(sub.get("tools", [])):
            tool = raw.rsplit("/", 1)[-1]
            bare = (
                tool[len(dynamic_mod.PUX_DYN_PREFIX) :]
                if tool.startswith(dynamic_mod.PUX_DYN_PREFIX)
                else tool
            )
            if classify_slug(tool) is None and tool not in declared_names and bare not in dyn_names:
                v.append(
                    Problem(
                        "error",
                        "tool-resolves",
                        f"{name}/{slug}: tool {raw!r} -> "
                        f"{prefixed(tool, Category.SPECIALIST)!r} "
                        f"not a native fs tool, a "
                        f"pux_sandbox_* specialist, a declared "
                        f"sandbox tool, or a pux_dyn_* dynamic tool",
                    )
                )
        for raw in _parse_list(sub.get("middleware", [])):
            if raw not in mw_by_name:
                v.append(
                    Problem(
                        "error",
                        "agent-middleware-scope",
                        f"{name}/{slug}: per-agent middleware {raw!r} is not "
                        f"registered (known: {sorted(mw_by_name)})",
                    )
                )
            elif stack_mod.Scope.SUBAGENT not in mw_by_name[raw].scope:
                v.append(
                    Problem(
                        "error",
                        "agent-middleware-scope",
                        f"{name}/{slug}: per-agent middleware {raw!r} is not "
                        f"allowed in the subagent scope (allowed: "
                        f"{sorted(s.value for s in mw_by_name[raw].scope)})",
                    )
                )

    pol: policy_mod.Policy | None = None
    policy_path = org_dir / "policy.yaml"
    if policy_path.is_file():
        try:
            parsed = yaml.safe_load(policy_path.read_text())
        except yaml.YAMLError as e:
            v.append(
                Problem("error", "policy-parse", f"{name}: policy.yaml is not valid YAML: {e}")
            )
            parsed = None
        if isinstance(parsed, dict):
            bad_sections = sorted(k for k in parsed if k not in KNOWN_POLICY_SECTIONS)
            if bad_sections:
                v.append(
                    Problem(
                        "error",
                        "policy-sections",
                        f"{name}: policy.yaml unknown sections "
                        f"{bad_sections}; allowed: "
                        f"{sorted(KNOWN_POLICY_SECTIONS)}",
                    )
                )
            if "tool_servers" in parsed:
                v.append(
                    Problem(
                        "error",
                        "no-legacy-tool-servers",
                        f"{name}: policy.yaml: the `tool_servers:` block is the "
                        f"forbidden legacy MCP declaration — move its entries to "
                        f"org.yaml `capabilities:` (kind: mcp), the one canonical "
                        f"site (CU-4).",
                    )
                )
            try:
                pol = policy_mod.load(name, _orgs_dir().parent)
                policy_mod.resolve_mounts(pol)
            except policy_mod.PolicyError as e:
                v.append(
                    Problem("error", "policy-schema", f"{name}: policy.yaml schema error: {e}")
                )
            except policy_mod.NoPolicy:
                pass
            if pol is not None:
                v.extend(_validate_host_setup(name, pol))
                v.extend(_validate_build_spec(name, pol))
                v.extend(_validate_protocols(name, pol))
        elif parsed is not None:
            v.append(
                Problem(
                    "error",
                    "policy-shape",
                    f"{name}: policy.yaml top-level must be a mapping, got {type(parsed).__name__}",
                )
            )

    v.extend(_validate_capabilities(name, pol))

    if (org_dir / "profile.yaml").is_file():
        v.extend(audit_profile(name))

    try:
        from pux_harness.agent.prompt_show import (
            budget_for,
            stats_for_org,
            stats_for_agent,
        )

        project_root = _orgs_dir().parent

        budget = budget_for(name, project_root, scope="supervisor")
        if budget is not None:
            stats = stats_for_org(name, project_root)
            over_by = stats["total_tokens"] - budget
            if over_by > 0:
                v.append(
                    Problem(
                        "error",
                        "prompt-budget",
                        f"{name}: supervisor prompt {stats['total_tokens']:,} "
                        f"tokens > budget {budget:,} (over by {over_by:,}). "
                        f"Slim the prompt (AGENTS.md / skills escape hatch) or "
                        f"raise the budget in orgs/_shared/budgets.yaml with a "
                        f"waiver_reason.",
                    )
                )

        sub_budget = budget_for(name, project_root, scope="subagent")
        if sub_budget is not None:
            for slug in org_agent_slugs(name):
                agent_stats = stats_for_agent(name, slug, project_root)
                if agent_stats is None:
                    continue
                over_by = agent_stats["total_tokens"] - sub_budget
                if over_by > 0:
                    v.append(
                        Problem(
                            "error",
                            "prompt-budget-subagent",
                            f"{name}/{slug}: subagent prompt "
                            f"{agent_stats['total_tokens']:,} tokens > budget "
                            f"{sub_budget:,} (over by {over_by:,}). Slim the agent "
                            f"body (move how-to to a skill reference, keep only "
                            f"role + rubric) or raise subagent_default in "
                            f"orgs/_shared/budgets.yaml.",
                        )
                    )
    except (FileNotFoundError, ImportError, ValueError):
        pass

    return v


def audit_org(name: str) -> list[Problem]:
    """AUDIT mode — validate one org's bundle, return every problem (errors +
    warns) without raising. For diagnostic CLI reports and non-blocking checks."""
    try:
        return _audit_org(name)
    except FileNotFoundError:
        return [Problem("error", "org-agents-md", f"{name}: orgs/{name}/AGENTS.md missing")]


def load_org(name: str) -> list[Problem]:
    """STRICT mode — validate one org's bundle, raise ``OrgContractError`` on
    the first error (carrying ALL problems found), else return the warns.

    The enforcement boundary: a malformed org fails this call, not the first
    build. Warns never raise — they're reported, not blocking."""
    problems = _audit_org(name)
    errors = [p for p in problems if p.severity == "error"]
    if errors:
        raise OrgContractError(name, problems)
    return problems


def has_errors(problems: list[Problem]) -> bool:
    return any(p.severity == "error" for p in problems)


# --- the profile.yaml surface (same strict/audit contract as the org) ---------


def audit_profile(name: str) -> list[Problem]:
    """AUDIT mode for ``orgs/<name>/profile.yaml`` — every problem as a granular
    ``Problem`` (schema, the legacy ``subagents:`` block, the ``base_system_prompt``
    nuclear-replace, rubric / middleware / retry / ask_user block shapes, and the
    middleware name/scope overrides). Never raises; ``[]`` when the org ships no
    profile (absence is 'skipped', not a violation).

    Mirrors ``profile.validate_profile``'s exercise-every-loader intent, but
    COLLECTS instead of stopping at the first raise — a profile with a bad
    ``rubric:`` AND a bad ``middleware:`` block reports both."""

    v: list[Problem] = []
    org_dir = _org_path(name)
    if not (org_dir / "profile.yaml").is_file():
        return []
    try:
        raw = yaml.safe_load((org_dir / "profile.yaml").read_text())
    except yaml.YAMLError as exc:
        v.append(
            Problem(
                "error", "profile-schema", f"{name}: profile.yaml does not parse as YAML: {exc}"
            )
        )
        return v
    if isinstance(raw, dict) and "subagents" in raw:
        v.append(
            Problem(
                "error",
                "no-legacy-subagents-block",
                f"{name}: profile.yaml: the top-level `subagents:` block was "
                f"removed — folded into per-agent `extends:` + delta frontmatter "
                f"fields (tools_add / tools_remove / skills_add / "
                f"description_append / tool_description_overrides / "
                f"system_prompt_suffix / excluded_tools). "
                f"Move each subagent's override into its own "
                f"`orgs/{name}/agents/<slug>.md` (or a shared base + `extends:`).",
            )
        )
    if not isinstance(raw, dict):
        v.append(
            Problem(
                "error",
                "profile-schema",
                f"{name}: profile.yaml: top level must be a mapping, got {type(raw).__name__}",
            )
        )
        return v

    try:
        cfg = profile_mod.load_profile(name)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        v.append(Problem("error", "profile-schema", f"{name}: profile.yaml schema error: {exc}"))
        cfg = None
    if cfg is not None and cfg.base_system_prompt is not None:
        v.append(
            Problem(
                "error",
                "profile-base-system-prompt",
                f"{name}: profile.yaml: `base_system_prompt` is removed — it was a "
                f"global-REPLACE that wiped the assembled prompt. Use "
                f"`system_prompt_suffix` (append) instead.",
            )
        )

    _PROFILE_LOADERS: tuple[tuple[str, Any, tuple[type, ...]], ...] = (
        ("profile-rubric", profile_mod.load_rubric_gate, (TypeError, ValueError)),
        (
            "profile-middleware-shape",
            profile_mod.load_middleware_overrides,
            (TypeError, ValueError),
        ),
        ("profile-model-retry", profile_mod.load_model_retry, (TypeError, ValueError)),
        ("profile-tool-retry", profile_mod.load_tool_retry, (TypeError, ValueError)),
        ("profile-ask-user", profile_mod.load_ask_user_enabled, (TypeError, ValueError)),
    )
    for rule, loader, exc_types in _PROFILE_LOADERS:
        try:
            loader(name)
        except exc_types as exc:
            v.append(Problem("error", rule, f"{name}: profile.yaml: {exc}"))

    for err in stack_mod.validate_overrides(name):
        v.append(Problem("error", "middleware-overrides", err))
    return v


def load_profile(name: str) -> list[Problem]:
    """STRICT mode for ``orgs/<name>/profile.yaml`` — raise ``OrgContractError``
    on the first error (carrying ALL problems found), else return the warns.

    The enforcement boundary for the profile: a malformed profile fails this
    call, not the first ``build_graph``. Warns never raise."""
    problems = audit_profile(name)
    errors = [p for p in problems if p.severity == "error"]
    if errors:
        raise OrgContractError(name, problems)
    return problems
