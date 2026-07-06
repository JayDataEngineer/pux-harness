"""The single place the per-org agent stack is resolved.

``build_stack(org, ...)`` is the factory: given an org's deps (the specialist
tools, the loaded profile, the rubric gate, the docker-exec client) it emits a
fully-resolved ``StackPlan`` — supervisor tools, supervisor middleware (in
order), supervisor prompt, and the compiled subagents. ``graph.build_graph``
calls it and hands the plan to ``create_deep_agent``; it does no stack assembly
of its own.

Why a factory (the user's stated goal): "one place to adjust defaults that
cascade across tools, middleware, agents/subagents, prompts — the org system is
an override system." Everything that varies per-org flows through HERE:

* **Defaults** live in this module as plain lists (``DEFAULT_SUPERVISOR``,
  ``DEFAULT_SUBAGENT`` + the tool/prompt defaults). Change a default here, every
  org changes.
* **Org overrides** live in ``orgs/<org>/profile.yaml`` and are resolved here:
  the deepagents fields (``system_prompt_suffix`` / ``base_system_prompt`` /
  ``tool_description_overrides`` / ``excluded_tools`` / ``excluded_middleware``)
  PLUS a harness-local ``middleware:`` block (``supervisor``/``subagent`` ×
  ``add``/``remove``) that finally makes the middleware stack data-driven.

The selectable middleware is a ``MIDDLEWARE_REGISTRY`` of ``MiddlewareSpec`` —
the sibling of the tool ``REGISTRY`` (``sandbox/tools/registry.py``). Adding a
middleware means one ``MiddlewareSpec`` line + its ``build``; every default +
override path picks it up. There is no second hand-maintained middleware list
anywhere — ``graph.py`` imports none of these classes directly (the
``no-legacy-middleware-in-graph`` contract tripwire enforces that).

Org-keyed, not model-keyed: like ``profile.py`` we deliberately do NOT use
deepagents' ``_HARNESS_PROFILES`` registry (two orgs on one model would
merge-collide; the long-lived server builds many orgs per process). The factory
hands ``create_deep_agent`` a RESOLVED stack; deepagents' own
``excluded_middleware`` filtering only fires through a registered profile, so we
honor that field OURSELVES here (it was a dead path before — see
``_resolve_toggles``).

The context layer (``ContextMiddleware`` + ``ctx_recall``/``ctx_search``) and
``BrowserVisionMiddleware`` are FIRST-CLASS registry specs (``context`` /
``browser_vision``), default-on and removable like every other middleware —
the user's "selectively remove/add middleware" request, applied uniformly. The
context spec's coupled (middleware, retrieval-tools) pair escapes via a mutable
``StackCtx.emitted_tools_supervisor`` side channel (one shared ``EventStore``,
two scope-local instances, byte-identical to the pre-Phase-3 build that called
``build_context_layer`` once per scope). ``browser_vision`` is env-gated (on for
the multimodal mimo-v2.5 driver; off → ``None`` → clean absent-from-list) and
listed LAST in the registry so it mounts innermost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from deepagents import RubricMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool

from pux_harness.agent.model import get_model
from pux_harness.agent.orgs import (
    _GENERAL_PURPOSE_NAME,
    _build_general_purpose_sub,
    build_system_prompt,
    load_subagents,
)
from pux_harness.agent.profile import (
    HarnessProfileConfig,
    MiddlewareOverrides,
    apply_profile_to_tools,
    load_middleware_overrides,
)
from pux_harness.context.browser_vision import (
    BrowserVisionMiddleware,
    browser_vision_enabled,
)
from pux_harness.context.layer import build_context_layer
from pux_harness.context.sandbox_routing import RoutingMiddleware
from pux_harness.context.session_guide import SessionGuideMiddleware
from pux_harness.sandbox.tools import build_grader_tools

__all__ = [
    "Scope",
    "RuntimeFacts",
    "MiddlewareSpec",
    "StackCtx",
    "StackPlan",
    "MIDDLEWARE_REGISTRY",
    "DEFAULT_SUPERVISOR",
    "DEFAULT_SUBAGENT",
    "middleware_names",
    "build_stack",
    "validate_overrides",
]


# --- the vocabulary -------------------------------------------------------


class Scope(Enum):
    """Which agent tier a middleware may mount on.

    Plain ``Enum`` (not ``StrEnum``) so a scope never compares equal to a bare
    string."""

    SUPERVISOR = "supervisor"   # the CTO / main agent
    SUBAGENT = "subagent"       # every specialist under ``task``


@dataclass(frozen=True)
class RuntimeFacts:
    """Transport-/environment-level facts the rules layer predicates on.

    Most stack decisions are ORG-level (profile.yaml). A few are RUNTIME-level —
    they depend on how the graph is being driven, not which org. The motivating
    example (the user's): "if we call over MCP, remove the ``ask_user`` tool from
    supervisor agents." MCP + ``ask_user`` aren't wired yet, but the seam is here
    so that rule lands cleanly when they are — see ``_apply_rules``.

    ``build_graph`` doesn't currently thread facts (no caller has a rule to
    assert yet), so the default ``RuntimeFacts()`` is used. That's intentional:
    the seam exists, identity-by-default, ready to grow.
    """

    transport: str = "serve"        # serve | direct | acp | tui
    mcp_active: bool = False
    tool_servers_active: bool = False
    provider: str | None = None
    tier: str | None = None


@dataclass(frozen=True)
class MiddlewareSpec:
    """One selectable middleware, declared once. The unit of the middleware
    surface (sibling of the tool ``ToolSpec``).

    ``build`` receives the ``StackCtx`` AND the ``Scope`` it's being resolved
    for, and returns the instance, a list of instances, or ``None`` (``None``
    => skip — used by ``rubric`` when no gate is armed and ``browser_vision``
    when the driver is text-only, so the name can stay in the default list
    without forcing construction). The scope arg lets one spec build differently
    per tier (``context`` emits retrieval tools on the supervisor, not the
    subagent).
    """

    name: str
    scope: frozenset[Scope]
    build: Callable[["StackCtx", Scope], AgentMiddleware | list[AgentMiddleware] | None]


@dataclass(frozen=True)
class StackCtx:
    """The bundle of facts each ``MiddlewareSpec.build`` resolves against."""

    org: str
    facts: RuntimeFacts
    rubric_gate: Any | None    # profile.RubricGate | None
    exec_client: Any           # DockerExecClient — for the grader's evidence tools
    # Side-effect channel for the ``context`` spec: ``build_context_layer``
    # returns the coupled (middleware, retrieval-tools) pair, but a spec's
    # ``build`` returns only middleware. The supervisor context build deposits
    # its retrieval tools (ctx_recall/ctx_search) here, and ``build_stack`` reads
    # them AFTER the resolver runs, threading them into supervisor_tools + every
    # subagent whitelist (one shared EventStore). The field REFERENCE is frozen;
    # the list itself is mutable — specs extend it. Fresh per ``build_stack``
    # call (a new ``StackCtx`` each time), so no cross-org leakage.
    emitted_tools_supervisor: list = field(default_factory=list)


# --- THE registry ---------------------------------------------------------
# Order is preserved by ``_resolve_toggles`` → matches the historical mount
# order (routing, then session_guide; rubric is appended after the baseline).


def _build_routing(_ctx: StackCtx, _scope: Scope) -> AgentMiddleware:
    return RoutingMiddleware()


def _build_session_guide(_ctx: StackCtx, _scope: Scope) -> AgentMiddleware:
    return SessionGuideMiddleware()


def _build_context(ctx: StackCtx, scope: Scope) -> AgentMiddleware:
    """The ContextMiddleware (capture + offload into the shared ``EventStore``)
    for ONE tier. ``build_context_layer`` returns the coupled (middleware,
    retrieval-tools) pair; the tools escape via ``ctx.emitted_tools_supervisor``
    on the SUPERVISOR tier only — the subagent tree reuses the supervisor's
    ctx_tools (threaded through ``load_subagents(retrieval_tools=)``), one
    shared store, two scope-local instances. Byte-identical to the pre-Phase-3
    build (which called ``build_context_layer`` once per scope)."""
    mw, tools = build_context_layer()
    if scope is Scope.SUPERVISOR:
        ctx.emitted_tools_supervisor.extend(tools)
    return mw


def _build_browser_vision(ctx: StackCtx, _scope: Scope) -> AgentMiddleware | None:
    """``BrowserVisionMiddleware`` when the multimodal driver is on (default for
    mimo-v2.5); ``None`` (skip) when ``PUX_BROWSER_VISION=0`` — clean
    absent-from-list, not mounted-but-off. Listed LAST in the registry so it
    mounts INNERMOST: the raw tool string is still inline before
    ContextMiddleware offloads it (so ``screenshot_path`` stays findable).
    No-op for every non-``pux_sandbox_browser_*`` tool, so it costs nothing on
    the rest of the surface. One instance per scope mirrors the context spec."""
    if not browser_vision_enabled():
        return None
    return BrowserVisionMiddleware(ctx.exec_client)


def _log_rubric_evaluation(ev: dict) -> None:
    """``on_evaluation`` hook for ``RubricMiddleware`` — print each grader
    verdict so the gate is OBSERVABLE in the run trace.

    The grader runs through ``RubricMiddleware`` calling the ``pux_grader_*``
    tools, which exercise ``exec_client`` directly and so bypass
    ``backend.execute_log``. Without this hook the gate firing (verdict +
    explanation + per-criterion) is invisible to the operator — and an invisible
    gate can't be told from a decorative one. Exceptions are suppressed upstream
    (it logs + swallows), so a print is safe here."""
    result = ev.get("result")
    explanation = str(ev.get("explanation", "") or "").replace("\n", " ")[:240]
    print(f"[grader] iter={ev.get('iteration')} result={result} :: {explanation}")


def _build_rubric(ctx: StackCtx, _scope: Scope) -> AgentMiddleware | None:
    """``RubricMiddleware`` ONLY when the org's rubric gate is present + enabled;
    ``None`` otherwise (so a no-gate org is byte-identical to today — no
    construction, no-op). The grader model resolves through the ``grader`` role
    and grades from REAL evidence via the 3 ``pux_grader_*`` tools (run tests /
    read the diff / grep), never from the agent's summary."""
    gate = ctx.rubric_gate
    if gate is None or not gate.enabled:
        return None
    return RubricMiddleware(
        model=get_model(role="grader", org=ctx.org),
        tools=build_grader_tools(ctx.exec_client),
        max_iterations=gate.max_iterations,
        on_evaluation=_log_rubric_evaluation,
    )


MIDDLEWARE_REGISTRY: list[MiddlewareSpec] = [
    # Canonical mount ORDER — ``_resolve_toggles`` emits in this order, so a
    # spec's registry position IS its mount position (context outermost,
    # browser_vision innermost past rubric). This is what makes the stack
    # byte-identical to the pre-Phase-3 build: [context, routing, session_guide,
    # rubric?, browser_vision?] on the supervisor.
    MiddlewareSpec("context", frozenset({Scope.SUPERVISOR, Scope.SUBAGENT}), _build_context),
    MiddlewareSpec("routing", frozenset({Scope.SUPERVISOR}), _build_routing),
    MiddlewareSpec("session_guide", frozenset({Scope.SUPERVISOR}), _build_session_guide),
    MiddlewareSpec("rubric", frozenset({Scope.SUPERVISOR}), _build_rubric),
    MiddlewareSpec("browser_vision", frozenset({Scope.SUPERVISOR, Scope.SUBAGENT}), _build_browser_vision),
]


# --- defaults (the "one place") -------------------------------------------

# The default-on supervisor baseline. ``rubric`` is gate-driven (added to the
# on-set iff the org's gate is armed) so it's NOT in the default list but still
# lands at its registry position when armed. ``context`` + ``browser_vision``
# are now selectable specs (default-on, removable via ``middleware.supervisor.
# remove``) — the user's "selectively remove/add middleware" request, applied
# uniformly to the formerly-non-toggleable layers too.
DEFAULT_SUPERVISOR: list[str] = ["context", "routing", "session_guide", "browser_vision"]

# Subagents get the context layer + browser_vision by default; routing /
# session_guide / rubric are supervisor concerns. An org MAY add a subagent
# middleware via ``middleware.subagent.add``.
DEFAULT_SUBAGENT: list[str] = ["context", "browser_vision"]


def middleware_names() -> list[str]:
    """Every registered middleware name. Single source for the contract checker
    + tests — validates that an org's override names are all real."""
    return [s.name for s in MIDDLEWARE_REGISTRY]


def _specs_by_name() -> dict[str, MiddlewareSpec]:
    return {s.name: s for s in MIDDLEWARE_REGISTRY}


# --- the resolver ---------------------------------------------------------


def _normalize_overrides(
    profile: HarnessProfileConfig | None,
    overrides: MiddlewareOverrides,
) -> dict[Scope, tuple[list[str], set[str]]]:
    """Merge the harness ``middleware:`` block with the deepagents
    ``excluded_middleware`` field into one ``{scope: (add, remove)}`` map.

    The deepagents ``excluded_middleware`` field (a ``frozenset[str]`` on
    ``HarnessProfileConfig``) was a DEAD path before the factory —
    ``create_deep_agent`` only honors it through a *registered* profile, which
    the harness deliberately doesn't use (org-keyed, see module doc). We honor
    it ourselves here, treating it as an UNSCOPED supervisor remove (the
    supervisor is the only scope those names could ever mount on today). Both
    forms route through the same remove-set, so an org can use whichever reads
    better; the scoped ``middleware.supervisor.remove`` is primary."""
    sup_remove: set[str] = set(overrides.supervisor_remove)
    if profile is not None and profile.excluded_middleware:
        sup_remove |= set(profile.excluded_middleware)
    return {
        Scope.SUPERVISOR: (list(overrides.supervisor_add), sup_remove),
        Scope.SUBAGENT: (list(overrides.subagent_add), set(overrides.subagent_remove)),
    }


def _resolve_toggles(
    ctx: StackCtx,
    scope: Scope,
    default_names: list[str],
    adds: list[str],
    removes: set[str],
) -> list[AgentMiddleware]:
    """Resolve the toggleable middleware for one scope.

    The on-set is composed in the order that makes org overrides win: defaults +
    gate-driven ``rubric`` (supervisor only), then the runtime-facts rules seam,
    then removes, then adds — so an add wins a same-named remove. Every on-set
    name is validated (registered + in-scope) BEFORE building. The built list is
    emitted in REGISTRY order — a spec's mount position is its registry position,
    not insertion order — so the stack is byte-identical regardless of which
    names are defaults vs adds (e.g. ``browser_vision`` always lands innermost
    past ``rubric``). A ``build`` returning ``None`` is skipped (``rubric`` with
    no gate, ``browser_vision`` with a text-only driver)."""
    by_name = _specs_by_name()

    # defaults + gate-driven rubric → rules → removes → adds (add wins over remove).
    on: set[str] = set(default_names)
    if scope is Scope.SUPERVISOR and ctx.rubric_gate is not None and ctx.rubric_gate.enabled:
        on.add("rubric")
    on = set(_apply_rules(ctx.facts, scope, list(on)))
    on -= set(removes)
    on |= set(adds)

    # Validate the FULL on-set — fail loud on an unknown add, a rule typo, or a
    # scope mismatch. (Remove-name typos are caught offline by validate_overrides.)
    unknown = sorted(n for n in on if n not in by_name)
    if unknown:
        raise ValueError(
            f"{ctx.org}: unknown middleware name(s) {unknown}; "
            f"registered: {sorted(by_name)}")
    for n in on:
        spec = by_name[n]
        if scope not in spec.scope:
            raise ValueError(
                f"{ctx.org}: middleware {n!r} is not allowed in the {scope.value} "
                f"scope (allowed scopes: {sorted(s.value for s in spec.scope)})")

    # Emit in REGISTRY order (canonical mount position) + build each.
    names = [s.name for s in MIDDLEWARE_REGISTRY if s.name in on]
    out: list[AgentMiddleware] = []
    for n in names:
        built = by_name[n].build(ctx, scope)
        if built is None:
            continue
        if isinstance(built, list):
            out.extend(built)
        else:
            out.append(built)
    return out


def _apply_rules(facts: RuntimeFacts, scope: Scope, names: list[str]) -> list[str]:
    """Runtime-facts rules seam. Today identity (no rule is wired), kept as an
    explicit, tested function so the policy layer is legible + extensible rather
    than a hidden branch.

    The motivating rule (lands when ``ask_user`` + MCP ship): an org exposing
    tools over MCP should NOT also offer a supervisor ``ask_user`` (the MCP
    caller can't answer it) — so::

        if facts.mcp_active and scope is Scope.SUPERVISOR:
            names = [n for n in names if n != "ask_user"]

    Tool-level rules (drop ``ask_user`` from the tool list, not the middleware
    list) live alongside this in ``build_stack`` when the tool exists.
    """
    _ = facts  # no rule wired yet; param kept for the seam + signature stability
    return list(names)


# --- the plan + the factory ----------------------------------------------


@dataclass(frozen=True)
class StackPlan:
    """The fully-resolved per-org stack. ``graph.build_graph`` threads this
    straight into ``create_deep_agent``."""

    supervisor_tools: list[BaseTool]
    supervisor_middleware: list[AgentMiddleware]
    supervisor_prompt: str
    subagents: list[dict[str, Any]]


def build_stack(
    org: str,
    *,
    specialists: list[BaseTool],
    profile: HarnessProfileConfig | None,
    rubric_gate: Any | None,
    exec_client: Any,
    facts: RuntimeFacts | None = None,
    mcp_tools: Sequence[BaseTool] = (),
) -> StackPlan:
    """Resolve the full stack for ``org`` — the single entry point.

    Pure w.r.t. the agent surface: no ``create_deep_agent``, no checkpointer,
    no model-init for the supervisor driver (that's ``graph.py``'s job). It DOES
    read ``profile.yaml`` (for the ``middleware:`` override block) and build the
    context layer + subagents, because those ARE the stack.

    Byte-identical to the pre-factory build when the org ships no profile AND no
    ``middleware:`` block AND no ``mcp_tools``: same middleware order (context +
    routing + session_guide, + rubric iff gate, + browser_vision iff the driver
    is multimodal), same tools (specialists + retrieval), same prompt (root +
    org + addendum)."""
    if facts is None:
        facts = RuntimeFacts(tool_servers_active=bool(mcp_tools))
    ctx = StackCtx(
        org=org,
        facts=facts,
        rubric_gate=rubric_gate,
        exec_client=exec_client,
    )
    overrides = load_middleware_overrides(org)
    scoped = _normalize_overrides(profile, overrides)

    # The WHOLE middleware stack — context + routing + session_guide + rubric +
    # browser_vision — flows through the registry now. ``context`` is a spec,
    # so it (and browser_vision) is selectable like every other middleware; its
    # retrieval tools escape via ctx.emitted_tools_supervisor as a build-time
    # side effect (one shared EventStore, two scope-local instances).
    sup_add, sup_remove = scoped[Scope.SUPERVISOR]
    sub_add, sub_remove = scoped[Scope.SUBAGENT]
    supervisor_middleware: list[AgentMiddleware] = _resolve_toggles(
        ctx, Scope.SUPERVISOR, DEFAULT_SUPERVISOR, sup_add, sup_remove,
    )
    subagent_middleware: list[AgentMiddleware] = _resolve_toggles(
        ctx, Scope.SUBAGENT, DEFAULT_SUBAGENT, sub_add, sub_remove,
    )
    ctx_tools = list(ctx.emitted_tools_supervisor)

    # Tools: MCP tools first (so profile overrides can shape them), then every
    # specialist + the retrieval surface.
    supervisor_tools: list[BaseTool] = [*mcp_tools, *specialists, *ctx_tools]
    if profile is not None:
        supervisor_tools = apply_profile_to_tools(supervisor_tools, profile)

    # Prompt: root + org + addendum, then profile overrides.
    prompt = build_system_prompt(org)
    if profile is not None:
        if profile.base_system_prompt:
            prompt = profile.base_system_prompt
        if profile.system_prompt_suffix:
            prompt = f"{prompt}\n\n{profile.system_prompt_suffix}"

    subagents = load_subagents(
        org, specialists,
        profile=profile,
        subagent_middleware=subagent_middleware,
        retrieval_tools=ctx_tools,
    )

    # Phase 1 — own the general-purpose subagent. deepagents auto-adds a HEAVY
    # ``general-purpose`` subagent to EVERY graph (deepagents/graph.py:716-717)
    # unless ``gp_profile.enabled is False`` OR a spec named ``general-purpose``
    # is already in the inline subagents. pux passes no GP kwarg AND stays off
    # the model-keyed ``_HARNESS_PROFILES`` registry (two orgs on one model
    # would merge-collide; the long-lived server builds many orgs per process;
    # there is no ``unregister``), so without an explicit spec the auto-add
    # fires for EVERY org — even dev-bot, whose roster rule
    # (``dev-bot-no-general-subagent``) reads ``org.yaml`` and so NEVER sees the
    # auto-added slot.
    #
    # Honor the NATIVE field (no parallel grammar): when the org's profile
    # carries a ``general_purpose_subagent:`` block (flowed straight through
    # ``HarnessProfileConfig.from_dict``), pux emits its OWN ``name=
    # "general-purpose"`` spec → the name now already exists inline → deepagents
    # skips the auto-add. The no-block case emits NOTHING, so deepagents'
    # auto-add still fires (byte-identical to today — the parity path). Three
    # cases live in ``orgs._build_general_purpose_sub``: neutered (enabled:
    # false) vs customized vs default.
    if profile is not None and profile.general_purpose_subagent is not None:
        # Defense against an org that ALSO rostered a literal "general-purpose"
        # specialist in org.yaml — the roster entry wins (don't double-emit;
        # ``not any(...)`` mirrors deepagents' own guard so there's one slot).
        if not any(s["name"] == _GENERAL_PURPOSE_NAME for s in subagents):
            subagents.append(_build_general_purpose_sub(
                profile.general_purpose_subagent,
                org,
                tool_surface=[*specialists, *ctx_tools],
                middleware=subagent_middleware,
                profile=profile,
            ))

    return StackPlan(
        supervisor_tools=supervisor_tools,
        supervisor_middleware=supervisor_middleware,
        supervisor_prompt=prompt,
        subagents=subagents,
    )


def validate_overrides(org: str) -> list[str]:
    """Offline validation of an org's ``middleware:`` override block — every
    add/remove name must be a registered middleware, scoped correctly. Returns
    the list of error strings (empty = valid). Called from the contract checker
    so a typo'd override fails ``--check-contract``, not the first build."""
    errors: list[str] = []
    try:
        overrides = load_middleware_overrides(org)
    except (TypeError, ValueError) as exc:
        return [f"{org}/profile.yaml: malformed middleware: block — {exc}"]
    by_name = _specs_by_name()
    for scope, names in (
        (Scope.SUPERVISOR, overrides.supervisor_add),
        (Scope.SUPERVISOR, overrides.supervisor_remove),
        (Scope.SUBAGENT, overrides.subagent_add),
        (Scope.SUBAGENT, overrides.subagent_remove),
    ):
        for n in names:
            if n not in by_name:
                errors.append(
                    f"{org}/profile.yaml: middleware name {n!r} is not "
                    f"registered (known: {sorted(by_name)})"
                )
                continue
            if scope not in by_name[n].scope:
                errors.append(
                    f"{org}/profile.yaml: middleware {n!r} is not allowed in "
                    f"the {scope.value} scope"
                )
    return errors
