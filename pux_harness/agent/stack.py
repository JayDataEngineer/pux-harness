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

The context layer (``ContextMiddleware`` + ``ctx_recall``/``ctx_search``) is the
one NON-toggleable base: it's a coupled (middleware, retrieval-tools) seam built
per-scope (one instance for the supervisor, one for the subagent tree, both
bound to the shared ``EventStore`` — byte-identical to the pre-factory build).
It's listed in this module's docstring for "one place" visibility but an org
can't drop it via overrides (turning off capture/offload is rarely wanted; that
knob can be added later if ever needed). Everything ELSE is toggleable.
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
    load_subagent_overrides,
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

    ``build`` receives a ``StackCtx`` and returns the instance, a list of
    instances, or ``None`` (``None`` => skip — used by ``rubric`` when no gate is
    armed, so the name can stay in the default list without forcing
    construction).
    """

    name: str
    scope: frozenset[Scope]
    build: Callable[["StackCtx"], AgentMiddleware | list[AgentMiddleware] | None]


@dataclass(frozen=True)
class StackCtx:
    """The bundle of facts each ``MiddlewareSpec.build`` resolves against."""

    org: str
    facts: RuntimeFacts
    rubric_gate: Any | None    # profile.RubricGate | None
    exec_client: Any           # DockerExecClient — for the grader's evidence tools


# --- THE registry ---------------------------------------------------------
# Order is preserved by ``_resolve_toggles`` → matches the historical mount
# order (routing, then session_guide; rubric is appended after the baseline).


def _build_routing(_ctx: StackCtx) -> AgentMiddleware:
    return RoutingMiddleware()


def _build_session_guide(_ctx: StackCtx) -> AgentMiddleware:
    return SessionGuideMiddleware()


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


def _build_rubric(ctx: StackCtx) -> AgentMiddleware | None:
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
    MiddlewareSpec("routing", frozenset({Scope.SUPERVISOR}), _build_routing),
    MiddlewareSpec("session_guide", frozenset({Scope.SUPERVISOR}), _build_session_guide),
    MiddlewareSpec("rubric", frozenset({Scope.SUPERVISOR}), _build_rubric),
]


# --- defaults (the "one place") -------------------------------------------

# The toggleable supervisor baseline. ``rubric`` is appended at resolve time
# ONLY when the org's gate is armed (a runtime fact, not an org override) —
# matches the pre-factory behavior exactly.
DEFAULT_SUPERVISOR: list[str] = ["routing", "session_guide"]

# Subagents get the always-on context layer only; no toggleable middleware by
# default (RubricMiddleware is a supervisor-level gate; routing/session are
# supervisor concerns). An org MAY add one via ``middleware.subagent.add``.
DEFAULT_SUBAGENT: list[str] = []


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
    """Resolve the toggleable middleware for one scope: defaults + gate-driven
    rubric − removes + adds, then build each in order. Validate every name +
    scope. Returns the built instances (skipping any ``build`` that returns
    ``None``, e.g. rubric with no gate)."""
    by_name = _specs_by_name()

    # Start from defaults; append gate-driven rubric (supervisor only).
    names: list[str] = list(default_names)
    if scope is Scope.SUPERVISOR and ctx.rubric_gate is not None and ctx.rubric_gate.enabled:
        if "rubric" not in names:
            names.append("rubric")

    # Apply the rules seam BEFORE org overrides (rules see the default set).
    names = _apply_rules(ctx.facts, scope, names)

    # Org overrides: remove first, then add (an add wins a same-named remove).
    names = [n for n in names if n not in removes]
    for n in adds:
        if n not in names:
            names.append(n)

    # Validate every name + scope before building — fail loud, no silent skip.
    unknown = [n for n in names if n not in by_name]
    if unknown:
        msg = (
            f"{ctx.org}: unknown middleware name(s) {sorted(set(unknown))}; "
            f"registered: {sorted(by_name)}"
        )
        raise ValueError(msg)
    for n in names:
        spec = by_name[n]
        if scope not in spec.scope:
            msg = (
                f"{ctx.org}: middleware {n!r} is not allowed in the {scope.value} "
                f"scope (allowed scopes: {sorted(s.value for s in spec.scope)})"
            )
            raise ValueError(msg)

    out: list[AgentMiddleware] = []
    for n in names:
        built = by_name[n].build(ctx)
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
    routing + session_guide, + rubric iff gate), same tools (specialists +
    retrieval), same prompt (root + org + addendum)."""
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

    # Context layer — built per scope (one instance for the supervisor, one for
    # the subagent tree), both bound to the shared EventStore. The supervisor's
    # call also yields the retrieval tools (ctx_recall/ctx_search), reused for
    # every subagent whitelist below — byte-identical to the pre-factory build.
    ctx_mw_sup, ctx_tools = build_context_layer()
    ctx_mw_sub, _ = build_context_layer()

    # Toggleable middleware resolved via the registry.
    sup_add, sup_remove = scoped[Scope.SUPERVISOR]
    sub_add, sub_remove = scoped[Scope.SUBAGENT]
    supervisor_toggles = _resolve_toggles(
        ctx, Scope.SUPERVISOR, DEFAULT_SUPERVISOR, sup_add, sup_remove,
    )
    subagent_toggles = _resolve_toggles(
        ctx, Scope.SUBAGENT, DEFAULT_SUBAGENT, sub_add, sub_remove,
    )

    # BrowserVision mounts INNERMOST (last): handler(request) then returns the
    # RAW tool string before ContextMiddleware offloads it, so screenshot_path
    # is still inline to find. Default ON (mimo-v2.5 driver is multimodal); a
    # cloner with a text-only driver sets PUX_BROWSER_VISION=0 and it is NOT
    # mounted at all (clean absent-from-list semantics, not mounted-but-off).
    # No-op for every non-pux_sandbox_browser_* tool, so it costs nothing on the
    # rest of the surface. One instance per scope mirrors ContextMiddleware.
    supervisor_middleware: list[AgentMiddleware] = [*ctx_mw_sup, *supervisor_toggles]
    subagent_middleware: list[AgentMiddleware] = [*ctx_mw_sub, *subagent_toggles]
    if browser_vision_enabled():
        supervisor_middleware.append(BrowserVisionMiddleware(exec_client))
        subagent_middleware.append(BrowserVisionMiddleware(exec_client))

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

    subagent_overrides = load_subagent_overrides(org)
    subagents = load_subagents(
        org, specialists,
        profile=profile,
        subagent_middleware=subagent_middleware,
        retrieval_tools=ctx_tools,
        subagent_overrides=subagent_overrides,
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
