"""Single owner of "apply an org ``HarnessProfileConfig`` to a resolved stack".

WHY THIS MODULE EXISTS (the upstream gap)
------------------------------------------
deepagents' ``HarnessProfile`` system is **global + model-keyed**. Registering a
profile for model X means EVERY agent built on model X gets that profile — two
orgs on one model merge-collide, and there is no ``unregister``. The pux harness
is **org-keyed** (one long-lived server builds many orgs per process), so it
deliberately does NOT use the ``_HARNESS_PROFILES`` registry — pinned by the
``no-harness-profile-registration`` contract rule.

The cost: three narrow applications that upstream's ``create_deep_agent`` does
INTERNALLY through a registered profile must be applied HERE, by pux, on the
resolved stack. They are:

1. ``excluded_tools`` + ``tool_description_overrides`` → ``apply_profile_to_tools``
2. ``excluded_middleware`` → folded into the harness' scoped remove-set by
   ``merge_profile_excluded_middleware`` (the resolver then drops those names
   pre-build; never constructing the instance).
3. ``system_prompt_suffix`` → ``apply_profile_to_prompt`` (the canonical
   append; the assembler's ``PromptCtx.system_prompt_suffix`` is the SAME
   application at the prompt-assembly layer — documented here as one site).

UPSTREAM PR (the cutover)
-------------------------
``create_deep_agent(..., harness_profile: HarnessProfile | None = None)`` — a
per-call kwarg that bypasses the global registry. When it lands (langchain-ai/
deepagents PR, link here when filed), all three applications collapse to a
one-line pass-through and this module is DELETED. Until then, this is the seam:
the single place to audit, the single place to delete.

Re-exported from ``profile.py`` (``apply_profile_to_tools``) for back-compat —
existing imports keep working; new code imports from here.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool

from pux_harness.agent.profile import HarnessProfileConfig, MiddlewareOverrides

try:  # upstream helper — reused, NOT re-implemented
    from deepagents._tools import _apply_tool_description_overrides
except ImportError:  # pragma: no cover — deepagents is a core dep
    _apply_tool_description_overrides = None  # type: ignore[assignment]


__all__ = [
    "apply_profile_to_tools",
    "merge_profile_excluded_middleware",
    "apply_profile_to_prompt",
]


def apply_profile_to_tools(
    tools: list[BaseTool], cfg: HarnessProfileConfig
) -> list[BaseTool]:
    """Apply ``tool_description_overrides`` + ``excluded_tools`` to a tool list.

    Used at both application sites — the MAIN agent stack (in ``build_stack``)
    and EACH subagent's resolved whitelist (in ``load_subagents``) — so an
    org-wide override reaches the browser subagent, not just the CTO.
    ``_apply_tool_description_overrides`` copies + rewrites (it never mutates
    caller-owned tools), so this is safe to call per-subagent. Filtering by
    ``tool.name`` (the prefixed ``pux_sandbox_*`` identifier the profile keys
    on)."""
    out: list[BaseTool] = tools
    if cfg.tool_description_overrides and _apply_tool_description_overrides is not None:
        out = _apply_tool_description_overrides(out, cfg.tool_description_overrides)
    if cfg.excluded_tools:
        out = [t for t in out if t.name not in cfg.excluded_tools]
    return out


def merge_profile_excluded_middleware(
    overrides: MiddlewareOverrides,
    profile: HarnessProfileConfig | None,
) -> MiddlewareOverrides:
    """Fold the deepagents ``excluded_middleware`` field into the harness'
    scoped remove-set.

    The deepagents ``excluded_middleware`` field (a ``frozenset[str]`` on
    ``HarnessProfileConfig``) was a DEAD path before this seam —
    ``create_deep_agent`` only honors it through a *registered* profile, which
    the harness deliberately doesn't use (org-keyed, see module doc). We honor
    it OURSELVES here, treating it as an UNSCOPED supervisor remove (the
    supervisor is the only scope those names could ever mount on today). Both
    forms (``excluded_middleware`` and the scoped ``middleware.supervisor.
    remove``) route through the same remove-set, so an org can use whichever
    reads better; the scoped block is primary.

    Returns a NEW frozen ``MiddlewareOverrides`` — the resolver consumes it
    pre-build, so removed names never construct.
    """
    sup_remove: frozenset[str] = overrides.supervisor_remove
    if profile is not None and profile.excluded_middleware:
        sup_remove = sup_remove | frozenset(profile.excluded_middleware)
    return MiddlewareOverrides(
        supervisor_add=overrides.supervisor_add,
        supervisor_remove=sup_remove,
        subagent_add=overrides.subagent_add,
        subagent_remove=overrides.subagent_remove,
    )


def apply_profile_to_prompt(
    prompt: str, cfg: HarnessProfileConfig | None
) -> str:
    """Append the profile's ``system_prompt_suffix`` to an assembled prompt.

    The canonical application site for the org-wide suffix. The supervisor +
    subagent assemblers thread this SAME field through ``PromptCtx`` (so the
    suffix lands at its documented position in the part order); this function
    is the CANONICAL signature for callers that build prompts OUTSIDE the
    assembler (e.g. the general-purpose subagent's prompt in ``orgs.py``) and
    the named seam for the upstream-gap cutover. No-op when ``cfg`` is None or
    the suffix is empty.
    """
    if cfg is None or not cfg.system_prompt_suffix:
        return prompt
    return f"{prompt}\n\n{cfg.system_prompt_suffix}"
