"""Static prompt introspection — the renderer behind ``pux prompt show``.

Walks the SAME part registries (``SUPERVISOR_PROMPT_PARTS`` /
``SUBAGENT_PROMPT_PARTS``) that ``assemble_prompt`` uses at runtime, but labels
each part with its source, condition, and active/inactive state. This gives
experimenters a provenance-tagged view of the EXACT assembly without needing
Docker / exec_client / resolved specialists.

The conditional flags (``ask_user_active``, ``interpreter_mounted``) depend on
runtime state (transport, resolved middleware) that a static view can't fully
determine. Those parts are marked CONDITIONAL with their trigger explained, so
the experimenter knows what WOULD emit them at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pux_harness.agent.prompt_parts import (
    PromptCtx,
    PromptScope,
    SUBAGENT_PROMPT_PARTS,
    SUPERVISOR_PROMPT_PARTS,
    _ADDENDUM,
)
from pux_harness.kit import loaders as _loaders


# ── provenance metadata for each named part ──────────────────────────────
# Maps part name → (human-readable source, condition description).
# This is the ONLY place the "where did this come from?" labels live; the
# actual content comes from the part's build() over a real PromptCtx.

_PART_PROVENANCE: dict[str, tuple[str, str]] = {
    "agents_md_core": (
        "orgs/general/AGENTS.md + orgs/specialists/<org>/AGENTS.md (extends-chain overlay, "
        "concatenated root→child) + orgs/_shared/harness_addendum.md (the deepagents "
        "delegation/tool-surface prose — falls back to embedded _ADDENDUM when absent)",
        "always-on",
    ),
    "org_system_prompt_suffix": (
        "profile.yaml → system_prompt_suffix",
        "present (non-empty suffix configured)",
    ),
    "ask_user_suffix": (
        "orgs/_shared/ask_user_suffix.md (the turn-ending HITL instruction — "
        "falls back to embedded hitl.ASK_USER_PROMPT_SUFFIX when absent)",
        "ask_user active AND turn-based transport (direct/tui/acp/agui)",
    ),
    "dynamic_dispatch_suffix": (
        "orgs/_shared/dynamic_dispatch_suffix.md (the eval-tool dispatch strategy — "
        "falls back to embedded prompt_parts._DYNAMIC_DISPATCH_SUFFIX when absent)",
        "CodeInterpreterMiddleware mounted (strength-pro base, or middleware.supervisor.add: [interpreter])",
    ),
    "agent_body": (
        "orgs/<org>/agents/<slug>.md body (extends-merged). "
        "NOTE: deepagents prepends its own short DEFAULT_SUBAGENT_PROMPT at bind "
        "time — that prefix is NOT shown here but the model does see it.",
        "always-on",
    ),
    "agent_system_prompt_suffix": (
        "agent frontmatter → system_prompt_suffix",
        "present in frontmatter",
    ),
    # Default provenance for org-defined extra parts (name comes from the entry).
    # Any part name NOT in the dict above falls back to this label.
}


@dataclass(frozen=True)
class RenderedPart:
    """One part's rendering: name, source, condition, content (or None if inactive)."""

    name: str
    source: str
    condition: str
    content: str | None  # None = the part's build() returned None (conditional off)


def _resolve_org_suffix(org: str, project_root: Path) -> str | None:
    """The org-wide ``system_prompt_suffix`` from profile.yaml, resolved through
    the extends-chain (root→child, delta-wins on scalars so the child's suffix
    replaces the parent's). Uses the kit's ``project_root``-scoped resolvers so
    it works from any CWD (unlike the harness shim which is module-relative)."""
    import yaml

    from pux_harness.kit._paths import search_org_dir
    from pux_harness.kit.loaders import _resolved_org_chain

    suffix: str | None = None
    try:
        chain = _resolved_org_chain(org, project_root)  # root→child
    except (ValueError, FileNotFoundError):
        chain = [org]
    for ancestor in chain:
        try:
            org_dir = search_org_dir(ancestor, project_root)
        except FileNotFoundError:
            continue
        profile_path = org_dir / "profile.yaml"
        if not profile_path.is_file():
            continue
        data = yaml.safe_load(profile_path.read_text())
        if data and data.get("system_prompt_suffix"):
            suffix = data["system_prompt_suffix"]  # delta-wins: child replaces
    return suffix


def _build_supervisor_ctx(
    org: str,
    project_root: Path,
    *,
    ask_user: bool = False,
    interpreter: bool = False,
) -> PromptCtx:
    """Build a PromptCtx for the supervisor with as much static data as resolvable.

    ``ask_user`` and ``interpreter`` default False (their real values depend on
    runtime transport + middleware resolution); the renderer marks those parts
    CONDITIONAL so the experimenter understands why they're off. Pass
    ``ask_user=True`` / ``interpreter=True`` to SIMULATE the runtime-on state
    (the parts render ACTIVE with their real content) — the ``--with-ask-user``
    / ``--with-interpreter`` CLI flags feed into these kwargs.
    """
    from pux_harness.agent.prompt_parts import (
        load_ask_user_suffix,
        load_dynamic_dispatch_suffix,
        load_harness_addendum,
    )

    agents_md_base = _loaders.build_system_prompt(org, project_root=project_root)
    return PromptCtx(
        agents_md_base=agents_md_base,
        harness_addendum=load_harness_addendum(project_root),
        system_prompt_suffix=_resolve_org_suffix(org, project_root),
        ask_user_active=ask_user,
        ask_user_text=load_ask_user_suffix(project_root),
        interpreter_mounted=interpreter,
        dynamic_dispatch_text=load_dynamic_dispatch_suffix(project_root),
    )


def _build_subagent_ctx(
    org: str, slug: str, project_root: Path
) -> tuple[PromptCtx, str | None]:
    """Build a PromptCtx for one subagent. Returns (ctx, error) where error is a
    human-readable message when the slug can't be resolved (None on success)."""
    spec = _loaders._load_agent_spec(slug, org, project_root)
    if spec is None:
        msg = (
            f"agent {slug!r} not found under orgs/specialists/{org}/agents/, "
            f"orgs/{org}/agents/, or orgs/_shared/agents/"
        )
        return PromptCtx(), msg
    body = spec.get("system_prompt", "")
    agent_suffix = spec.get("system_prompt_suffix") or None
    return (
        PromptCtx(
            agent_body=body,
            system_prompt_suffix=_resolve_org_suffix(org, project_root),
            agent_system_prompt_suffix=agent_suffix,
        ),
        None,
    )


def render_parts(
    scope: PromptScope, ctx: PromptCtx, extra: tuple = ()
) -> list[RenderedPart]:
    """Walk the registry for ``scope`` (+ any org-defined ``extra`` parts) and
    render each with provenance. Extra parts (from ``profile.yaml →
    extra_prompt_parts``) are always-on file-injected sections appended AFTER
    the built-in parts."""
    registry = (
        SUPERVISOR_PROMPT_PARTS
        if scope is PromptScope.SUPERVISOR
        else SUBAGENT_PROMPT_PARTS
    )
    full_registry = (*registry, *extra)
    out: list[RenderedPart] = []
    for spec in full_registry:
        if scope not in spec.scope:
            continue
        chunk = spec.build(ctx)
        source, condition = _PART_PROVENANCE.get(
            spec.name,
            (
                f"profile.yaml → extra_prompt_parts → file (name={spec.name!r})",
                "always-on (org-defined extra part)",
            ),
        )
        out.append(
            RenderedPart(
                name=spec.name, source=source, condition=condition, content=chunk
            )
        )
    return out


def _resolve_extra_parts(
    org: str, project_root: Path, scope: PromptScope
) -> tuple:
    """CWD-independent extra-parts resolver for the introspection view. Walks
    the extends-chain root→child (child-wins on lists); file paths resolve
    relative to the declaring org's dir. Returns ``()`` when no org in the chain
    declares ``extra_prompt_parts``. Mirrors ``orgs._load_extra_parts_for_scope``
    but takes ``project_root`` explicitly (no CWD dependency, no harness import)."""
    import yaml

    from pux_harness.kit._paths import search_org_dir
    from pux_harness.kit.loaders import _resolved_org_chain
    from pux_harness.agent.prompt_parts import build_extra_parts

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


def format_prompt_with_provenance(
    parts: list[RenderedPart],
    *,
    scope_label: str,
) -> str:
    """Human-readable rendering: each part with a header (name, source, condition,
    char count) followed by its content. Conditional/inactive parts show their
    trigger instead of content."""
    lines: list[str] = [f"=== {scope_label} ===", ""]
    total_chars = 0
    active = 0
    conditional = 0
    n = len(parts)
    for i, part in enumerate(parts, 1):
        is_active = part.content is not None
        if is_active:
            active += 1
            total_chars += len(part.content)  # type: ignore[arg-type]
        else:
            conditional += 1

        bar = "─" * 56
        lines.append(f"── part {i}/{n}: {part.name} {bar}")
        lines.append(f"   source: {part.source}")
        status = "ACTIVE" if is_active else "CONDITIONAL"
        lines.append(f"   status: {status} — {part.condition}")
        if is_active:
            lines.append(f"   chars: {len(part.content):,}")  # type: ignore[arg-type]
            lines.append("")
            lines.append(part.content)  # type: ignore[arg-type]
        else:
            lines.append("   (not emitted at static time — depends on runtime state)")
        lines.append("")

    lines.append(
        f"=== TOTAL: {total_chars:,} chars "
        f"({active} active, {conditional} conditional) ==="
    )
    return "\n".join(lines)


def format_prompt_raw(parts: list[RenderedPart]) -> str:
    """Just the assembled text — what the model would see if all conditional
    parts were off (the common static case). Equivalent to assemble_prompt()."""
    from pux_harness.agent.prompt_parts import assemble_prompt

    chunks = [p.content for p in parts if p.content is not None]
    return "\n\n".join(chunks)


def show_supervisor(
    org: str, project_root: Path, *, raw: bool = False,
    ask_user: bool = False, interpreter: bool = False,
) -> str:
    """Render the supervisor (CTO) prompt for ``org``.

    ``ask_user`` / ``interpreter`` simulate the runtime-on state for the two
    conditional parts (fed by the ``--with-ask-user`` / ``--with-interpreter``
    CLI flags) so the experimenter can preview what those parts would emit
    without running over a real transport."""
    ctx = _build_supervisor_ctx(
        org, project_root, ask_user=ask_user, interpreter=interpreter
    )
    extra = _resolve_extra_parts(org, project_root, PromptScope.SUPERVISOR)
    parts = render_parts(PromptScope.SUPERVISOR, ctx, extra=extra)
    if raw:
        return format_prompt_raw(parts)
    return format_prompt_with_provenance(
        parts, scope_label=f"SUPERVISOR (CTO) prompt — org {org!r}"
    )


def show_subagent(
    org: str, slug: str, project_root: Path, *, raw: bool = False,
) -> str:
    """Render one subagent's prompt for ``org``."""
    ctx, error = _build_subagent_ctx(org, slug, project_root)
    if error:
        return f"ERROR: {error}"
    extra = _resolve_extra_parts(org, project_root, PromptScope.SUBAGENT)
    parts = render_parts(PromptScope.SUBAGENT, ctx, extra=extra)
    if raw:
        return format_prompt_raw(parts)
    return format_prompt_with_provenance(
        parts, scope_label=f"SUBAGENT prompt — org {org!r}, agent {slug!r}"
    )
