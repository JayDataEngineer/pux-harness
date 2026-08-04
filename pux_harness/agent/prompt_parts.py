"""Constructed-from-parts system-prompt assembly.

The supervisor/CTO prompt and each subagent prompt are assembled by joining an
ORDERED REGISTRY of named ``PromptPartSpec`` parts — the prompt analogue of
``stack.MIDDLEWARE_REGISTRY``. This replaces the scattered ad-hoc string ops that
used to live inline in ``build_stack`` (``build_system_prompt`` → ``base_system_prompt``
REPLACE → ``system_prompt_suffix`` → ``ask_user`` suffix → ``dynamic`` suffix).

**The no-gap property.** A part's ``build(ctx)`` returns the chunk to emit, or
``None`` when its condition is OFF (→ the part is skipped, never an error). "Is
there a gap?" reduces to "is every part registered with an explicit condition?" —
there is no global-REPLACE anywhere in the assembler. The old ``base_system_prompt``
nuclear-replace is GONE (a permanent contract failure — see
``profile.validate_profile``).

**The supervisor/subagent boundary (load-bearing).** The two registries are
DISJOINT: ``SUBAGENT_PROMPT_PARTS`` contains ``agents_md_core`` / ``harness_addendum``
/ ``dynamic_dispatch_suffix`` NOWHERE. A subagent never sees the base-org overlay,
the orchestrator pattern, the harness addendum, or the dynamic-dispatch notice —
only its OWN specialization body + optional suffixes. (Verified as today's behavior;
this module makes it structural.)

**Byte-identical to the pre-refactor assembly.** Parts join with ``"\\n\\n"``
(exactly today's ``f"{prompt}\\n\\n{suffix}"`` append semantics) EXCEPT the
``agents_md_core`` part, which folds the harness addendum in via its own baked
leading ``"\\n"`` seam — matching today's ``build_system_prompt`` byte-for-byte
(the overlay→addendum seam is a single ``"\\n"``, so the addendum is an ingredient
of the core part, not a separately-joined part). Conditions are PRECOMPUTED by the
caller (``build_stack`` / ``load_subagents``) and passed via ``PromptCtx`` flags, so
this module owns ONLY string assembly — no profile/RuntimeFacts/middleware coupling,
no import cycle back into ``stack``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

from pux_harness.agent.hitl import ASK_USER_PROMPT_SUFFIX

# --- the two prompt scopes (mirrors ``stack.Scope``) -------------------------
# SUPERVISOR = the CTO/orchestrator driver; SUBAGENT = a delegated specialist.
# A part's ``scope`` declares which prompt(s) it may appear in; ``assemble_prompt``
# filters by the requested scope. The registries are DISJOINT by construction (no
# part is scoped to BOTH), so a supervisor part can never leak into a subagent.


class PromptScope(Enum):
    SUPERVISOR = "supervisor"
    SUBAGENT = "subagent"


@dataclass(frozen=True)
class PromptPartSpec:
    """ONE named prompt part. Position in its registry == position in the output.
    ``build`` returns the chunk to emit, or ``None`` when the part's condition is
    OFF (the part is then skipped — never an error)."""

    name: str
    scope: frozenset[PromptScope]
    build: Callable[["PromptCtx"], str | None]


@dataclass(frozen=True)
class PromptCtx:
    """Everything a part builder needs. The caller (``build_stack`` for the
    supervisor, ``load_subagents`` per subagent) PRECOMPUTES the gate flags, so
    this struct carries only primitives + strings — no profile/facts/middleware
    objects, keeping the module free of an import cycle into ``stack``.

    Supervisor-relevant fields: ``agents_md_base`` / ``system_prompt_suffix``
    / ``ask_user_active`` / ``interpreter_mounted``.
    Subagent-relevant fields: ``agent_body`` / ``system_prompt_suffix`` /
    ``agent_system_prompt_suffix``. Unused fields stay at their defaults.
    """

    # --- supervisor ---
    agents_md_base: str = ""  # chain-inherited org overlay (base org `general` + own; from orgs.build_system_prompt)
    harness_addendum: str = ""  # orgs/_shared/harness_addendum.md body; empty = no file → builder falls back to _ADDENDUM
    system_prompt_suffix: str | None = None  # org-wide suffix (supervisor + subagent)
    ask_user_active: bool = False
    ask_user_text: str = ""  # orgs/_shared/ask_user_suffix.md body; empty = no file → builder falls back to ASK_USER_PROMPT_SUFFIX
    interpreter_mounted: bool = False
    dynamic_dispatch_text: str = ""  # orgs/_shared/dynamic_dispatch_suffix.md body; empty = no file → builder falls back to _DYNAMIC_DISPATCH_SUFFIX
    # --- subagent ---
    agent_body: str = ""
    agent_system_prompt_suffix: str | None = None  # per-agent suffix


def assemble_prompt(
    parts: Sequence[PromptPartSpec], ctx: PromptCtx, scope: PromptScope,
) -> str:
    """Join the parts whose scope matches AND whose ``build`` returns non-None, in
    registry order, with ``"\\n\\n"`` between them. Skipped parts (scope mismatch or
    ``None`` build) leave no trace — the no-gap property. ``agents_md_core`` /
    ``agent_body`` are always-on, so the result is never empty in practice."""
    out: list[str] = []
    for spec in parts:
        if scope not in spec.scope:
            continue
        chunk = spec.build(ctx)
        if chunk is None:
            continue
        out.append(chunk)
    return "\n\n".join(out)


# --- the harness addendum (moved verbatim from ``orgs._ADDENDUM``) ------------
# Folded into ``agents_md_core`` (NOT a separately-joined part): the overlay→addendum
# seam is a single ``"\n"`` (this constant's OWN leading newline), which the
# ``"\n\n"`` joiner cannot reproduce without a spurious blank line. Keeping the
# addendum an ingredient of the core part preserves byte-identical assembly.
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


def resolve_harness_addendum(body: str) -> str:
    """Normalize a frontmatter-stripped body into the addendum WITH the leading
    ``"\\n"`` seam (the single-newline join to the AGENTS.md overlay — the
    ``"\\n\\n"`` joiner cannot reproduce it without a spurious blank line) and a
    single trailing newline (byte-identical to the embedded ``_ADDENDUM``).
    Falls back to ``_ADDENDUM`` when ``body`` is empty (file absent — minimal
    fixtures / packed archives that omit ``_shared/``)."""
    return f"\n{body}\n" if body else _ADDENDUM


def load_harness_addendum(project_root: Path) -> str:
    """Read ``orgs/_shared/harness_addendum.md`` -> the addendum string WITH the
    leading ``"\\n"`` seam (ready for ``ctx.harness_addendum``). Falls back to
    the embedded ``_ADDENDUM`` when the file is absent. This is the ONE function
    both the runtime (``stack.build_stack``) and the introspection view
    (``prompt_show``) call to populate ``ctx.harness_addendum`` — CWD-independent
    (takes ``project_root`` explicitly)."""
    from pux_harness.kit.loaders import load_shared_prompt_body

    return resolve_harness_addendum(load_shared_prompt_body("harness_addendum.md", project_root))


def load_ask_user_suffix(project_root: Path) -> str:
    """Read ``orgs/_shared/ask_user_suffix.md`` -> the ask-user turn-ending
    suffix (ready for ``ctx.ask_user_text``). Falls back to the embedded
    ``ASK_USER_PROMPT_SUFFIX`` when the file is absent. Same pattern as
    ``load_harness_addendum`` — CWD-independent."""
    from pux_harness.kit.loaders import load_shared_prompt_body

    body = load_shared_prompt_body("ask_user_suffix.md", project_root)
    return body or ASK_USER_PROMPT_SUFFIX


def load_dynamic_dispatch_suffix(project_root: Path) -> str:
    """Read ``orgs/_shared/dynamic_dispatch_suffix.md`` -> the eval-tool dispatch
    strategy (ready for ``ctx.dynamic_dispatch_text``). Falls back to the
    embedded ``_DYNAMIC_DISPATCH_SUFFIX`` when the file is absent. Same pattern
    as ``load_harness_addendum`` — CWD-independent."""
    from pux_harness.kit.loaders import load_shared_prompt_body

    body = load_shared_prompt_body("dynamic_dispatch_suffix.md", project_root)
    return body or _DYNAMIC_DISPATCH_SUFFIX


def build_extra_parts(
    entries: Sequence[Any],
    org: str,
    org_dir: Path,
    scope: "PromptScope",
) -> tuple["PromptPartSpec", ...]:
    """Build always-on ``PromptPartSpec`` instances from a profile.yaml
    ``extra_prompt_parts:`` list, filtered to ``scope``. Each entry:

    ``{name: <str>, file: <rel-path>, scope: [supervisor, subagent]}``

    * ``name`` — unique provenance label (appears in ``pux prompt show``).
    * ``file`` — path relative to the org's own directory (e.g.
      ``extra/my_section.md`` -> ``orgs/specialists/<org>/extra/my_section.md``).
    * ``scope`` — optional list of ``"supervisor"`` and/or ``"subagent"``
      (default: both). Controls which prompt(s) the part appears in.

    Content is the file body, read ONCE at build time and captured in a closure
    (always-on — conditional logic still requires Python; this is the
    file-injection escape hatch for experimenters who need a named, ordered,
    scope-targeted section without touching code). Returns ``()`` when
    ``entries`` is empty / no entry matches ``scope``.

    THIS function owns only the list->specs transform; the CALLER resolves the
    chain-merged ``extra_prompt_parts`` list (runtime: ``_resolved_profile_yaml``;
    introspection: chain walk) and the org directory path.
    """
    scope_name = scope.value  # "supervisor" | "subagent"
    out: list[PromptPartSpec] = []
    for entry in entries:
        if not isinstance(entry, dict):
            msg = f"{org}: extra_prompt_parts: entry must be a mapping, got {type(entry).__name__}"
            raise TypeError(msg)
        name = entry.get("name")
        file_rel = entry.get("file")
        if not name or not file_rel:
            msg = (f"{org}: extra_prompt_parts: each entry needs `name` and `file`, "
                   f"got {entry!r}")
            raise ValueError(msg)
        scopes = entry.get("scope", ["supervisor", "subagent"])
        if scope_name not in scopes:
            continue
        file_path = org_dir / file_rel
        if not file_path.is_file():
            msg = (f"{org}: extra_prompt_parts: `file: {file_rel}` not found "
                   f"(resolved to {file_path})")
            raise FileNotFoundError(msg)
        body = file_path.read_text()
        # Capture body in the default arg (closure-safe across loop iterations).
        out.append(PromptPartSpec(
            name=name,
            scope=frozenset({scope}),
            build=lambda ctx, b=body: b,
        ))
    return tuple(out)


# --- the dynamic-dispatch upgrade notice + its mount detector ----------------
# Moved from ``stack.py`` (this is prompt content). The detector is structural
# (qualified class name) so a weak build never loads quickjs just to ask the
# question — the caller precomputes ``interpreter_mounted`` and passes the flag.


def _interpreter_mounted(middleware: Sequence[object]) -> bool:
    """True iff the dynamic-subagent ``CodeInterpreterMiddleware`` is actually in
    ``middleware`` — the post-resolution signal that this orchestrator was given the
    ``eval`` tool (a strength-``pro`` base, or an explicit ``add: [interpreter]``
    override). Honors ``add``/``remove`` (a strong base with ``remove:[interpreter]``
    reports False), so the dynamic-prompt block tracks the REAL mount decision.

    Detected by QUALIFIED class name — NOT an ``isinstance`` import — so the many
    weak-model builds that never mount the interpreter never load quickjs/wasmtime
    here either. That keeps the strength gate a real PERF gate (not just a mount
    gate): a flash orchestrator pays zero native-load cost, end to end. The class
    name + module root are a stable upstream contract (langchain-quickjs surfaces
    exactly ``CodeInterpreterMiddleware``); a rename surfaces first in
    ``test_build_interpreter_returns_eval_exposing_middleware``."""
    for m in middleware:
        t = type(m)
        if (
            getattr(t, "__name__", "") == "CodeInterpreterMiddleware"
            and t.__module__.split(".", 1)[0] == "langchain_quickjs"
        ):
            return True
    return False


_DYNAMIC_DISPATCH_SUFFIX = """\
## Dynamic dispatch (you are interpreter-enabled)

You have the ``eval`` tool — a sandboxed JS REPL — so you can drive the
**dynamic** happy path. For ANY multi-unit task, PREFER it over the
static ``task``-one-call-at-a-time flow above:

- ``eval`` runs ONE short dispatch script. ``task({subagentType, description})``
  dispatches a subagent and returns its response; ``Promise.all([...])`` fans
  workers out in parallel; ``tools.glob`` / ``tools.grep`` / ``tools.ls`` /
  ``tools.read_file`` do read-only discovery without a round-trip per call.
- The happy path becomes: recon via an explorer ``task``, INLINE its report into
  each worker ``description``, fan the workers out with ``Promise.all``, return
  the synthesis as the script's value.
- KEEP YOUR THREAD LEAN — that is the whole point. You hold only the dispatch
  logic + the final result; the explorers / workers absorb the file contents and
  the context blow. Do NOT read the explored files into your own thread — inline
  the explorer's report into the worker calls instead. Hoarding context on the
  dynamic path duplicates the explorer's work in your thread, which is the very
  token cost dynamic dispatch exists to avoid.

The ``eval`` tool's own description + the injected ``task()`` / PTC guide carry
the exact JS API — follow them; do not invent a different shape. (Note:
``task()`` dispatches inside the already-approved ``eval`` and bypasses parent
HITL approval per dispatch — by design.)"""


_SUPER = frozenset({PromptScope.SUPERVISOR})
_SUB = frozenset({PromptScope.SUBAGENT})

# --- SUPERVISOR registry (the CTO prompt) -----------------------------------
# Order == output order. Maps today's build_stack prompt block 1:1 (minus the dead
# base_system_prompt REPLACE). Part 1 folds root+overlay+addendum (the static base);
# parts 2-4 are conditional suffixes joined with "\n\n" (today's append semantics).
SUPERVISOR_PROMPT_PARTS: tuple[PromptPartSpec, ...] = (
    PromptPartSpec(
        name="agents_md_core",
        scope=_SUPER,
        build=lambda ctx: f"{ctx.agents_md_base}{ctx.harness_addendum or _ADDENDUM}",
    ),
    PromptPartSpec(
        name="org_system_prompt_suffix",
        scope=_SUPER,
        build=lambda ctx: ctx.system_prompt_suffix,
    ),
    PromptPartSpec(
        name="ask_user_suffix",
        scope=_SUPER,
        build=lambda ctx: (ctx.ask_user_text or ASK_USER_PROMPT_SUFFIX) if ctx.ask_user_active else None,
    ),
    PromptPartSpec(
        name="dynamic_dispatch_suffix",
        scope=_SUPER,
        build=lambda ctx: (ctx.dynamic_dispatch_text or _DYNAMIC_DISPATCH_SUFFIX) if ctx.interpreter_mounted else None,
    ),
)

# --- SUBAGENT registry (a delegated specialist's prompt) ---------------------
# Order == output order. NO supervisor content: a subagent gets its OWN body +
# the org-wide suffix + its own per-agent suffix — never the base org overlay,
# the orchestrator pattern, the harness addendum, or the dynamic-dispatch notice.
# (The user's hard rule: subagents are SPECIALIZED for independent tasks.)
SUBAGENT_PROMPT_PARTS: tuple[PromptPartSpec, ...] = (
    PromptPartSpec(
        name="agent_body",
        scope=_SUB,
        build=lambda ctx: ctx.agent_body,
    ),
    PromptPartSpec(
        name="org_system_prompt_suffix",
        scope=_SUB,
        build=lambda ctx: ctx.system_prompt_suffix,
    ),
    PromptPartSpec(
        name="agent_system_prompt_suffix",
        scope=_SUB,
        build=lambda ctx: ctx.agent_system_prompt_suffix,
    ),
)


__all__ = [
    "PromptCtx",
    "PromptPartSpec",
    "PromptScope",
    "SUPERVISOR_PROMPT_PARTS",
    "SUBAGENT_PROMPT_PARTS",
    "_DYNAMIC_DISPATCH_SUFFIX",
    "_interpreter_mounted",
    "assemble_prompt",
    "build_extra_parts",
    "load_ask_user_suffix",
    "load_dynamic_dispatch_suffix",
    "load_harness_addendum",
    "resolve_harness_addendum",
]
