"""Compile an org + its skills into a running deepagents agent — no Docker.

``compile_org(org, ...)`` is the kit's ONE entry point for a standalone consumer
app (e.g. Wan2GP + CopilotKit). It reads the org off the local filesystem via
:mod:`pux_harness.kit.loaders`, resolves the specialist roster + skills sources,
and binds everything through deepagents' ``create_deep_agent`` against a LOCAL
``FilesystemBackend`` (deepagents ships this — no sandbox container required).

What the consumer supplies:
- ``model``  — any ``BaseChatModel`` (or a model id string).
- ``tools``  — the app's own tools (e.g. a ``generate_wan2gp_form`` tool). These
  are the agent's real surface; the kit adds no ``pux_sandbox_*`` tools.

What the kit does NOT bring (deliberately): the pux Docker sandbox, the pux
context/memory/browser-vision middleware, the rubric gate, profile overrides.
A consumer that wants any of those uses the pux harness directly. The kit is
for authoring + running an org in a different project with none of that baggage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from ._paths import project_root as _default_project_root
from .loaders import (
    _load_agent_spec,
    _agent_search_dirs,
    _parse_list,
    _resolve_skills,
    build_system_prompt,
    org_agent_slugs,
)


def _resolve_tools(raw: Any, tool_map: dict[str, BaseTool]) -> list[BaseTool]:
    """Map an agent's ``tools`` list to the consumer's tools by EXACT name.

    Unlike the harness (which classifies slugs into native/specialist
    categories against the pux tool registry), the kit has no tool registry —
    the consumer's ``tools`` ARE the universe. A listed name that isn't in the
    consumer's tool set is SKIPPED (not raised): an org authored under the pux
    harness may reference ``pux_sandbox_*`` tools a standalone consumer doesn't
    ship, and a strict raise would make those orgs un-compilable here. The
    slug's trailing ``/path`` segment is stripped first (``shared/researcher``
    -> ``researcher``) so shared-agent references resolve.
    """
    resolved: list[BaseTool] = []
    for entry in _parse_list(raw):
        name = entry.rsplit("/", 1)[-1]
        tool = tool_map.get(name)
        if tool is not None and tool not in resolved:
            resolved.append(tool)
    return resolved


def _build_sub(
    slug: str,
    spec: dict[str, Any],
    tool_map: dict[str, BaseTool],
    *,
    model_resolver: Callable[[str | None], BaseChatModel | str],
    project_root: Path,
) -> dict[str, Any]:
    """Build a deepagents SubAgent dict from a spec, kit-style.

    ``model_resolver`` maps a frontmatter ``model:`` override (or ``None``) to a
    model. The default resolver (see :func:`load_subagents`) always returns the
    supervisor model — the kit doesn't know about model roles, so every
    specialist runs on the same driver unless the consumer overrides.
    """
    sub: dict[str, Any] = {
        "name": spec.get("name", slug),
        "description": spec.get("description", slug),
        "system_prompt": spec["system_prompt"],
    }
    if spec.get("tools"):
        sub["tools"] = _resolve_tools(spec["tools"], tool_map)
    sub["model"] = model_resolver(spec.get("model"))
    if "skills" in spec:
        sub["skills"] = _resolve_skills(spec["skills"], slug, project_root=project_root)
    return sub


def load_subagents(
    org: str,
    tools: Sequence[BaseTool],
    *,
    project_root: Path,
    model: BaseChatModel | str | None = None,
    model_resolver: Callable[[str | None], BaseChatModel | str] | None = None,
) -> list[dict[str, Any]]:
    """Build deepagents SubAgent dicts for ``org``'s specialists, kit-style.

    For each slug in ``org.yaml``, load ``orgs/<org>/agents/<slug>.md`` (or
    ``orgs/_shared/agents/<slug>.md``) and build from its frontmatter + body.
    No pux context layer, no profile, no tool-registry classification — just
    exact-name tool whitelisting against the consumer's ``tools``.
    """
    if model_resolver is None:
        # Default: every specialist runs on the supervisor model.
        model_resolver = lambda _override, _m=model: _m  # noqa: E731
    tool_map: dict[str, BaseTool] = {t.name: t for t in tools}
    subs: list[dict[str, Any]] = []
    for slug in org_agent_slugs(org, project_root):
        spec = _load_agent_spec(slug, org, project_root)
        if spec is None:
            searched = [str(p / f"{slug}.md") for p in _agent_search_dirs(org, project_root)]
            raise FileNotFoundError(
                f"no agent {slug!r} for org {org!r} — searched {searched}"
            )
        subs.append(
            _build_sub(slug, spec, tool_map, model_resolver=model_resolver, project_root=project_root)
        )
    return subs


def compile_org(
    org: str,
    *,
    model: BaseChatModel | str | None,
    tools: Sequence[BaseTool],
    middleware: Sequence[AgentMiddleware] = (),
    checkpointer: Any = None,
    project_root: Path | str | None = None,
    addendum: str | None = None,
    skills: list[str] | None = None,
    subagents: "list[dict[str, Any]] | None" = None,
    backend: Any = "filesystem",
) -> CompiledStateGraph:
    """Compile ``org`` into a deepagents ``CompiledStateGraph`` — standalone.

    Parameters
    - model: the supervisor/CTO driver model (``BaseChatModel`` or model id).
    - tools: the consumer's tools (the agent's real surface).
    - middleware: extra middleware to mount (e.g. CopilotKit/AG-UI surface
      middleware from the consumer app). Default none.
    - checkpointer: a langgraph checkpointer (e.g. ``MemorySaver``). Default
      ``None`` (ephemeral).
    - project_root: dir containing ``orgs/`` + the root ``AGENTS.md``. Defaults
      to ``$PUX_PROJECT_ROOT`` or the CWD.
    - addendum: extra text appended to the supervisor system prompt (after the
      org overlay). Default empty — the kit adds no harness-specific addendum.
    - skills: skills-ROOT paths for the SUPERVISOR (top-level ``skills=``).
      Default ``None`` — skills ride on subagents via their frontmatter.
    - subagents: pre-built SubAgent dicts. If ``None``, built from the org
      roster via :func:`load_subagents` against ``tools``.
    - backend: ``"filesystem"`` (default) -> a local ``FilesystemBackend`` rooted
      at ``project_root`` (skills resolve on the host fs; deepagents' own
      ``FilesystemMiddleware`` gives the agent local read/write/shell tools).
      Pass a ``BackendProtocol`` to customize, or ``None`` for no backend (no
      fs/shell tools, and skills sources won't resolve — supply your own).

    Returns a compiled graph; invoke with ``{"messages": [...]}``.
    """
    root = Path(project_root).resolve() if project_root is not None else _default_project_root()
    prompt = build_system_prompt(org, project_root=root, addendum=addendum or "")
    if subagents is None:
        subagents = load_subagents(org, tools, project_root=root, model=model)

    if backend == "filesystem":
        # ``virtual_mode=False`` (pinned) resolves paths against the real host fs
        # rooted at ``root`` — the local-app use case, where the agent reads/writes
        # the consumer's own project files. deepagents' default flips in 0.6.0; we
        # set it explicitly so the kit's behavior doesn't shift under a consumer.
        backend_obj: Any = FilesystemBackend(root_dir=root, virtual_mode=False)
    else:
        backend_obj = backend  # None or a caller-supplied BackendProtocol

    return create_deep_agent(
        model=model,
        system_prompt=prompt,
        tools=list(tools),
        subagents=subagents or None,
        skills=skills,
        middleware=list(middleware),
        backend=backend_obj,
        checkpointer=checkpointer,
    )
