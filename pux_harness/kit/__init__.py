"""pux_harness.kit — the portable org + skill compiler (the slim core).

This is the slim, Docker-free core of the pux agent system: it turns a folder
of org + skills into a running deepagents agent with NO sandbox, NO context/
memory/browser-vision middleware, NO server — just deepagents + a local
``FilesystemBackend``. The kit is positively defined (the org+skill compiler),
not as "the harness minus Docker" — everything heavy attaches to ``pux-harness``
as an optional-dependency extra, and a lazy-import + tripwire keep this module's
import graph slim (``import pux_harness.kit`` pulls neither ``docker`` nor any
heavy ``pux_harness.<subsystem>``). The typical consumer is a different,
standalone project (e.g. a Wan2GP + CopilotKit app) that authors its own org +
a skill and wires the compiled graph to its UI.

Quick start::

    from pux_harness import compile_org
    from langgraph.checkpoint.memory import MemorySaver

    graph = compile_org(
        "my_org",
        model=my_chat_model,
        tools=[my_wan2gp_tool],
        project_root="./my_app",
        checkpointer=MemorySaver(),
    )
    graph.invoke({"messages": [{"role": "user", "content": "..."}]})

See ``examples/README.md`` for the org + skill format and a full walk-through.
"""
from __future__ import annotations

from ._paths import project_root
from .compile import compile_org, load_subagents
from .loaders import (
    build_system_prompt,
    discover_orgs,
    load_org_prompt,
    load_root_prompt,
    org_agent_slugs,
)

__all__ = [
    "compile_org",
    "load_subagents",
    "discover_orgs",
    "org_agent_slugs",
    "load_root_prompt",
    "load_org_prompt",
    "build_system_prompt",
    "project_root",
]

__version__ = "0.1.0"
