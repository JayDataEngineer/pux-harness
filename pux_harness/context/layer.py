"""Pux adapter for ``build_context_layer`` — injects sandbox exec tools.

The real implementation lives in ``deepagents_context.layer``. This adapter
keeps the pux-facing ``sandbox`` parameter (a ``BaseSandbox``) and
converts it to ``extra_tools`` for the package version, keeping all existing
pux call sites unchanged.
"""
from __future__ import annotations

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool

from deepagents.backends.sandbox import BaseSandbox
from deepagents_context import EventStore, shared_event_store
from deepagents_context.layer import build_context_layer as _build_context_layer


def build_context_layer(
    store: EventStore | None = None,
    *,
    threshold: int | None = None,
    preview: int | None = None,
    enabled: bool = True,
    sandbox: BaseSandbox | None = None,
) -> tuple[list[AgentMiddleware], list[BaseTool]]:
    """Pux-facing ``build_context_layer`` — same signature as before.

    When ``sandbox`` is provided, the 4 sandbox exec tools
    (ctx_execute/ctx_execute_file/ctx_batch_execute/ctx_fetch_and_index) are
    built and passed as ``extra_tools`` to the package version.
    """
    extra_tools: list[BaseTool] | None = None
    if sandbox is not None:
        from pux_harness.context.exec_tools import build_exec_tools  # noqa: PLC0415
        s = store or shared_event_store()
        extra_tools = build_exec_tools(s, sandbox)
    return _build_context_layer(
        store,
        threshold=threshold,
        preview=preview,
        enabled=enabled,
        extra_tools=extra_tools,
    )
