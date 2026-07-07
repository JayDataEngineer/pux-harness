"""Langfuse tracing for the deepagents graph — optional, no-op unless configured.

The ONE langgraph invoke-config builder for both graph-invoke sites (``main.py``
for ``pux direct`` and ``server.py`` for ``pux serve``). When Langfuse is
installed AND the ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` env vars are
set, every run is traced to a Langfuse session (one session per ``thread_id``,
tagged with the org + transport). When either is absent,
``build_invoke_config`` returns the plain config the graph already used — zero
behavior change.

Rely-on-upstream: we attach langfuse's own LangChain ``CallbackHandler`` to the
langgraph config ``callbacks`` — no hand-rolled tracing. langgraph fans that
callback across every node / LLM / tool call in the deepagents graph
automatically, so plans, delegations, tool calls, and token cost are all
captured.

A down / unreachable Langfuse host NEVER blocks a run: the handler is
best-effort and queues locally. We gate on env PRESENCE (not a live auth check)
precisely so a configured-but-down host can't take a run down with it.

ACP (``acp.py``) is NOT wired here: deepagents-acp owns graph invocation and
exposes no per-thread ``session_id`` seam at build time. Documented as a known
gap (Phase D live proof covers the ``direct`` + ``serve`` lanes).
"""
from __future__ import annotations

import os
from typing import Any

# Guarded import: langfuse is the first OPTIONAL extra (``pip install
# pux-harness[observability]``). The base install does NOT pull it in, so this
# module must never fail to import when langfuse is absent — both invoke sites
# import ``build_invoke_config`` unconditionally.
try:
    from langfuse.langchain import CallbackHandler as _LangfuseHandler

    _HAS_LANGFUSE = True
except ImportError:  # pragma: no cover - exercised via the env-gated live proof
    _LangfuseHandler = None  # type: ignore[assignment,misc]
    _HAS_LANGFUSE = False


def _env_configured() -> bool:
    """True only when both Langfuse credentials are present in the env.

    Presence (not a live auth check) on purpose: a configured-but-down host must
    never block a run — the handler queues locally and flushes best-effort.
    """
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
    )


def langfuse_handler(org: str, transport: str, thread_id: str) -> Any | None:
    """A Langfuse ``CallbackHandler`` for one run, or ``None`` (no-op).

    ``None`` (run untraced) when langfuse is not importable OR the credential
    env vars are unset. ``transport`` tags the surface (``direct`` / ``serve``);
    the ``thread_id`` becomes the Langfuse ``session_id`` so a resumed thread is
    a single session in the UI.
    """
    if not _HAS_LANGFUSE or not _env_configured():
        return None
    return _LangfuseHandler(  # type: ignore[misc]
        session_id=thread_id,
        tags=[f"org:{org}", f"transport:{transport}"],
        metadata={"org": org, "transport": transport, "thread_id": thread_id},
    )


def build_invoke_config(
    thread_id: str,
    recursion_limit: int,
    org: str,
    transport: str,
) -> dict[str, Any]:
    """The ONE langgraph invoke-config builder for both invoke sites.

    Identical to the prior inline dict (``configurable.thread_id`` +
    ``recursion_limit``) when Langfuse is off; adds ``callbacks`` when on. Use
    from ``main.py`` (transport ``"direct"``) and ``server.py`` (transport
    ``"serve"``).
    """
    cfg: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    handler = langfuse_handler(org, transport, thread_id)
    if handler is not None:
        cfg["callbacks"] = [handler]
    return cfg
