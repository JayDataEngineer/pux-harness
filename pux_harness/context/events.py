"""Backward-compat re-export shim.

The real implementation now lives in the ``deepagents-context`` package
(``deepagents_context.store``). This shim keeps existing
``from pux_harness.context.events import ...`` calls working while consumers
migrate to importing directly from ``deepagents_context``.
"""
from deepagents_context.store import *  # noqa: F401,F403
from deepagents_context.store import (  # noqa: F401
    EventStore,
    P1,
    P2,
    P3,
    SearchHit,
    StashResult,
    shared_event_store,
)
