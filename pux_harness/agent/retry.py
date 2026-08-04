"""Stream-stall retry — the LangGraph-native layer that makes a stalled
upstream model stream transparent.

Background
----------
``super().prompt()`` runs LangGraph's astream loop with NO try/except of its
own. When the upstream model stream stalls (TCP alive, provider silent —
langchain-openai raises ``StreamChunkTimeoutError`` after ``stream_chunk_timeout``
seconds, a subclass of ``asyncio.TimeoutError``), the exception walks straight
out and (without the wrappers in ``pux_harness.acp``) freezes the editor.

The PREVIOUS fix at the ``acp.py`` prompt boundary (commit b1d67f4) catches the
stall in a wrapper retry loop. That fix is correct about hiding the stall from
the caller, but its resume semantic is wrong: re-entering ``super().prompt()``
re-passes the user prompt as NEW input, which causes LangGraph to APPEND the
message and re-run the turn from the start. Prior nodes' work in the failing
turn is overwritten, not preserved.

The RIGHT layer is the LangGraph-native node-level RetryPolicy. When attached
to the ``model`` node of a deepagents graph, LangGraph retries the stalled
NODE in-place — the checkpointer preserves every node that completed before
the stall, the stalled node picks up from its own beginning (no input change),
and the rest of the turn continues normally. No wrapper-level re-entry, no
duplicate work, no lost work.

This module owns the shared classifier (used by both ``acp.py`` and
``graph.py``) and the policy factory.
"""
from __future__ import annotations

import asyncio

from langgraph.types import RetryPolicy

# How many times LangGraph re-enters a stalled node before giving up.
# Tuned so 4 × 120s (``stream_chunk_timeout``) ≈ 8 min of patience for the
# provider to recover before surfacing the end_turn + resume notice.
_STREAM_STALL_MAX_ATTEMPTS = 4
# Backoff base in seconds. Production uses 2.0 (2s, 4s, 8s) — JITTER is on by
# default in RetryPolicy, so the actual intervals vary slightly to avoid
# thundering-herd re-connects when many sessions stall at once.
_STREAM_STALL_INITIAL_INTERVAL = 2.0


def retry_on_stream_stall(exc: BaseException) -> bool:
    """True iff ``exc`` is a transient stream stall worth retrying.

    A stall = the upstream model stream went silent (TCP alive, provider not
    sending tokens) OR a transient connection / rate-limit blip. Re-entering
    the node retries the model call from the START of the node's execution —
    the checkpointer guarantees nodes that completed before the stall are NOT
    re-run, so retry is safe AND productive.

    Deterministic errors (``ValidationError``, ``TypeError``,
    ``AttributeError``, auth failures, schema mismatches) return ``False``:
    they will not change shape across attempts, and retrying only delays the
    inevitable end_turn.

    This is the SINGLE classifier — both ``pux_harness.acp._is_stream_stall_recoverable``
    (the prompt-boundary fallback) and the LangGraph ``RetryPolicy(retry_on=...)``
    on the model node use this exact predicate.
    """
    # Tool-side timeouts are deterministic — NEVER retry. The sandbox's
    # ``ExecTimeout`` is raised when a single tool command exceeds the 120s
    # wall-clock budget. Its message ("exec timed out after 120s: ...")
    # contains both "timed out" and "timeout", which the substring fallback
    # at the bottom of this predicate would otherwise match — sending
    # LangGraph into 4 × 120s of useless retries on the SAME tool call
    # before surfacing the misleading "⚠️ model stream stalled" banner.
    #
    # Defense-in-depth: ``ctx_execute`` / ``ctx_execute_file`` /
    # ``ctx_batch_execute`` / ``ctx_fetch_and_index`` all catch ExecTimeout
    # at the tool boundary and convert it to a result envelope, so this
    # branch should never fire for those. But other surfaces (raw fs
    # scripts, dynamic tools, browser tools) may still let it escape, and
    # this is the SINGLE classifier — pin it here so the contract holds
    # regardless of which tool raised.
    from pux_harness.sandbox.docker_exec import ExecTimeout as _SandboxExecTimeout
    if isinstance(exc, _SandboxExecTimeout):
        return False

    # ``StreamChunkTimeoutError`` (langchain-openai) subclasses
    # ``asyncio.TimeoutError`` per the upstream source — the most common
    # concrete trigger for the "⚠️ This turn ended early" symptom.
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, (ConnectionError, asyncio.IncompleteReadError)):
        return True
    name = (type(exc).__name__ or "").lower()
    msg = str(exc).lower()
    recoverable_classes = {
        "apiconnectionerror", "apitimeouterror", "internalservererror",
        "ratelimiterror", "apierror", "readtimeouterror", "readerror",
        "remoteprotocolerror", "protocolerror", "streamchunktimeouterror",
    }
    if name in recoverable_classes:
        return True
    # ``BadRequest`` is normally deterministic — but provider streams sometimes
    # surface ``bad_request: stream stalled`` / ``... timeout`` which ARE
    # transient. Only retry those narrow messages.
    if "badrequest" in name:
        return any(t in msg for t in ("stream", "stall", "timeout", "connection"))
    return any(t in msg for t in (
        "stream stalled", "connection reset", "timed out", "timeout",
        "temporarily", "overloaded", "rate limit", "too many requests",
        "503", "502", "500",
    ))


def stream_stall_retry_policy(
    *, max_attempts: int = _STREAM_STALL_MAX_ATTEMPTS,
    initial_interval: float = _STREAM_STALL_INITIAL_INTERVAL,
) -> RetryPolicy:
    """Build the RetryPolicy attached to the deepagents ``model`` node.

    Returns a SINGLE RetryPolicy instance. Callers that assign to
    ``graph.nodes["model"].retry_policy`` MUST wrap it in a list:
    ``graph.nodes["model"].retry_policy = [policy]`` — LangGraph's runtime
    iterates ``retry_policy`` as ``Sequence[RetryPolicy]``, and iterating a
    bare ``RetryPolicy`` (a NamedTuple) yields its field values (floats,
    ints, the ``retry_on`` callable), which then crashes
    ``_should_retry_on`` with ``AttributeError: 'float' object has no
    attribute 'retry_on'``. ``add_node(retry_policy=...)`` normalizes single
    → ``[single]`` internally; direct post-compile mutation does NOT.
    """
    return RetryPolicy(
        retry_on=retry_on_stream_stall,  # type: ignore[arg-type]
        max_attempts=max_attempts,
        initial_interval=initial_interval,
        backoff_factor=2.0,
        jitter=True,
    )


def attach_stream_stall_retry(graph: Any, node_names: tuple[str, ...] = ("model",)) -> Any:
    """Attach the stream-stall RetryPolicy to ``node_names`` on a compiled graph.

    Idempotent: re-attaching replaces the policy (no accumulation). Returns
    the same graph (mutated in place — deepagents' create_deep_agent result
    is mutable per node). Skips names that aren't present (a no-skills org
    has a different node set; failing loudly on a missing node would block
    legitimate org shapes).

    The policy is wrapped in a LIST per the contract above.
    """
    policy = [stream_stall_retry_policy()]
    for name in node_names:
        node = graph.nodes.get(name) if hasattr(graph, "nodes") else None
        if node is not None:
            node.retry_policy = policy
    return graph


# Avoid the Any import at module top — used only in the type hint above.
from typing import Any  # noqa: E402
