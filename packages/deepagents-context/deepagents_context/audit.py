"""AuditMiddleware — observe-only tool-call audit, opt-in per org.

THE PURPOSE
    A selectable, default-OFF middleware that records EVERY tool call the agent
    makes — which tool, with what (hashed) args, the outcome, how long — into the
    shared ``EventStore`` (the same store ``ContextMiddleware`` captures into; no
    second DB, no second surface). It is the user's "selectively add the audit
    middleware" request, applied uniformly: opt in via ``middleware.supervisor.
    add: [audit]`` (or ``middleware.subagent.add: [audit]``).

OBSERVE-ONLY (the contract)
    This middleware NEVER intercepts, mutates, or replaces tool I/O. It calls the
    handler, records what happened, and returns the handler's result UNCHANGED —
    no ``Command``, no offload, no companion message. A failure to write the
    audit row is swallowed (an audit hook must NEVER break the agent's tool call)
    — the tool result always lands; the audit row is best-effort.

ARGS HASH, NOT RAW ARGS
    The audit row stores ``args_hash`` (sha256, 16 hex chars) — NEVER the raw
    args payload. Tool args routinely carry secrets (cookies, tokens, file
    contents) and the audit surface is for *observability* (correlating repeated
    calls, spotting the tool that ran 200×), not for reconstructing inputs. The
    hash lets two identical calls correlate without ever persisting their
    contents. This is deliberately stricter than ``ContextMiddleware``'s 500-char
    raw ``args`` preview, which lives in the activity feed for the agent's own
    recall — audit is a separate, append-only record for the operator.

COMPLETE LOG (defeating capture's dedup)
    ``EventStore.capture`` dedups by ``type + data_hash`` within a window — great
    for the activity feed (collapses noisy repeats), WRONG for an audit log
    (every call must appear). Each row carries a monotonic per-middleware ``seq``
    counter, so the data hash differs on every call and dedup never fires. ``seq``
    is also genuinely useful audit metadata (call ordering within a scope).

REUSE, NOT DUPLICATION
    One ``EventStore`` (``shared_event_store``), one ``capture`` path, one FTS5
    index — same as the context layer. Audit rows are ``type="tool_audit"``,
    ``category="audit"`` (so ``ctx_search``/``recent`` can filter them in or out).
    Audit rows participate in the store's normal lifecycle (they are events, not
    a separate table) — this is observability audit, not compliance-grade
    tamper-proofing; a hardened audit store is a separate concern.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from deepagents_context.store import EventStore, shared_event_store

# sha256 truncated to 16 hex chars (64 bits) — ample for call correlation,
# compact in the row. sort_keys on the JSON so identical args hash identically
# regardless of dict insertion order.
_ARGS_HASH_LEN = 16


def _hash_args(args: Any) -> str:
    """Stable 16-hex-char sha256 of the tool-call args. NEVER store the raw
    payload — see module docstring (secrets, observability-not-reconstruction)."""
    raw = json.dumps(args, ensure_ascii=False, default=str, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_ARGS_HASH_LEN]


def _outcome_of(result: Any) -> str:
    """``"ok"`` / ``"error"`` — derived from a ToolMessage's status when present
    (the framework sets ``status="error"`` on a raised/failed tool). A non-
    ToolMessage result (e.g. a raw string in tests) is treated as ok."""
    if isinstance(result, ToolMessage) and result.status == "error":
        return "error"
    return "ok"


class AuditMiddleware(AgentMiddleware):
    """Record every tool call into the shared ``EventStore`` (``type=tool_audit``)
    without ever touching the tool's result."""

    def __init__(
        self,
        store: EventStore | None = None,
        *,
        org: str = "",
        scope: str = "",
        enabled: bool = True,
    ) -> None:
        self.store = store or shared_event_store()
        self.org = org
        self.scope = scope
        self.enabled = enabled
        self._seq = 0  # monotonic per-middleware call counter (defeats dedup)

    # -- request introspection (mirrors ContextMiddleware — works for langchain's
    # ToolCallRequest or a SimpleNamespace stand-in in tests) ------------------

    @staticmethod
    def _tool_name(request: Any) -> str:
        tc = getattr(request, "tool_call", None) or {}
        if isinstance(tc, dict):
            name = tc.get("name")
            return str(name) if name is not None else "tool"
        return "tool"

    @staticmethod
    def _tool_args(request: Any) -> Any:
        tc = getattr(request, "tool_call", None) or {}
        return tc.get("args", {}) if isinstance(tc, dict) else {}

    @staticmethod
    def _thread_id(request: Any) -> str:
        state = getattr(request, "state", None) or {}
        if isinstance(state, dict):
            return state.get("configurable", {}).get("thread_id", "")
        return ""

    def _record(
        self, *, tool: str, args_hash: str, thread_id: str,
        outcome: str, elapsed: float, error: str = "",
    ) -> None:
        """Append one ``tool_audit`` row. Best-effort: a store failure is
        swallowed so the audit hook never breaks the audited tool call."""
        self._seq += 1
        data: dict[str, Any] = {
            "org": self.org,
            "scope": self.scope,
            "tool": tool,
            "args_hash": args_hash,
            "outcome": outcome,
            "elapsed_s": round(elapsed, 3),
            "seq": self._seq,
        }
        if error:
            data["error"] = error[:200]
        try:
            self.store.capture(
                "tool_audit", data, thread_id=thread_id, category="audit",
            )
            self.store.flush()
        except Exception:
            pass  # observe-only: never break the tool call over a log write

    # -- sync ------------------------------------------------------------------

    def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled:
            return handler(request)
        tool = self._tool_name(request)
        args_hash = _hash_args(self._tool_args(request))
        thread_id = self._thread_id(request)
        t0 = time.time()
        try:
            result = handler(request)
        except Exception as exc:
            self._record(
                tool=tool, args_hash=args_hash, thread_id=thread_id,
                outcome="error", elapsed=time.time() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise  # observe-only: the exception still propagates
        self._record(
            tool=tool, args_hash=args_hash, thread_id=thread_id,
            outcome=_outcome_of(result), elapsed=time.time() - t0,
        )
        return result  # UNCHANGED — never mutate the audited I/O

    # -- async (the production path — the server/runner use ainvoke) ------------

    async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled:
            return await handler(request)
        tool = self._tool_name(request)
        args_hash = _hash_args(self._tool_args(request))
        thread_id = self._thread_id(request)
        t0 = time.time()
        try:
            result = await handler(request)
        except Exception as exc:
            self._record(
                tool=tool, args_hash=args_hash, thread_id=thread_id,
                outcome="error", elapsed=time.time() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._record(
            tool=tool, args_hash=args_hash, thread_id=thread_id,
            outcome=_outcome_of(result), elapsed=time.time() - t0,
        )
        return result
