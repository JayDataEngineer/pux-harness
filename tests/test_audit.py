"""Phase 4 — AuditMiddleware behavior (observe-only tool-call audit).

Owns the AuditMiddleware CLASS: every behavior contract (records ok + error
outcomes, observe-only result passthrough, args-hashed-not-raw, dedup-defeating
seq, sync + async paths, disabled=no-op) is proven here against a REAL temp
``EventStore`` (so the actual ``capture()`` + FTS5 + ``_row_to_event`` path is
exercised — prove, don't assert). The registry WIRING (opt-in via
``middleware.supervisor.add: [audit]``, default-off, outermost mount) lives in
``tests/test_stack.py``; the contract coverage (``audit`` is an allowed scoped
override name) is auto-derived from the registry.
"""
from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from pux_harness.context.audit import AuditMiddleware
from pux_harness.context.events import EventStore


# --- shared helpers -------------------------------------------------------

def _store(tmp_path):
    """A real per-test EventStore on a temp sqlite file (the FTS5 path is
    exercised; no shared global store is touched)."""
    return EventStore(tmp_path / "audit.sqlite")


def _req(name="pux_sandbox_python", args=None, thread_id="t1"):
    """A fake ToolCallRequest: ``tool_call`` + ``state.configurable.thread_id``
    — the two surfaces AuditMiddleware reads (mirrors ContextMiddleware)."""
    return SimpleNamespace(
        tool_call={"name": name, "args": args if args is not None else {}},
        state={"configurable": {"thread_id": thread_id}},
    )


def _tm(content="ok", *, status="success", name="pux_sandbox_python"):
    return ToolMessage(content=content, name=name, tool_call_id="c1", status=status)


def _ret(value):
    """Async handler factory returning ``value`` (the async path ``await``s the
    handler, so a bare lambda won't do — Python has no async lambda)."""
    async def _h(_req):
        return value
    return _h


def _raise(exc):
    """Async handler factory that raises ``exc`` (so the reraise path is real)."""
    async def _h(_req):
        raise exc
    return _h


def _rows(store, *, thread_id="t1"):
    """All tool_audit rows for the thread (newest-first). ``recent`` is the
    public read surface; data is parsed back to a dict by ``_row_to_event``."""
    return store.recent(thread_id=thread_id, event_type="tool_audit", limit=50)


def _run(coro):
    return asyncio.run(coro)


# --- the happy path: one ok call → one row --------------------------------

def test_records_ok_call_with_full_payload(tmp_path):
    store = _store(tmp_path)
    mw = AuditMiddleware(store, org="acme", scope="supervisor")
    _run(mw.awrap_tool_call(_req(args={"code": "print(1)"}),
                            handler=_ret(_tm("done"))))

    rows = _rows(store)
    assert len(rows) == 1, rows
    d = rows[0].data
    assert d["outcome"] == "ok"
    assert d["tool"] == "pux_sandbox_python"
    assert d["org"] == "acme"
    assert d["scope"] == "supervisor"
    assert d["seq"] == 1
    assert d["elapsed_s"] >= 0.0
    # args_hash is a 16-char hex string (sha256 truncated) — present, not empty.
    assert re.fullmatch(r"[0-9a-f]{16}", d["args_hash"]), d["args_hash"]


def test_records_error_outcome_from_toolmessage_status(tmp_path):
    """A ToolMessage with ``status="error"`` → outcome ``"error"`` (the framework
    marks failed tools this way). No exception raised — the tool returned an
    error result."""
    store = _store(tmp_path)
    mw = AuditMiddleware(store)
    _run(mw.awrap_tool_call(_req(), handler=_ret(_tm("boom", status="error"))))
    rows = _rows(store)
    assert rows[0].data["outcome"] == "error"


def test_records_and_reraises_on_raised_exception(tmp_path):
    """If the handler RAISES, the audit row records outcome=error + the error
    text, AND the exception propagates unchanged (observe-only ⇒ never swallow).
    The tool call still surfaces its failure to the agent."""
    store = _store(tmp_path)
    mw = AuditMiddleware(store)

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        _run(mw.awrap_tool_call(_req(), handler=_raise(_Boom("exploded"))))
    rows = _rows(store)
    assert len(rows) == 1
    d = rows[0].data
    assert d["outcome"] == "error"
    assert "_Boom" in d["error"]
    assert "exploded" in d["error"]


# --- the core contract: observe-only (result UNCHANGED) -------------------

def test_observe_only_result_returned_unchanged(tmp_path):
    """AuditMiddleware MUST NOT mutate the tool result — it returns the handler's
    exact object (identity check). No Command, no wrapping, no enrichment."""
    store = _store(tmp_path)
    mw = AuditMiddleware(store)
    sentinel = _tm("the real result")
    out = _run(mw.awrap_tool_call(_req(), handler=_ret(sentinel)))
    assert out is sentinel  # exact object identity — never swapped/wrapped


# --- args-hashed, never raw (the privacy contract) ------------------------

def test_args_hash_stable_for_identical_args(tmp_path):
    """Two calls with identical args → identical args_hash (deterministic sha256
    over sort_keyed JSON, so insertion-order doesn't matter)."""
    store = _store(tmp_path)
    mw = AuditMiddleware(store)
    for _ in range(2):
        _run(mw.awrap_tool_call(_req(args={"b": 1, "a": 2}),
                                handler=_ret(_tm())))
    rows = _rows(store)
    assert rows[0].data["args_hash"] == rows[1].data["args_hash"]


def test_raw_args_payload_never_persisted(tmp_path):
    """The audit row stores ONLY the args HASH — the raw payload (which may carry
    secrets: cookies, tokens, file contents) is never written. A canary secret
    in the args must not appear anywhere in the stored row."""
    store = _store(tmp_path)
    mw = AuditMiddleware(store)
    _run(mw.awrap_tool_call(_req(args={"token": "TOPSECRET-DO-NOT-LOG"}),
                            handler=_ret(_tm())))
    rows = _rows(store)
    assert len(rows) == 1
    blob = json.dumps(rows[0].data, default=str)
    assert "TOPSECRET-DO-NOT-LOG" not in blob, blob


# --- the completeness contract: seq defeats capture's dedup ---------------

def test_seq_defeats_capture_dedup_repeated_calls_all_logged(tmp_path):
    """``EventStore.capture`` dedups by type+data_hash within a window — great
    for the activity feed, wrong for an audit log (every call must appear). Each
    row carries a monotonic ``seq``, so the data hash differs every call and
    dedup never collapses two REAL calls. Two identical tool calls → two rows."""
    store = _store(tmp_path)
    mw = AuditMiddleware(store)
    for _ in range(3):
        _run(mw.awrap_tool_call(_req(args={"x": 1}), handler=_ret(_tm())))
    rows = _rows(store)
    assert len(rows) == 3, rows  # not collapsed to 1
    seqs = sorted(r.data["seq"] for r in rows)
    assert seqs == [1, 2, 3], seqs  # monotonic, gapless


# --- the sync path + the master switch ------------------------------------

def test_sync_wrap_tool_call_also_records(tmp_path):
    """The sync ``wrap_tool_call`` (used by synchronous runners) records a row
    too — not just the async path. Uses a SYNC handler (the sync path does not
    ``await`` its handler)."""
    store = _store(tmp_path)
    mw = AuditMiddleware(store)
    mw.wrap_tool_call(_req(args={"q": "hi"}), handler=lambda r: _tm())
    assert len(_rows(store)) == 1


def test_disabled_is_noop_and_still_runs_handler(tmp_path):
    """``enabled=False`` ⇒ no row written, but the handler STILL runs and its
    result is returned (the master switch never blocks the audited tool)."""
    store = _store(tmp_path)
    mw = AuditMiddleware(store, enabled=False)
    called = {"n": 0}

    async def _h(_req):
        called["n"] += 1
        return _tm("ran")

    out = _run(mw.awrap_tool_call(_req(), handler=_h))
    assert called["n"] == 1
    assert out.content == "ran"
    assert _rows(store) == []
