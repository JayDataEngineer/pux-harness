"""Agent-callable retrieval + meta surface over the unified context store.

All tools bind to the same ``EventStore`` the middleware writes to:

RETRIEVAL
* ``ctx_search`` — BM25 search across the UNION of offloaded blobs (full tool
  results parked behind ``ctx:<id>``) + structured events (tool-call previews,
  errors, decisions). Returns matching handles + a snippet each, tagged
  ``[blob]`` (recoverable in full via ``ctx_recall``) or ``[event]``.
* ``ctx_recall`` — pull the FULL content of a stashed blob back by its handle.

INDEXING
* ``ctx_index`` — stash arbitrary inline content as a blob behind a ``ctx:<id>``
  handle so it's FTS5/BM25-searchable later. The complement to the
  middleware's AUTO offload (which only fires on oversized tool results):
  ``ctx_index`` is the agent's MANUAL lever to park reference text (a doc, a
  paste, an API response it already has in-context) without waiting for the
  offload threshold.

META
* ``ctx_stats`` — counts (events / blobs), per-type breakdown, db size, FTS5
  status. The "how much have I saved?" surface.
* ``ctx_doctor`` — one-line health checks: store reachable? FTS5 compiled in?
  db writable? Same payload shape context-mode's doctor returns.
* ``ctx_purge`` — delete all events + blobs (or a single thread's). Destructive;
  the agent-facing surface mirrors context-mode's ctx_purge (a fresh slate).

Only the slice the agent asks for re-enters its context — fewer tokens
recalled = lower cost per call. The retrieval pair is EXEMPT from offload (the
middleware's ``_RETRIEVAL_TOOLS``): their job is to inject content, so
re-stashing their output would trap the agent the instant it retrieves a large
stash. The meta trio is exempt too — their output is small, never worth offload.

This replaces the old ``event_recent`` / ``event_query`` pair (the
resume snapshot already gives chronological orientation; ``ctx_search`` covers
query-based recall over both blobs and events in one tool).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from pux_harness.context.events import EventStore, SearchHit, shared_event_store
from pux_harness.context.exec_tools import build_exec_tools


class _RecallArgs(BaseModel):
    handle: str = Field(
        ...,
        description='A ctx handle like "ctx:1a2b3c4d5e6f", shown at the top of a truncated tool result.',
    )


class _SearchArgs(BaseModel):
    query: str = Field(
        ...,
        description=(
            "A distinctive phrase to find across prior tool outputs and events. "
            "Ranked by relevance; returns handles + a snippet each."
        ),
    )
    limit: int = Field(8, description="Max hits to return (default 8).")


class _IndexArgs(BaseModel):
    content: str = Field(
        ...,
        description=(
            "The text to index verbatim. After this call it is searchable via "
            "ctx_search and recallable in full via ctx_recall using the returned handle."
        ),
    )
    source: str = Field(
        "",
        description=(
            "Optional label for the indexed content (e.g. 'react-useEffect-docs', "
            "'api-response-2024-03'). Surfaces in ctx_search hits to aid recall."
        ),
    )


class _PurgeArgs(BaseModel):
    thread_id: str = Field(
        "",
        description=(
            "Only purge events + blobs for this thread. Empty (default) = purge "
            "everything. Destructive; cannot be undone."
        ),
    )


def _format_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return "no prior tool output or event matched that query."
    lines = [f"{len(hits)} hit(s):"]
    for h in hits:
        tag = f"[{h.kind}]"  # [blob] (recallable in full) or [event]
        tool = f" {h.tool}" if h.tool else ""
        handle = f" {h.handle}" if h.handle else ""
        evtype = f" ({h.type})" if h.type else ""
        lines.append(f"- {tag}{tool}{evtype}{handle}: {h.snippet}")
    return "\n".join(lines)


def build_context_tools(
    store: EventStore | None = None,
    *,
    exec_client: object | None = None,
) -> list[StructuredTool]:
    """The retrieval + indexing + meta tools, all bound to ``store`` (default:
    the process-wide shared event store). Built fresh per call so a test can
    pass its own ``EventStore(tmp_path)`` and have offload + recall + purge all
    share it.

    When ``exec_client`` is provided (a ``DockerExecClient``), the 4
    exec-dependent tools (ctx_execute, ctx_execute_file, ctx_batch_execute,
    ctx_fetch_and_index) are appended. When None (tests, offline use), they're
    omitted — no Docker dependency.

    Order matters: retrieval first (the high-value surface), indexing next,
    meta last, exec last — so the agent's tool-list presentation foregrounds recall."""
    s = store or shared_event_store()

    def _recall(handle: str) -> str:
        out = s.recall_blob(handle)
        return out if out is not None else f"no truncated result found for handle {handle!r}"

    def _search(query: str, limit: int = 8) -> str:
        return _format_hits(s.search_context(query, limit=limit))

    def _index(content: str, source: str = "") -> str:
        if not content:
            return "no content provided; nothing indexed."
        tool_tag = f"ctx_index:{source}" if source else "ctx_index"
        stash = s.stash_blob(content, tool=tool_tag)
        s.flush()
        return (
            f"Indexed {stash.chars} chars under handle {stash.handle}. "
            f"Find it later with ctx_search(<phrase>); recover in full with "
            f"ctx_recall({stash.handle!r})."
        )

    def _stats() -> str:
        return json.dumps(s.stats(), ensure_ascii=False, default=str)

    def _doctor() -> str:
        checks = _run_doctor(s)
        lines = [f"[{'OK' if ok else 'FAIL'}] {name}: {detail}" for name, ok, detail in checks]
        return "\n".join(lines)

    def _purge(thread_id: str = "") -> str:
        out = s.purge(thread_id=thread_id)
        scope = f"thread {thread_id!r}" if thread_id else "all threads"
        return (
            f"Purged {out['events_deleted']} events + {out['blobs_deleted']} blobs "
            f"({scope}). session_resume snapshots left intact."
        )

    recall = StructuredTool.from_function(
        _recall,
        name="ctx_recall",
        description=(
            "Return the complete output of a tool call whose result was truncated. "
            "Pass the ctx: handle shown at the top of the truncated result, "
            'e.g. "ctx:1a2b3c4d5e6f".'
        ),
        args_schema=_RecallArgs,
    )
    search = StructuredTool.from_function(
        _search,
        name="ctx_search",
        description=(
            "Search prior tool outputs and events for a phrase; returns matching "
            "handles + a snippet each. Use when you remember a detail but not which "
            "call produced it, then ctx_recall the handle for the full text."
        ),
        args_schema=_SearchArgs,
    )
    index = StructuredTool.from_function(
        _index,
        name="ctx_index",
        description=(
            "Index arbitrary text into the persistent knowledge base so it is "
            "searchable (ctx_search) and recallable in full (ctx_recall). Use to "
            "park reference docs, paste, or an API response you already hold in "
            "context — keeps it queryable without re-fetching."
        ),
        args_schema=_IndexArgs,
    )
    stats = StructuredTool.from_function(
        _stats,
        name="ctx_stats",
        description=(
            "Report context-store stats: event + blob counts, per-type breakdown, "
            "threads seen, db size, FTS5 availability. Use to gauge how much prior "
            "tool output is parked behind handles vs still in your working window."
        ),
    )
    doctor = StructuredTool.from_function(
        _doctor,
        name="ctx_doctor",
        description=(
            "Run health checks on the context store: reachable? FTS5 compiled in? "
            "db writable? Returns one line per check, [OK] or [FAIL]."
        ),
    )
    purge = StructuredTool.from_function(
        _purge,
        name="ctx_purge",
        description=(
            "Permanently delete events + blobs from the context store (all threads, "
            "or one thread via thread_id). Destructive; cannot be undone. "
            "Compaction snapshots (session_resume) are NOT touched."
        ),
        args_schema=_PurgeArgs,
    )
    return [recall, search, index, stats, doctor, purge] + (
        build_exec_tools(s, exec_client) if exec_client is not None else []
    )


# -- ctx_doctor implementation ------------------------------------------------


def _run_doctor(store: EventStore) -> list[tuple[str, bool, str]]:
    """Run the same checks context-mode's ctx_doctor runs, adapted to the
    harness's SQLite-backed store. Each tuple is ``(check_name, ok, detail)``.

    Pure + synchronous — no network, no docker. Safe for the agent to call any
    time; the output is the entire tool result (small, never offloaded)."""
    checks: list[tuple[str, bool, str]] = []

    # 1. Store path exists + is a file.
    path = Path(store.db_path)
    exists = path.is_file()
    checks.append((
        "store_path",
        exists,
        f"{path} {'exists' if exists else 'missing (will be created on first write)'}",
    ))

    # 2. DB is openable + writable (round-trip a no-op pragma).
    writable = False
    detail = "unknown"
    try:
        conn = store._get_conn()  # noqa: SLF001 — intentional, this is the doctor
        cur = conn.execute("PRAGMA quick_check")
        row = cur.fetchone()
        ok_text = row[0] if row else ""
        writable = ok_text == "ok"
        detail = f"quick_check={ok_text}"
    except sqlite3.Error as e:
        detail = f"sqlite error: {e}"
    checks.append(("db_writable", writable, detail))

    # 3. FTS5 compiled in (the store degrades to LIKE without it — surfaces as a
    # hard miss so an operator notices the weaker search path is live).
    stats = store.stats()
    fts5 = bool(stats["fts5"])
    checks.append((
        "fts5",
        fts5,
        "available" if fts5 else "NOT available — search falls back to LIKE",
    ))

    # 4. Schema tables present (catches a half-init / corrupt DB).
    try:
        conn = store._get_conn()  # noqa: SLF001
        tbls = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        needed = {"events", "blobs", "session_resume"}
        have = needed.issubset(tbls)
        missing = needed - tbls
        detail = "all present" if have else f"missing: {sorted(missing)}"
        checks.append(("schema", have, detail))
    except sqlite3.Error as e:
        checks.append(("schema", False, f"sqlite error: {e}"))

    return checks
