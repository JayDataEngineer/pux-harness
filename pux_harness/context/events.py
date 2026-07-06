"""Unified context store — structured events + offloaded blobs in one SQLite DB.

Two kinds of long-lived, queryable context live here, behind a single FTS5
search surface:

1. **Events** — discrete structured records (tool calls, file edits,
   git ops, decisions, errors, blockers) captured by the middleware layer.
   Priority tiers (P1 critical → P4 low) drive budget enforcement in the
   snapshot builder; P1 always included, P4 dropped first under the ≤2KB budget.
2. **Blobs** — the FULL verbatim text of oversized
   tool results that ``ContextMiddleware`` parked behind a ``ctx:<id>`` handle
   so they don't crowd the working context. The agent pulls a blob back on
   demand via ``ctx_recall`` (by handle) or finds it via ``ctx_search`` (BM25
   over events + blobs). Blobs are NEVER deduped — two big results are genuinely
   distinct — and have no priority budget (recall is on-demand).

This is the *proactive* complement to deepagents' reactive
``SummarizationMiddleware`` (which only offloads once the window has already
overflowed): we keep large tool results out of the prompt before they accumulate.
Modeled on ``mksglu/context-mode``'s single SessionDB. Consolidating the old
plain-file ``CtxStore`` into here means ONE queryable store, ONE
cleanup target (``.pux/events.sqlite``), ONE search surface — instead of two
parallel systems (one weak substring grep, one strong FTS5).

FTS5 powers BM25-ranked retrieval across both events + blobs so the agent pulls
back only the relevant slice — fewer tokens recalled = lower cost per call.

Layering note: ``context/`` is LOWER-level than ``agent/`` (``agent/`` depends
on ``context/``, never the reverse). ``PROJECT_ROOT`` used to be computed
locally from ``__file__`` to avoid importing ``agent.orgs`` (a cycle, since
``agent.orgs`` imports the context layer for subagents). It is now sourced from
the kit's location-independent resolver (:func:`pux_harness.kit._paths.project_root`)
— the kit is the slim, import-pinned core (kit → ``context`` is forbidden and
tripwired), so this lower-level layer can depend on it with no cycle, AND the
root no longer shatters when ``pux_harness`` is installed outside the
orchestrator repo.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pux_harness.kit._paths import project_root

# The app root (where ``.pux/`` lives) is injected, not derived from the
# install path — see the kit's ``_paths.project_root`` for the resolution rule.
PROJECT_ROOT = project_root()

EVENTS_DB = PROJECT_ROOT / ".pux" / "events.sqlite"

# Path-safe blob handle: ``ctx:<hex>``. Hex-only, fixed-ish length range — no
# path separators, no dots — so a handle can never escape the blob lookup.
_HANDLE_RE = re.compile(r"^ctx:(?P<id>[0-9a-f]+)$")


@dataclass(frozen=True)
class StashResult:
    """What ``stash_blob`` hands back to the middleware + agent."""

    handle: str  # "ctx:<id>"
    id: str  # bare id
    chars: int


@dataclass(frozen=True)
class SearchHit:
    """One row from the unified events+blobs search.

    ``kind`` is ``"blob"`` (recoverable in full via ``ctx_recall(handle)``) or
    ``"event"`` (small — the snippet IS the useful content; no handle)."""

    kind: str
    tool: str
    snippet: str
    handle: str = ""  # set for blobs only
    type: str = ""  # event type, set for events only


@dataclass(frozen=True)
class Event:
    """A single captured event."""

    id: int = 0
    ts: float = 0.0
    type: str = ""
    priority: int = 3  # P1-P4, default P3
    thread_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    # v2 additions (backward-compatible — old callers never set these)
    category: str = ""
    data_hash: str = ""
    created_at: str = ""


# Priority constants -----------------------------------------------------------

P1 = 1  # Critical state — always preserved
P2 = 2  # Working state — preserved unless budget tight
P3 = 3  # Context — dropped first under budget pressure
P4 = 4  # Low — analytics/debug

# Event type → default priority mapping (original API, preserved)
EVENT_PRIORITIES: dict[str, int] = {
    # P1 — critical
    "task_started": P1,
    "task_completed": P1,
    "task_failed": P1,
    "decision_made": P1,
    "error": P1,
    "blocker": P1,
    # P2 — working state
    "file_modified": P2,
    "git_operation": P2,
    "tool_call": P2,
    # P3 — context
    "user_correction": P3,
    "approach_rejected": P3,
    "env_change": P3,
    # P4 — low
    "session_start": P4,
    "session_end": P4,
    "compaction": P4,
}

# Category grouping (v2, for snapshot builder)
EVENT_CATEGORIES: dict[str, str] = {
    "task_started": "task",
    "task_completed": "task",
    "task_failed": "task",
    "file_modified": "file",
    "git_operation": "git",
    "tool_call": "data",
    "error": "error",
    "blocker": "error",
    "decision_made": "decision",
    "user_correction": "decision",
    "approach_rejected": "decision",
    "env_change": "env",
    "session_start": "data",
    "session_end": "data",
    "compaction": "data",
}

# Dedup + eviction constants
MAX_EVENTS_PER_SESSION = 1000
DEDUP_WINDOW = 5


def _data_hash(data: str) -> str:
    """SHA-256 dedup hash (first 16 hex chars)."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16].upper()


class EventStore:
    """Append-only SQLite store for structured agent events.

    Original API preserved for backward compatibility with event_middleware
    and event_tools.  v2 additions (category, data_hash, session_resume,
    dedup, FIFO eviction) are additive — old callers unaffected.

    Indexes:
    - ``idx_events_thread`` — resume a session (thread_id + ts range)
    - ``idx_events_type`` — filter by event type

    FTS5 virtual table ``events_fts`` enables BM25-ranked keyword search
    across event data.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Per-thread connections: LangGraph runs agent nodes (and therefore the
        # context middleware's wrap hooks) in a threadpool, so the store is
        # touched from multiple threads within one ``.invoke``/.ainvoke``. A
        # sqlite connection is thread-bound by default (check_same_thread=True);
        # rather than disable that check and add a global lock, each thread gets
        # its OWN connection to the same WAL-mode DB. WAL lets a reader on one
        # connection see a writer's committed row on another (``flush()``
        # commits), and serializes concurrent writers via ``busy_timeout``.
        # Surfaced by ``test_context_subagent`` driving the real
        # ``create_agent.invoke`` — without this, offload-by-thread-A
        # + recall-by-thread-B raised ``ProgrammingError``.
        self._tls = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=3000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._init_schema(conn)
            self._tls.conn = conn
        return conn

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                ts REAL NOT NULL,
                type TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 3,
                thread_id TEXT NOT NULL DEFAULT '',
                data TEXT NOT NULL DEFAULT '{}',
                category TEXT NOT NULL DEFAULT '',
                data_hash TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_events_thread
                ON events(thread_id, ts);
            CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(type, ts);

            CREATE TABLE IF NOT EXISTS session_resume (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                snapshot TEXT NOT NULL,
                event_count INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                consumed INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS blobs (
                id TEXT PRIMARY KEY,
                ts REAL NOT NULL,
                tool TEXT NOT NULL DEFAULT '',
                thread_id TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                chars INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_blobs_thread
                ON blobs(thread_id, ts);
            """
        )
        # FTS5 for ranked search — created separately so a missing FTS5
        # build doesn't block the main table.
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
                    type, data, thread_id,
                    content=events, content_rowid=id
                )
                """
            )
            # External-content sync triggers: keep events_fts consistent with
            # events on every INSERT / DELETE / UPDATE. events_fts is an
            # external-content table (content=events), so without the AFTER
            # DELETE trigger, FIFO eviction (which DELETEs from events only)
            # would orphan the matching FTS row — leaving dead entries that
            # skew BM25 corpus stats. This is the canonical sync pattern from
            # the SQLite FTS5 docs; capture() no longer hand-syncs the index.
            # (Events rows are never UPDATEd today, but the AFTER UPDATE
            # trigger is included for correctness if that ever changes.)
            conn.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS events_fts_ai AFTER INSERT ON events BEGIN
                    INSERT INTO events_fts(rowid, type, data, thread_id)
                    VALUES (new.id, new.type, new.data, new.thread_id);
                END;
                CREATE TRIGGER IF NOT EXISTS events_fts_ad AFTER DELETE ON events BEGIN
                    INSERT INTO events_fts(events_fts, rowid, type, data, thread_id)
                    VALUES ('delete', old.id, old.type, old.data, old.thread_id);
                END;
                CREATE TRIGGER IF NOT EXISTS events_fts_au AFTER UPDATE ON events BEGIN
                    INSERT INTO events_fts(events_fts, rowid, type, data, thread_id)
                    VALUES ('delete', old.id, old.type, old.data, old.thread_id);
                    INSERT INTO events_fts(rowid, type, data, thread_id)
                    VALUES (new.id, new.type, new.data, new.thread_id);
                END;
                """
            )
            # Blobs get their OWN standalone fts5 table (separate corpus, so a
            # blob search isn't drowned out by event volume). It is NOT an
            # external-content table: blobs.id is TEXT (the hex handle), but
            # fts5 content_rowid must be INTEGER — so we store content + tool
            # here directly, carrying hex_id + thread_id as UNINDEXED columns
            # for projection + filtering.
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS blobs_fts USING fts5(
                    content, tool, hex_id UNINDEXED, thread_id UNINDEXED
                )
                """
            )
            # One-time resync, gated by PRAGMA user_version (persisted in the
            # DB file, so this runs exactly once across all connections): an
            # existing DB upgraded in place may carry orphaned events_fts rows
            # from pre-trigger evictions. 'rebuild' repopulates the index from
            # the events content table, dropping any orphans. Committed at
            # once so other thread-local connections don't block on the write
            # lock during this migration.
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version < 1:
                conn.execute("INSERT INTO events_fts(events_fts) VALUES('rebuild')")
                conn.execute("PRAGMA user_version = 1")
                conn.commit()
        except sqlite3.OperationalError:
            pass  # FTS5 not compiled in — degrade to LIKE
        # Migrate: add category + data_hash columns to existing tables.
        try:
            cols = {r["name"] for r in conn.execute("PRAGMA table_xinfo(events)").fetchall()}
            if "category" not in cols:
                conn.execute("ALTER TABLE events ADD COLUMN category TEXT NOT NULL DEFAULT ''")
            if "data_hash" not in cols:
                conn.execute("ALTER TABLE events ADD COLUMN data_hash TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass

    # -- write -----------------------------------------------------------------

    def capture(
        self,
        event_type: str,
        data: Any = None,
        *,
        priority: int | None = None,
        thread_id: str = "",
        category: str = "",
    ) -> int:
        """Capture an event. Returns the event id.

        ``data`` accepts a dict (legacy) or a string (v2).  Dicts are
        JSON-serialized.  If *priority* is ``None`` it is looked up from
        ``EVENT_PRIORITIES`` (defaulting to P3 for unknown types).

        v2: dedup skips if same type+data_hash in last DEDUP_WINDOW events.
        v2: FIFO eviction of lowest-priority event when over MAX_EVENTS.
        """
        if priority is None:
            priority = EVENT_PRIORITIES.get(event_type, P3)
        if category == "":
            category = EVENT_CATEGORIES.get(event_type, "data")

        # Normalize data to string for storage + hashing.
        if isinstance(data, dict):
            data_str = json.dumps(data, ensure_ascii=False, default=str)
        elif isinstance(data, str):
            data_str = data
        else:
            data_str = json.dumps(data, ensure_ascii=False, default=str) if data is not None else ""

        dhash = _data_hash(data_str)
        now = time.time()
        conn = self._get_conn()

        # v2 dedup: skip if same type+hash in last N events for this thread.
        if thread_id:
            dup = conn.execute(
                "SELECT 1 FROM ("
                "  SELECT type, data_hash FROM events"
                "  WHERE thread_id = ? ORDER BY id DESC LIMIT ?"
                ") AS recent WHERE recent.type = ? AND recent.data_hash = ? LIMIT 1",
                (thread_id, DEDUP_WINDOW, event_type, dhash),
            ).fetchone()
            if dup:
                return 0

        # v2 FIFO eviction of lowest-priority (then oldest) event.
        if thread_id:
            count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM events WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if count_row and count_row["cnt"] >= MAX_EVENTS_PER_SESSION:
                conn.execute(
                    "DELETE FROM events WHERE id = ("
                    "  SELECT id FROM events WHERE thread_id = ?"
                    "  ORDER BY priority ASC, id ASC LIMIT 1"
                    ")",
                    (thread_id,),
                )

        cur = conn.execute(
            "INSERT INTO events (ts, type, priority, thread_id, data, category, data_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now, event_type, priority, thread_id, data_str, category, dhash),
        )
        rowid = cur.lastrowid
        # FTS index is kept in sync by the events_fts_ai trigger (see
        # _init_schema) — no manual INSERT here.
        return rowid  # type: ignore[return-value]

    def flush(self) -> None:
        """Commit any pending writes on the calling thread's connection.

        Thread-local: the worker thread that wrote the rows is the one whose
        connection holds the uncommitted transaction, so we commit *that*
        connection — not some other thread's. Readers on other connections
        only see the rows once committed (WAL lets them read past the commit
        without reopening)."""
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.commit()

    # -- read ------------------------------------------------------------------

    def recent(
        self,
        *,
        thread_id: str = "",
        event_type: str = "",
        limit: int = 20,
        min_priority: int = P4,
    ) -> list[Event]:
        """Return the most recent events, newest first.

        Filters:
        - *thread_id*: only events for this thread (empty = all threads)
        - *event_type*: only events of this type (empty = all types)
        - *min_priority*: only events with priority ≤ this value (lower = more critical)
        """
        conn = self._get_conn()
        clauses: list[str] = ["priority <= ?"]
        params: list[int | str] = [min_priority]
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if event_type:
            clauses.append("type = ?")
            params.append(event_type)
        where = " AND ".join(clauses)
        params.append(limit)
        rows = conn.execute(
            f"SELECT id, ts, type, priority, thread_id, data, category, data_hash "
            f"FROM events WHERE {where} ORDER BY ts DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def query(
        self,
        search: str,
        *,
        thread_id: str = "",
        limit: int = 10,
    ) -> list[Event]:
        """BM25-ranked search across EVENT data only (events table + events_fts).

        For the unified events+blobs search used by the ``ctx_search`` tool, see
        ``search_context``. Each word is matched as a prefix (``auth*`` matches
        ``authentication``) so partial stems still hit. Falls back to LIKE if
        FTS5 is unavailable.
        """
        conn = self._get_conn()
        # Build FTS5 query: each word gets a trailing * for prefix matching.
        fts_query = " ".join(w.strip() for w in search.split() if w.strip())
        fts_query = " ".join(f"{w}*" for w in fts_query.split() if w)
        if not fts_query:
            return []

        # Try FTS5 first.
        try:
            if thread_id:
                rows = conn.execute(
                    "SELECT e.id, e.ts, e.type, e.priority, e.thread_id, e.data, "
                    "       e.category, e.data_hash "
                    "FROM events_fts f "
                    "JOIN events e ON e.id = f.rowid "
                    "WHERE events_fts MATCH ? AND e.thread_id = ? "
                    "ORDER BY rank LIMIT ?",
                    [fts_query, thread_id, limit],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT e.id, e.ts, e.type, e.priority, e.thread_id, e.data, "
                    "       e.category, e.data_hash "
                    "FROM events_fts f "
                    "JOIN events e ON e.id = f.rowid "
                    "WHERE events_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    [fts_query, limit],
                ).fetchall()
            return [self._row_to_event(r) for r in rows]
        except sqlite3.OperationalError:
            pass  # FTS5 not available — fall through to LIKE

        # LIKE fallback.
        like = f"%{search}%"
        clauses: list[str] = ["data LIKE ?"]
        params: list[int | str] = [like]
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        where = " AND ".join(clauses)
        params.append(limit)
        rows = conn.execute(
            f"SELECT id, ts, type, priority, thread_id, data, category, data_hash "
            f"FROM events WHERE {where} ORDER BY ts DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    # -- blobs (offloaded tool output) ----------------------------------------

    def stash_blob(
        self, content: str, *, tool: str = "", thread_id: str = "",
    ) -> StashResult:
        """Park the FULL text of an oversized tool result; return its
        ``ctx:<id>`` handle. Ids are uuid4 hex (unique per call) so two oversized
        results from the same tool never collide. Blobs are NEVER deduped — two
        big results are genuinely distinct (unlike events, where a repeated
        identical tool_call is noise)."""
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO blobs (id, ts, tool, thread_id, content, chars) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, now, tool, thread_id, content, len(content)),
        )
        try:
            conn.execute(
                "INSERT INTO blobs_fts(content, tool, hex_id, thread_id) "
                "VALUES (?, ?, ?, ?)",
                (content, tool, sid, thread_id),
            )
        except sqlite3.OperationalError:
            pass
        return StashResult(handle=f"ctx:{sid}", id=sid, chars=len(content))

    def recall_blob(self, handle: str) -> str | None:
        """Return the full blob content for ``ctx:<id>`` (or a bare id), else
        None. Missing/garbage handles → None (not an error): a bad handle is a
        normal agent mistake, surfaced as 'not found' to the model. ``_strip_handle``
        rejects anything that isn't a hex id so the lookup can't be escaped."""
        sid = _strip_handle(handle)
        if sid is None:
            return None
        row = self._get_conn().execute(
            "SELECT content FROM blobs WHERE id = ?", (sid,),
        ).fetchone()
        return row["content"] if row else None

    def search_context(
        self, query: str, *, thread_id: str = "", limit: int = 8,
    ) -> list[SearchHit]:
        """Unified BM25 search across blobs + events — the one retrieval surface
        the ``ctx_search`` tool calls.

        Blobs are surfaced first (they're the recoverable big results the agent
        stashed — the primary recall surface; ``ctx_recall(handle)`` pulls the
        full bytes), then events (small structured records; the snippet IS the
        content). BM25 ``rank`` is not comparable across the two FTS5 corpora
        (different token stats), so each table is ranked independently and the
        results merged blobs-first. Empty query returns nothing.

        Thread filter applies to both. Falls back to LIKE if FTS5 unavailable.
        """
        query = query.strip()
        if not query:
            return []
        # Each word gets a trailing * for prefix matching (auth -> authentication).
        fts_query = " ".join(f"{w}*" for w in query.split() if w)
        if not fts_query:
            return []
        conn = self._get_conn()
        hits: list[SearchHit] = []

        # --- blobs (recoverable in full via ctx_recall) ---------------------
        try:
            if thread_id:
                blob_rows = conn.execute(
                    "SELECT hex_id, tool, content FROM blobs_fts "
                    "WHERE blobs_fts MATCH ? AND thread_id = ? "
                    "ORDER BY rank LIMIT ?",
                    [fts_query, thread_id, limit],
                ).fetchall()
            else:
                blob_rows = conn.execute(
                    "SELECT hex_id, tool, content FROM blobs_fts "
                    "WHERE blobs_fts MATCH ? ORDER BY rank LIMIT ?",
                    [fts_query, limit],
                ).fetchall()
            for r in blob_rows:
                hits.append(SearchHit(
                    kind="blob",
                    tool=r["tool"],
                    handle=f"ctx:{r['hex_id']}",
                    snippet=_snippet(r["content"], query, window=240),
                ))
        except sqlite3.OperationalError:
            # FTS5 unavailable — LIKE fallback for blobs (over the blobs table).
            like = f"%{query}%"
            if thread_id:
                blob_rows = conn.execute(
                    "SELECT id, tool, content FROM blobs "
                    "WHERE content LIKE ? AND thread_id = ? ORDER BY ts DESC LIMIT ?",
                    [like, thread_id, limit],
                ).fetchall()
            else:
                blob_rows = conn.execute(
                    "SELECT id, tool, content FROM blobs "
                    "WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
                    [like, limit],
                ).fetchall()
            for r in blob_rows:
                hits.append(SearchHit(
                    kind="blob", tool=r["tool"], handle=f"ctx:{r['id']}",
                    snippet=_snippet(r["content"], query, window=240),
                ))

        # --- events (small structured records) ------------------------------
        if len(hits) < limit:
            remaining = limit - len(hits)
            try:
                if thread_id:
                    ev_rows = conn.execute(
                        "SELECT e.type, e.data FROM events_fts f "
                        "JOIN events e ON e.id = f.rowid "
                        "WHERE events_fts MATCH ? AND e.thread_id = ? "
                        "ORDER BY rank LIMIT ?",
                        [fts_query, thread_id, remaining],
                    ).fetchall()
                else:
                    ev_rows = conn.execute(
                        "SELECT e.type, e.data FROM events_fts f "
                        "JOIN events e ON e.id = f.rowid "
                        "WHERE events_fts MATCH ? ORDER BY rank LIMIT ?",
                        [fts_query, remaining],
                    ).fetchall()
            except sqlite3.OperationalError:
                like = f"%{query}%"
                if thread_id:
                    ev_rows = conn.execute(
                        "SELECT type, data FROM events "
                        "WHERE data LIKE ? AND thread_id = ? ORDER BY ts DESC LIMIT ?",
                        [like, thread_id, remaining],
                    ).fetchall()
                else:
                    ev_rows = conn.execute(
                        "SELECT type, data FROM events "
                        "WHERE data LIKE ? ORDER BY ts DESC LIMIT ?",
                        [like, remaining],
                    ).fetchall()
            for r in ev_rows:
                hits.append(SearchHit(
                    kind="event", tool=_event_tool(r["data"]),
                    type=r["type"], snippet=_snippet(r["data"], query, window=240),
                ))
        return hits[:limit]

    def count(self, *, thread_id: str = "") -> int:
        """Total event count, optionally filtered by thread."""
        conn = self._get_conn()
        if thread_id:
            row = conn.execute(
                "SELECT COUNT(*) FROM events WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the calling thread's connection (thread-local).

        Other threads' connections are out of reach from here by design — each
        worker owns its own. ``flush()`` is the operation that actually matters
        for durability (it commits); ``close()`` is a teardown nicety for the
        main thread."""
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            self._tls.conn = None

    # -- v2: resume snapshot ---------------------------------------------------

    def upsert_resume(self, session_id: str, snapshot: str, event_count: int) -> None:
        """Store a compaction snapshot (replaces any existing for this session)."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO session_resume (session_id, snapshot, event_count) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "  snapshot = excluded.snapshot, "
            "  event_count = excluded.event_count, "
            "  created_at = datetime('now'), "
            "  consumed = 0",
            (session_id, snapshot, event_count),
        )
        conn.commit()

    def claim_latest_unconsumed_resume(
        self, exclude_session: str = ""
    ) -> dict[str, Any] | None:
        """Atomically pick the newest unconsumed snapshot and mark it consumed."""
        conn = self._get_conn()
        row = conn.execute(
            "UPDATE session_resume "
            "SET consumed = 1 "
            "WHERE id = ("
            "  SELECT id FROM session_resume "
            "  WHERE consumed = 0 AND session_id != ? "
            "  ORDER BY created_at DESC, id DESC LIMIT 1"
            ") "
            "RETURNING session_id, snapshot, event_count, consumed",
            (exclude_session,),
        ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row["session_id"],
            "snapshot": row["snapshot"],
            "event_count": row["event_count"],
        }

    def get_resume(self, session_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT snapshot, event_count, consumed "
            "FROM session_resume WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "snapshot": row["snapshot"],
            "event_count": row["event_count"],
            "consumed": bool(row["consumed"]),
        }

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        # Parse data: try JSON first (dict legacy), fall back to raw string.
        raw_data = row["data"]
        try:
            parsed_data = json.loads(raw_data)
        except (json.JSONDecodeError, TypeError):
            parsed_data = {"raw": raw_data}

        return Event(
            id=row["id"],
            ts=row["ts"],
            type=row["type"],
            priority=row["priority"],
            thread_id=row["thread_id"],
            data=parsed_data if isinstance(parsed_data, dict) else {"raw": parsed_data},
            category=row["category"] if "category" in row.keys() else "",
            data_hash=row["data_hash"] if "data_hash" in row.keys() else "",
        )


# -- module helpers -----------------------------------------------------------


def _strip_handle(handle: str) -> str | None:
    """Accept ``ctx:<id>`` or a bare ``<id>``; reject anything that isn't a hex
    id so the blob lookup can't be escaped (``..``/``/`` etc.)."""
    if not handle:
        return None
    m = _HANDLE_RE.match(handle.strip())
    sid = m.group("id") if m else handle.strip()
    if not re.fullmatch(r"[0-9a-f]{6,32}", sid):
        return None
    return sid


def _snippet(text: str, query: str, *, window: int = 240) -> str:
    """A ``window``-char snippet around the first case-insensitive match of any
    query word, ellipsized at the edges. For events (``data`` is JSON), the raw
    string is searched — the match usually lands in a stored value."""
    if not text:
        return ""
    lower = text.lower()
    needle = next((w for w in query.split() if w.lower() in lower), None)
    idx = lower.find(needle.lower()) if needle else -1
    if idx < 0:
        # No word matched directly (FTS5 stemmed/prefixed past the literal) —
        # just return the head.
        return ("…" + text[:window]) if len(text) > window else text
    start = max(0, idx - window // 2)
    end = min(len(text), idx + len(needle) + window // 2)
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _event_tool(data_str: str) -> str:
    """Best-effort pull of the ``tool`` field from an event's JSON data blob."""
    try:
        parsed = json.loads(data_str)
        if isinstance(parsed, dict):
            t = parsed.get("tool")
            if isinstance(t, str):
                return t
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


# -- process-wide singleton ---------------------------------------------------

_store: EventStore | None = None


def shared_event_store() -> EventStore:
    """One process-wide event store at ``<project>/.pux/events.sqlite``."""
    global _store
    if _store is None:
        _store = EventStore(EVENTS_DB)
    return _store
