"""Unified thread store — the ONE owner of the persistent checkpointer.

Every server-side runtime (``serve`` / ``acp`` / ``direct``) builds its per-org
graph with the checkpointer from :func:`open_thread_store`, so a thread created
by ``pux direct`` is visible to ``pux show`` / ``pux resume`` (the server), and
an ACP session's checkpoints survive a process restart. All share the SAME
``.pux/agent-protocol.sqlite`` + the ``pux_threads(thread_id, org, metadata,
created_at)`` index that maps a thread_id → the org whose graph owns it.

**Critical detail (verified from
``langgraph/checkpoint/sqlite/aio.py``):** ``AsyncSqliteSaver.from_conn_string``
opens its OWN aiosqlite connection. A ``PRAGMA busy_timeout`` set on a SEPARATE
``pux_threads`` connection would NOT apply to the saver's writes — so two
processes (``pux direct`` + the Aegra runtime) hitting the same DB would
intermittently raise ``database is locked``. :func:`open_thread_store` opens ONE connection,
sets WAL + ``busy_timeout=5000`` on it, then constructs ``AsyncSqliteSaver(conn)``
(the documented raw form) so the saver and the index share the same hardened
connection. ``tests/test_threads.py`` proves the multi-process case.

The TUI (``pux tui`` / dcode) is intentionally NOT a consumer — it owns its own
``~/.deepagents/`` state and cannot be pointed at an external server.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from pux_harness.kit._paths import project_root

# The checkpointer DB path. ``$PUX_API_DB`` overrides; otherwise the default is
# resolved LIVE (no import-time snapshot) from ``project_root()`` so a late
# ``$PUX_PROJECT_ROOT`` is still honored. Kept as a module attribute so tests
# can still monkeypatch ``pux_harness.threads.PUX_API_DB`` hermetically (read at
# call time in ``open_thread_store``); only the project-root default is now live.
PUX_API_DB = os.environ.get("PUX_API_DB")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ThreadStore:
    """The shared checkpointer + the ``pux_threads`` index, one connection.

    ``saver`` is the langgraph checkpointer every graph is built against.
    ``db`` is the SAME aiosqlite connection the saver holds (the raw
    ``AsyncSqliteSaver(conn)`` form), so the WAL + ``busy_timeout`` pragmas set
    at open cover both the saver's writes and the index's.
    """

    saver: AsyncSqliteSaver
    db: aiosqlite.Connection

    async def register_thread(
        self, thread_id: str, org: str, metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a thread in the ``pux_threads`` index.

        Idempotent on ``thread_id`` (``INSERT OR IGNORE``) so a caller resuming
        an existing thread (``pux direct --thread <id>``) can re-register it
        without a duplicate-key failure.
        """
        await self.db.execute(
            "INSERT OR IGNORE INTO pux_threads (thread_id, org, metadata, created_at) "
            "VALUES (?,?,?,?)",
            (thread_id, org, json.dumps(metadata or {}), _now()),
        )
        await self.db.commit()

    async def list_threads(self, org: str | None = None) -> list[dict[str, Any]]:
        """Return ``pux_threads`` rows (newest first), optionally filtered by org.

        Each row is ``{"thread_id", "org", "metadata", "created_at"}``. Backs the
        ACP ``session/list`` surface so a client (Hermes daemon, acpx) can
        enumerate an org's sessions across ``pux acp`` process restarts.
        """
        sql = (
            "SELECT thread_id, org, metadata, created_at FROM pux_threads"
            + (" WHERE org = ?" if org is not None else "")
            + " ORDER BY created_at DESC"
        )
        params: tuple[Any, ...] = (org,) if org is not None else ()
        cur = await self.db.execute(sql, params)
        rows = await cur.fetchall()
        return [
            {"thread_id": r[0], "org": r[1], "metadata": r[2], "created_at": r[3]}
            for r in rows
        ]

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Return one ``pux_threads`` row by id, or ``None`` if absent.

        Backs the ACP ``session/load`` existence check: ``load_session`` verifies
        the requested ``session_id`` is ours (and belongs to this org) before
        handing the client a handle to resume.
        """
        cur = await self.db.execute(
            "SELECT thread_id, org, metadata, created_at FROM pux_threads "
            "WHERE thread_id = ?",
            (thread_id,),
        )
        r = await cur.fetchone()
        if r is None:
            return None
        return {"thread_id": r[0], "org": r[1], "metadata": r[2], "created_at": r[3]}

    async def merge_metadata(
        self, thread_id: str, metadata: dict[str, Any],
    ) -> bool:
        """Shallow-merge ``metadata`` into the thread's stored metadata (the
        ``PATCH /threads/{id}`` path — the SDK ``threads.update``).

        Returns ``False`` when the thread isn't registered (caller raises 404);
        ``True`` on a successful merge. The merge is shallow (top-level keys
        overwritten/added), matching langgraph-api's behavior.
        """
        cur = await self.db.execute(
            "SELECT metadata FROM pux_threads WHERE thread_id = ?", (thread_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        if row is None:
            return False
        existing = json.loads(row[0] or "{}")
        existing.update(metadata)
        await self.db.execute(
            "UPDATE pux_threads SET metadata = ? WHERE thread_id = ?",
            (json.dumps(existing), thread_id),
        )
        await self.db.commit()
        return True


@asynccontextmanager
async def open_thread_store(
    db_path: Path | None = None,
) -> AsyncIterator[ThreadStore]:
    """Open the ONE shared checkpointer + thread index.

    Opens a single aiosqlite connection, sets ``WAL`` + ``busy_timeout=5000`` on
    it, runs ``saver.setup()`` (creates the ``checkpoints`` / ``writes`` /
    ``migration`` tables), creates the ``pux_threads`` index table, then yields a
    :class:`ThreadStore` sharing that connection. The connection closes on
    context exit.

    Reads the module-level :data:`PUX_API_DB` at CALL time (not import time), so
    tests can monkeypatch ``pux_harness.threads.PUX_API_DB`` hermetically.
    """
    if db_path is not None:
        path = Path(db_path)
    elif PUX_API_DB:
        path = Path(PUX_API_DB)  # env override or a test's monkeypatch
    else:
        path = project_root() / ".pux" / "agent-protocol.sqlite"  # live default
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path))
    try:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS pux_threads ("
            "thread_id TEXT PRIMARY KEY, org TEXT, metadata TEXT, created_at TEXT)"
        )
        await conn.commit()
        yield ThreadStore(saver=saver, db=conn)
    finally:
        await conn.close()
