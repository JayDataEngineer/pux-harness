"""Run-completion event bus — SSE fan-out + persisted catch-up log.

Closes the gap an MCP client that fired ``start_run`` (non-blocking) and has NO
HTTP webhook receiver otherwise cannot close: with nothing pushing completion it
must poll ``list_runs``. A client that *can't* host a receiver (e.g. Hermes, a
Telegram bot behind NAT — "Hermes can't make webhooks on the sandbox") instead
subscribes ONCE to ``GET /events/stream`` and receives every ``run.completed``
as it happens — across all orgs/threads/runs — or catches up via
``GET /events?since=<ts>``.

``server.py`` publishes here at the SAME call site as the outbound webhook
(``_dispatch_run_webhook``), so the SSE stream and the outbound POST carry
IDENTICAL payloads (``run_id``/``status``/``output``/``event="run.completed"``).
The bus is the receiver-of-last-resort that lives on the pux side, so the caller
never has to expose an endpoint.

Stdlib only (``asyncio``/``json``/``pathlib``) — no new dependency.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    """Timezone-aware UTC ISO-8601 (microsecond resolution), lexicographically
    sortable. Microsecond resolution so two completions in the same millisecond
    still order distinctly for catch-up (``?since=<ts>``)."""
    return (
        datetime.now(tz=timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass
class EventBus:
    """In-process pub/sub for run-completion events + bounded ring + jsonl log.

    - ``publish`` fans out to every live SSE subscriber AND appends to the ring
      (in-memory catch-up) + the optional jsonl log (cross-restart catch-up).
      Each event gets a monotonic ``seq`` (deterministic ordering + client
      dedup) and a server-stamped ``ts``.
    - ``subscribe`` returns an ``asyncio.Queue`` the SSE endpoint drains; a full
      queue drops (slow subscriber) — ``?since=`` covers the gap on reconnect.
    - No delivery guarantees ([[no-fallbacks-no-aliases]]: this is a notification
      channel, not a reliability gate; a missed event degrades to the existing
      poll-with-``list_runs`` path). The ring + log make re-sync cheap.
    """

    cap: int = 200
    log_path: Path | None = None
    _subs: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)
    _ring: deque[dict[str, Any]] = field(default_factory=deque)
    _seq: int = 0

    def __post_init__(self) -> None:
        self._ring = deque(maxlen=self.cap)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    async def publish(self, event: dict[str, Any]) -> None:
        ev = dict(event)
        ev.setdefault("ts", _now_iso())
        ev["seq"] = self._seq
        self._seq += 1
        self._ring.append(ev)
        if self.log_path:
            try:
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(ev, default=str) + "\n")
            except OSError:  # disk full / read-only fs — bus still works in-memory
                pass
        for q in list(self._subs):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                pass  # slow subscriber; reconnect via ?since=

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subs.discard(q)

    def recent(self, since: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Most-recent events, optionally filtered to ``ts > since``.

        Lexicographic ISO-8601 comparison == chronological here because every
        stamp is the same fixed-width UTC microsecond format.
        """
        items = list(self._ring)
        if since:
            items = [e for e in items if str(e.get("ts", "")) > since]
        if limit > 0:
            items = items[-limit:]
        return items

    def health(self) -> dict[str, Any]:
        return {"ok": True, "subscribers": len(self._subs), "events": len(self._ring)}
