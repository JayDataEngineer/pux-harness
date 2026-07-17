"""Sweep inert session history under a project's ``.pux/`` so it stays bounded.

Three things accumulate per run, all under ``<project>/.pux/``:

* ``sessions/*.meta.json`` — one tiny (≈200 B) finished-thread record per run.
  Useful for ``pux resume`` history; worthless after a couple of weeks.
* ``run_events.jsonl`` — one append-only line per run (success/error + timing).
  Grows without bound; the one file that can get large.
* ``wild-logs/*.log`` — a full transcript per ``pux run`` (background / "wild"
  runs). Large per file; definitely worth pruning.

None of these are a leak (they are inert — no live process holds them) and they
do not cause slowness, but months of heavy use produces thousands of files /
one large JSONL. ``pux sandbox prune-sessions`` keeps them bounded on a
retention window (default 14 days). ``--dry-run`` previews.

DELIBERATELY untouched: ``.pux/agent-protocol.sqlite`` (the live thread store —
pruning it needs FK/active-thread awareness, out of scope) and
``.pux/venvs/`` (cached tool venvs — regenerated lazily, not history).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CATEGORIES = ("meta", "events", "wildlogs")


def _is_older(path: Path, cutoff_epoch: float) -> bool:
    try:
        return path.stat().st_mtime < cutoff_epoch
    except OSError:
        return False


def _parse_iso_ts(line: str) -> float | None:
    """Pull the ``ts`` field from a run_events.jsonl line → epoch seconds.

    Each event line is ``{"...": ..., "ts": "2026-07-08T21:46:25.543Z", ...}``.
    Lines with no parseable ``ts`` return None (caller falls back to file mtime)."""
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    ts = obj.get("ts") if isinstance(obj, dict) else None
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _prune_meta(sessions_dir: Path, cutoff: float, dry_run: bool) -> dict[str, Any]:
    if not sessions_dir.is_dir():
        return {"swept": [], "kept": 0}
    swept: list[str] = []
    kept = 0
    for p in sorted(sessions_dir.glob("*.meta.json")):
        if _is_older(p, cutoff):
            swept.append(str(p))
            if not dry_run:
                try:
                    p.unlink()
                except OSError:
                    pass
        else:
            kept += 1
    return {"swept": swept, "kept": kept}


def _prune_wildlogs(wild_dir: Path, cutoff: float, dry_run: bool) -> dict[str, Any]:
    if not wild_dir.is_dir():
        return {"swept": []}
    swept: list[str] = []
    for p in sorted(wild_dir.glob("*.log")):
        if _is_older(p, cutoff):
            swept.append(str(p))
            if not dry_run:
                try:
                    p.unlink()
                except OSError:
                    pass
    return {"swept": swept}


def _prune_events(events_file: Path, cutoff: float, dry_run: bool) -> dict[str, Any]:
    """Rewrite ``run_events.jsonl`` keeping only lines newer than ``cutoff``.

    A line's age is its ``ts`` field; lines without one fall back to the file's
    mtime (conservative — never drop a line we can't date). Atomic tmp+rename."""
    if not events_file.is_file():
        return {"removed_lines": 0, "kept_lines": 0}
    try:
        lines = events_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"removed_lines": 0, "kept_lines": 0}
    try:
        file_mtime = events_file.stat().st_mtime
    except OSError:
        file_mtime = time.time()
    kept: list[str] = []
    removed = 0
    for line in lines:
        if not line.strip():
            continue
        ts_epoch = _parse_iso_ts(line)
        if ts_epoch is None:
            ts_epoch = file_mtime
        if ts_epoch < cutoff:
            removed += 1
        else:
            kept.append(line)
    if not dry_run and removed > 0:
        tmp = events_file.with_suffix(".jsonl.tmp")
        tmp.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
        try:
            os.replace(tmp, events_file)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
    return {"removed_lines": removed, "kept_lines": len(kept)}


def prune_pux_dir(
    pux_dir: Path,
    *,
    days: int = 14,
    dry_run: bool = False,
    categories: Iterable[str] = CATEGORIES,
) -> dict[str, Any]:
    """Prune a project's ``.pux/`` history. Returns a structured report.

    ``days``     — retention window; entries older than this are swept (0 = all).
    ``dry_run``  — report only; do not unlink/rewrite.
    ``categories`` — subset of ``CATEGORIES`` (meta | events | wildlogs).
    """
    days = max(0, int(days))
    cutoff = time.time() - days * 86400
    cats = set(categories)
    report: dict[str, Any] = {
        "pux_dir": str(pux_dir),
        "days": days,
        "dry_run": dry_run,
    }
    if "meta" in cats:
        report["meta"] = _prune_meta(pux_dir / "sessions", cutoff, dry_run)
    if "events" in cats:
        report["events"] = _prune_events(pux_dir / "run_events.jsonl", cutoff, dry_run)
    if "wildlogs" in cats:
        report["wildlogs"] = _prune_wildlogs(pux_dir / "wild-logs", cutoff, dry_run)
    return report


def summarize(report: dict[str, Any]) -> str:
    """One-paragraph human summary of a prune report (for the CLI)."""
    head = (f"pux prune ({report['days']}d window, "
            f"{'DRY-RUN' if report['dry_run'] else 'applied'}): {report['pux_dir']}")
    parts = [head]
    meta = report.get("meta") or {}
    if meta:
        parts.append(f"  sessions/*.meta.json : swept {len(meta.get('swept', []))} "
                     f"file(s), kept {meta.get('kept', 0)}")
    ev = report.get("events") or {}
    if ev:
        parts.append(f"  run_events.jsonl     : removed {ev.get('removed_lines', 0)} "
                     f"line(s), kept {ev.get('kept_lines', 0)}")
    wl = report.get("wildlogs") or {}
    if wl:
        parts.append(f"  wild-logs/*.log      : swept {len(wl.get('swept', []))} file(s)")
    return "\n".join(parts)
