"""Cross-session project registry + switch banner.

The ``/sandbox/workspace`` bind mount is a transparent window onto a host
directory — it is NOT storage. Whatever directory you launch ``pux`` from
(``$PUX_PROJECT_PATH``) gets mounted, so launching from a different project
in a new session points that window at a different host dir. The previous
project's files are SAFE on disk where they always were; only the window
moved.

Without a record of which projects exist and which was last used, a session
that launches from a different directory silently shows a different
workspace — and a user (or a recovery agent searching *inside* the new
container) can conclude the previous work was lost. It wasn't. This module
records every project path the harness has bound and emits a calm, explicit
banner whenever a session switches projects, naming the previous path and
the exact command to resume it.

Storage: a user-level JSON registry at ``~/.pux/projects.json`` (host-level
on purpose — it must outlive any single project's ``.pux/`` dir). Honors
``PUX_STATE_HOME`` for test redirection. Every function is best-effort:
wayfinding is advisory and must NEVER block a sandbox from starting, so all
I/O errors are swallowed with a log line.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_LOG_PREFIX = "pux projects:"

# --- registry location ------------------------------------------------------


def _state_dir() -> Path:
    """User-level state dir. ``PUX_STATE_HOME`` wins, else ``~/.pux``.

    Host-level (not per-project) so the registry survives regardless of which
    project is currently bound — the whole point is to remember projects
    OTHER than the current one."""
    override = os.environ.get("PUX_STATE_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".pux"


def registry_path() -> Path:
    return _state_dir() / "projects.json"


# --- load / save ------------------------------------------------------------


def load() -> dict:
    """Read the registry. Returns ``{"projects": {}}`` on any failure
    (missing file, corrupt JSON, permission error) — never raises."""
    p = registry_path()
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
    except FileNotFoundError:
        return {"projects": {}}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{_LOG_PREFIX} could not read {p} ({exc}); starting fresh",
              file=sys.stderr)
        return {"projects": {}}
    if not isinstance(data, dict) or not isinstance(data.get("projects"), dict):
        return {"projects": {}}
    return data


def save(data: dict) -> None:
    """Atomic write (tmp + rename). Best-effort; failures log to stderr."""
    p = registry_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(p)
    except OSError as exc:
        print(f"{_LOG_PREFIX} could not write {p} ({exc}); registry not updated",
              file=sys.stderr)


# --- public API -------------------------------------------------------------


def record(project_path: str, sandbox_id: str = "", org: str = "") -> None:
    """Upsert a project entry with a fresh ``last_used`` timestamp.

    Called from :meth:`SandboxContainer.ensure` after a successful boot/reuse
    so the registry always reflects the most recent state. Idempotent;
    survives duplicate calls within one session.

    Stores BOTH a human-readable ISO ``last_used`` (for display) and a
    high-resolution ``ts`` epoch float (for ordering) — two records within
    the same wall-clock second would otherwise tie on the second-granularity
    ISO string, and a stable sort would then keep the older entry first,
    breaking the "newest-first" contract."""
    if not project_path:
        return
    project_path = os.path.abspath(project_path)
    now = datetime.now(timezone.utc)
    data = load()
    data["projects"][project_path] = {
        "last_used": now.isoformat(timespec="seconds"),
        "ts": now.timestamp(),
        "sandbox_id": sandbox_id,
        "org": org,
    }
    save(data)


def list_projects() -> list[dict]:
    """All known projects, newest-first. Each row:
    ``{path, last_used, sandbox_id, org, exists}``.

    Ordering uses the high-resolution ``ts`` epoch float so two records within
    the same wall-clock second still order correctly (a second-granularity ISO
    string would tie and fall back to insertion order). Pre-``ts`` entries
    (recorded before this field existed) parse their ISO ``last_used`` back to
    an epoch so they sort in alongside the rest."""
    data = load()
    rows = []
    for path, meta in data.get("projects", {}).items():
        ts = meta.get("ts")
        if ts is None:
            iso = meta.get("last_used", "")
            try:
                ts = datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
            except ValueError:
                ts = 0.0
        rows.append({
            "path": path,
            "last_used": meta.get("last_used", ""),
            "ts": float(ts),
            "sandbox_id": meta.get("sandbox_id", ""),
            "org": meta.get("org", ""),
            "exists": Path(path).exists(),
        })
    rows.sort(key=lambda r: r["ts"], reverse=True)
    return rows


def previous_project(current_path: str) -> dict | None:
    """The most-recently-used project that is NOT ``current_path`` and whose
    host path still exists on disk. Returns ``None`` when there is no such
    project (first run, or the only known project is the current one)."""
    if not current_path:
        return None
    current_path = os.path.abspath(current_path)
    for row in list_projects():
        if row["path"] == current_path:
            continue
        if row["exists"]:
            return row
    return None


def format_switch_banner(current_path: str) -> str | None:
    """Render the project-switch notice, or ``None`` if no switch occurred.

    The banner is the single fix for the "I lost all my work" panic: it names
    the previous project, states in plain language that its files are SAFE at
    their host path, and gives the literal ``cd … && pux …`` command to
    resume it. Calm by design — no alarmist language."""
    prev = previous_project(current_path)
    if prev is None:
        return None
    current_path = os.path.abspath(current_path)
    prev_path = prev["path"]
    last = prev["last_used"]
    return (
        "\n"
        "──────────────────────────── project switch ────────────────────────────\n"
        f"  You last worked on:  {prev_path}\n"
        f"    (last used {last})  — files there are SAFE on disk, unchanged.\n"
        f"  This session binds:  {current_path}\n"
        "    → /sandbox/workspace now points at the path above.\n"
        "\n"
        "  Nothing was lost. /sandbox/workspace is a bind-mount (a window onto\n"
        "  a host directory), not storage. The previous project's files are still\n"
        f"  at {prev_path}.\n"
        "\n"
        "  To resume the previous project instead:\n"
        f"    cd {prev_path} && pux acp\n"
        "  To list every known project:\n"
        "    pux sandbox projects\n"
        "────────────────────────────────────────────────────────────────────────\n"
    )


def warn_if_switched(current_path: str) -> bool:
    """Print the switch banner to stderr if a project switch is detected.

    Returns True if a banner was printed. Called from
    :meth:`SandboxContainer.ensure` BEFORE the container is created/reused,
    so the user sees the notice immediately on boot. Best-effort: never
    raises."""
    try:
        banner = format_switch_banner(current_path)
    except Exception as exc:  # noqa: BLE001 — wayfinding must never block boot
        print(f"{_LOG_PREFIX} switch-check failed ({exc})", file=sys.stderr)
        return False
    if banner:
        sys.stderr.write(banner)
        sys.stderr.flush()
        return True
    return False
