"""Priority-tiered structured snapshot builder (Phase 11).

Reads events from the EventStore and builds a ≤2KB XML resume snapshot
that survives compaction.  Modeled after mksglu/context-mode's
``buildResumeSnapshot()`` — each section includes a ``ctx_search`` tool
call for on-demand full retrieval.

The snapshot is a *table of contents*, not a data dump.  Full data lives
in the event store; the snapshot tells the resuming agent *what happened*
and *how to look up details*.

Budget enforcement:
- Target ≤2KB (≈500 tokens)
- P1 sections always included (files, tasks, rules, errors)
- P2 sections included next (git, decisions, env)
- P3/P4 sections dropped first if budget exceeded
- Each section truncates item count before dropping entirely
"""
from __future__ import annotations

import html
from typing import Any

from pux_harness.context.events import (
    P1,
    P2,
)

# Maximum snapshot size in bytes.
MAX_SNAPSHOT_BYTES = 2048

# Section renderers return (priority, tag, lines).  The builder collects
# all sections, sorts by priority, and trims from the tail until budget fits.
Section = tuple[int, str, list[str]]


def _esc(text: str) -> str:
    """XML-escape text content."""
    return html.escape(text, quote=True)


def _build_files_section(events: list[Any], search_tool: str) -> Section:
    """<files count="N"> — active file operations, deduped by path."""
    file_events = [e for e in events if e.category == "file" and e.type in ("file_modified",)]
    if not file_events:
        return (99, "files", [])

    # Dedupe by data (path), keep last occurrence per path.
    seen: dict[str, Any] = {}
    for ev in file_events:
        path = ev.data.get("path", "") if isinstance(ev.data, dict) else str(ev.data)
        if path:
            seen[path] = ev

    paths = list(seen.keys())[-10:]  # last 10 files
    lines = [f'  <files count="{len(seen)}">']
    for p in paths:
        lines.append(f"    <file>{_esc(p)}</file>")
    if search_tool:
        queries = ", ".join(f'"{_esc(p.split("/")[-1])}"' for p in paths[:4])
        lines.append(f"    <search>Use {search_tool} to retrieve full details: {queries}</search>")
    lines.append("  </files>")
    return (P1, "files", lines)


def _build_tasks_section(events: list[Any], search_tool: str) -> Section:
    """<tasks> — pending/active tasks from task events."""
    task_events = [e for e in events if e.category == "task"]
    if not task_events:
        return (99, "tasks", [])

    lines = ['  <tasks>']
    for ev in task_events[-8:]:
        data = ev.data if isinstance(ev.data, dict) else {"raw": str(ev.data)}
        status = data.get("status", ev.type.replace("task_", ""))
        desc = data.get("description", data.get("task", data.get("raw", "")))
        lines.append(f'    <task status="{_esc(str(status))}">{_esc(str(desc)[:120])}</task>')
    lines.append("  </tasks>")
    return (P1, "tasks", lines)


def _build_errors_section(events: list[Any], search_tool: str) -> Section:
    """<errors count="N"> — unresolved errors."""
    error_events = [e for e in events if e.category == "error"]
    if not error_events:
        return (99, "errors", [])

    lines = [f'  <errors count="{len(error_events)}">']
    for ev in error_events[-5:]:
        data_str = ev.data.get("error", str(ev.data)) if isinstance(ev.data, dict) else str(ev.data)
        lines.append(f"    <error>{_esc(data_str[:200])}</error>")
    lines.append("  </errors>")
    return (P1, "errors", lines)


def _build_decisions_section(events: list[Any], search_tool: str) -> Section:
    """<decisions count="N"> — key decisions made."""
    decision_events = [e for e in events if e.category == "decision"]
    if not decision_events:
        return (99, "decisions", [])

    seen: set[str] = set()
    lines = ['  <decisions>']
    for ev in decision_events[-5:]:
        data_str = str(ev.data) if not isinstance(ev.data, dict) else str(ev.data.get("decision", ev.data))
        if data_str in seen:
            continue
        seen.add(data_str)
        lines.append(f"    <decision>{_esc(data_str[:150])}</decision>")
    lines.append("  </decisions>")
    return (P2, "decisions", lines)


def _build_git_section(events: list[Any], search_tool: str) -> Section:
    """<git> — recent git operations."""
    git_events = [e for e in events if e.category == "git"]
    if not git_events:
        return (99, "git", [])

    lines = ['  <git>']
    for ev in git_events[-5:]:
        data_str = str(ev.data) if not isinstance(ev.data, dict) else str(ev.data.get("operation", ev.data))
        lines.append(f"    <operation>{_esc(data_str[:100])}</operation>")
    lines.append("  </git>")
    return (P2, "git", lines)


def _build_env_section(events: list[Any], search_tool: str) -> Section:
    """<env> — environment changes (cwd, env vars)."""
    env_events = [e for e in events if e.category == "env"]
    if not env_events:
        return (99, "env", [])

    lines = ['  <env>']
    for ev in env_events[-3:]:
        data_str = str(ev.data) if not isinstance(ev.data, dict) else str(ev.data)
        lines.append(f"    <change>{_esc(data_str[:100])}</change>")
    lines.append("  </env>")
    return (P2, "env", lines)


def _build_rules_section(events: list[Any], search_tool: str) -> Section:
    """<rules count="N"> — project rules/constraints captured."""
    rule_events = [e for e in events if e.category == "rule"]
    if not rule_events:
        return (99, "rules", [])

    lines = [f'  <rules count="{len(rule_events)}">']
    for ev in rule_events[-3:]:
        data_str = str(ev.data) if not isinstance(ev.data, dict) else str(ev.data)
        lines.append(f"    <rule>{_esc(data_str[:200])}</rule>")
    lines.append("  </rules>")
    return (P1, "rules", lines)


def _build_summary_line(events: list[Any], thread_id: str) -> str:
    """One-line session summary for the top of the snapshot."""
    total = len(events)
    p1_count = sum(1 for e in events if e.priority <= P1)
    p2_count = sum(1 for e in events if e.priority == P2)
    cats = set(e.category for e in events if e.category)
    return (
        f'<session_summary events="{total}" p1="{p1_count}" p2="{p2_count}" '
        f'categories="{_esc(",".join(sorted(cats)))}" '
        f'thread="{_esc(thread_id)}"/>'
    )


# -- Public API ----------------------------------------------------------------

def build_snapshot(
    events: list[Any],
    *,
    thread_id: str = "",
    search_tool: str = "ctx_search",
    max_bytes: int = MAX_SNAPSHOT_BYTES,
) -> str:
    """Build a ≤max_bytes XML resume snapshot from events.

    Events should be in chronological order (oldest first).  The builder
    renders sections by priority (P1 first), then trims from the tail
    until the total fits the byte budget.

    Returns the XML string ready for injection into the agent's context.
    """
    if not events:
        return "<session_snapshot empty=\"true\"/>"

    section_renderers = [
        _build_files_section,
        _build_tasks_section,
        _build_errors_section,
        _build_rules_section,
        _build_decisions_section,
        _build_git_section,
        _build_env_section,
    ]

    sections: list[Section] = []
    for renderer in section_renderers:
        sections.append(renderer(events, search_tool))

    # Sort by priority (P1 first), then by original order for same priority.
    indexed = list(enumerate(sections))
    indexed.sort(key=lambda x: (x[1][0], x[0]))

    # Build the XML, trimming from the tail if over budget.
    summary = _build_summary_line(events, thread_id)
    lines = ["<session_snapshot>", summary]

    included_tags: list[str] = []
    for _, (prio, tag, section_lines) in indexed:
        candidate = "\n".join(lines + section_lines + ["</session_snapshot>"])
        if len(candidate.encode("utf-8")) > max_bytes and included_tags:
            break  # budget exceeded — stop adding sections
        lines.extend(section_lines)
        included_tags.append(tag)

    lines.append("</session_snapshot>")
    result = "\n".join(lines)

    # Final truncation if still over budget (shouldn't happen with section
    # trimming, but safety net).
    encoded = result.encode("utf-8")
    if len(encoded) > max_bytes:
        result = encoded[:max_bytes].decode("utf-8", errors="ignore")
        # Close the tag if we truncated mid-tag.
        if "</session_snapshot>" not in result:
            result = result.rstrip() + "\n</session_snapshot>"

    return result
