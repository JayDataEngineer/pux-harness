# deepagents-context

Proactive context management for [deepagents](https://github.com/langchain-ai/deepagents) — capture, offload, and retrieval middleware that keeps large tool results out of the context window **before** they accumulate, complementing deepagents' reactive `SummarizationMiddleware` (which only evicts on overflow).

Zero pux-harness coupling. Pure stdlib `sqlite3` + langchain/langgraph/deepagents.

## Installation

```bash
uv add deepagents-context
# or
pip install deepagents-context
```

## Quick start (native deepagents)

```python
from deepagents_context import (
    EventStore,
    ContextMiddleware,
    build_context_layer,
)
from deepagents import create_deep_agent

# 1. Create a store (SQLite-backed, FTS5-searchable)
store = EventStore(".myapp/events.sqlite")

# 2. Get the middleware + retrieval tools in one call
middleware, tools = build_context_layer(store=store, threshold=8000)

# 3. Wire into any deepagents agent
agent = create_deep_agent(
    tools=[*my_tools, *tools],
    middleware=middleware,
    model=model,
)
```

That's it. Every tool call is now captured as a structured event, oversized results are automatically stashed behind `ctx:<id>` handles, and the agent can retrieve them on demand via `ctx_recall` / `ctx_search`.

## What's in the box

### Middleware

| Class | What it does |
|---|---|
| `ContextMiddleware` | Capture every tool call as an event AND offload oversized results (>threshold chars) to a blob — one `wrap_tool_call` pass, one store. |
| `PromptCaptureMiddleware` | Captures user prompts (`before_model`) and final assistant turns (`after_agent`) — the UserPromptSubmit + Stop equivalents. |
| `SessionGuideMiddleware` | Cross-session rehydration: builds a structured snapshot on compaction, injects it as `<session_knowledge>` on resume. |
| `FullPrefixCachingMiddleware` | Tags 3 Anthropic cache breakpoints (system prompt + last tool + last message) — the stock middleware misses the rolling conversation history. |
| `AuditMiddleware` | Observe-only tool-call audit: records every call (name, hashed args, outcome, elapsed) without touching the result. Opt-in. |

### Store

| Object | Description |
|---|---|
| `EventStore` | SQLite-backed event + blob store with FTS5 full-text search. Thread-safe, process-wide singleton via `shared_event_store()`. |
| `shared_event_store()` | Returns the process-wide singleton (3-tier path resolution via env vars). |

### Tools (agent-callable)

`build_context_tools(store)` returns:

| Tool | Purpose |
|---|---|
| `ctx_search` | BM25 search across offloaded blobs + structured events. |
| `ctx_recall` | Pull the full content of a stashed blob by its `ctx:<id>` handle. |
| `ctx_index` | Manually park reference text as a searchable blob. |
| `ctx_stats` | Event/blob counts, db size, FTS5 status. |
| `ctx_doctor` | Health checks: store reachable? FTS5 compiled in? db writable? |
| `ctx_purge` | Delete all events + blobs (or a single thread's). |

### Builders

| Function | Description |
|---|---|
| `build_context_layer(store, *, threshold, preview, enabled, extra_tools)` | Returns `(middleware_list, tools_list)` — the single wiring seam. |
| `build_snapshot(events, *, thread_id, search_tool)` | Builds a structured XML snapshot from recent events for session rehydration. |

## Configuration

### Offload threshold

```python
ContextMiddleware(store, threshold=8000, preview=1500)
# threshold: results this size or larger are stashed (default 8000 chars ≈ 2K tokens)
# preview: how much of a stashed result to keep inline in the stub message
# threshold <= 0 disables offload entirely (kill-switch)
```

### Database path (3-tier resolution)

```bash
# Tier 1: explicit path (native users)
export DEEPAGENTS_CONTEXT_DB=/data/events.sqlite

# Tier 2: pux compat (uses $PUX_PROJECT_ROOT/.pux/events.sqlite)
export PUX_PROJECT_ROOT=/path/to/project

# Tier 3: generic default (cwd/.deepagents-context/events.sqlite)
# — no env var needed
```

## How it works

### Proactive offload (not reactive)

`SummarizationMiddleware` evicts messages **after** they overflow the window — the damage is already done (tokens were billed, then summarization costs more tokens to compress). `ContextMiddleware` intercepts oversized tool results **before** they enter the message history:

1. Tool returns 20K-char directory listing
2. Middleware stashes the full content as a blob in SQLite
3. Model sees a 1.5K-char stub: `tool returned 20000 chars (first 1500 shown). For the complete output, call ctx_recall('ctx:1a2b3c4d').`
4. On the next turn, the 20K chars are NOT in the resent message history — only the 1.5K stub is

The agent retrieves the full content only when it actually needs it, via `ctx_recall`.

### Capture (structured activity feed)

Every tool call is recorded as a structured event: tool name, truncated args, success/error, elapsed seconds, and a 300-char output preview. These events feed:
- The snapshot builder (cross-session rehydration)
- `ctx_search` (query-based recall over both blobs and events)

### Retrieval (on-demand re-entry)

Only the slice the agent asks for re-enters its context. `ctx_search` returns matching handles + a snippet each; `ctx_recall` pulls the full content of one handle. Fewer tokens recalled = lower cost per call.

## Architecture

```
deepagents_context/
├── store.py            # EventStore — SQLite + FTS5, thread-safe singleton
├── middleware.py       # ContextMiddleware — capture + offload
├── prompt_capture.py   # PromptCaptureMiddleware — prompts + turn-ends
├── session_guide.py    # SessionGuideMiddleware — cross-session rehydration
├── snapshot.py         # build_snapshot — structured XML from events
├── prefix_caching.py   # FullPrefixCachingMiddleware — 3-breakpoint Anthropic caching
├── audit.py            # AuditMiddleware — observe-only tool-call audit
├── tools.py            # ctx_recall, ctx_search, ctx_index, ctx_stats, ctx_doctor, ctx_purge
├── layer.py            # build_context_layer — the wiring seam
├── interpreter_hints.py
├── read_file_vision.py
└── browser_vision.py
```

## Requirements

- Python ≥3.12, <3.14
- langchain ≥0.3
- langgraph ≥0.6
- langchain-anthropic ≥1.5.3

## License

MIT
