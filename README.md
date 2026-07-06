# pux-harness

The deepagents-based Pux harness, as a standalone library. Two surfaces, one
package:

- **`pux_harness.kit`** — the slim, Docker-free org+skill **compiler**
  (`from pux_harness import compile_org`). Turns a folder of org + skills into
  a running deepagents `CompiledStateGraph` with no sandbox, no server, no
  context/memory middleware. The typical consumer is a different, standalone
  project (e.g. a Wan2GP + CopilotKit app) that authors its own org + skills
  and wires the compiled graph to its UI. The kit's import graph is pinned slim
  by a contract tripwire (`kit-import-isolation`) — it pulls neither `docker`
  nor any heavy `pux_harness.<subsystem>`.

- **The heavy runtimes** (`pux_harness.agent`, `.sandbox`, `.context`,
  `.browser`, the `serve`/`acp`/`direct`/`mcp`/`tui` entry points) — the Docker
  sandbox lifecycle, the unified thread store, the context/memory layer, the
  browser tooling, the `pux` console script. A consumer that wants any of this
  uses the harness directly.

## Location independence

The harness does **not** derive its app root from its install path. Every
subsystem that needs the consumer's app root (`orgs/`, `.pux/`, `AGENTS.md`)
calls `pux_harness.kit._paths.project_root()`, which reads `$PUX_PROJECT_ROOT`
(default: the process CWD). A consumer pins it once (e.g. its launcher exports
`PUX_PROJECT_ROOT=$APP_ROOT`) and the harness finds its orgs from anywhere.

## Quick start (kit)

```python
from pux_harness import compile_org
from langgraph.checkpoint.memory import MemorySaver

graph = compile_org(
    "my_org",
    model=my_chat_model,
    tools=[my_tool],
    project_root="./my_app",      # contains orgs/my_org/...
    checkpointer=MemorySaver(),
)
graph.invoke({"messages": [{"role": "user", "content": "..."}]})
```

See `examples/` for a runnable walk-through (the Wan2GP demo org).

## Repo layout

```
pux_harness/        the package (kit + heavy runtimes)
examples/           the kit quick-start demo
tests/              the library's own org-agnostic suite (kit, registry,
                    browser-JS, reasoning adapter, model resolver)
```

Tests that build against a real `orgs/` tree live in the **consumer** repo,
which installs `pux-harness` and runs them against its own orgs.

## Develop

```sh
uv sync
uv run pytest
```
