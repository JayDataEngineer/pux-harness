# pux-harness — the portable kit (slim core)

Compile a folder of **org + skills** into a running [deepagents](https://github.com/langchain-ai/deepagents) agent — **with no Docker, no sandbox, no server.**

The org+skill compiler is the **slim core** of `pux-harness`: it lives at `pux_harness.kit` and is re-exported at the top level (`from pux_harness import compile_org`). Everything heavy (the Docker sandbox, the context/memory/browser-vision stack, the ACP/server/TUI runtimes) attaches to `pux-harness` as **optional-dependency extras**; the kit's import graph pulls none of them. Use it from a *different, standalone* project (e.g. a Wan2GP image-gen app with a CopilotKit chat UI) that wants to author its own org + a skill, plug its own tools in, and wire the compiled graph to its UI.

## What an org is

An org is a self-contained folder of files — pure prose + YAML, no Python:

```
my_app/
├── AGENTS.md                         # base system prompt (optional)
└── orgs/
    └── my_org/
        ├── AGENTS.md                 # the CTO / supervisor overlay (prose)
        ├── org.yaml                  # roster: agents: [form_builder]
        ├── agents/
        │   └── form_builder.md       # ONE specialist: frontmatter + body
        └── skills/
            └── wan2gp/
                └── SKILL.md          # a skill bundle
```

- **`orgs/<name>/AGENTS.md`** — prose appended to the system prompt (the supervisor's persona/instructions).
- **`orgs/<name>/org.yaml`** — `agents: [slug, ...]`, the specialists this org delegates to.
- **`orgs/<name>/agents/<slug>.md`** — one specialist. YAML frontmatter (`name`, `description`, optional `tools` / `skills` / `model`) + a markdown body that **is** the specialist's system prompt (mirrors `SKILL.md`).
- **`orgs/_shared/agents/<slug>.md`** — a specialist shared across orgs (org-local file wins on name collision).
- **skills** — a `SKILL.md` bundle under a skills **root** dir (e.g. `orgs/<name>/skills/` or `orgs/_shared/skills/`). A frontmatter `skills: [orgs/my_org/skills]` points at a *root*, and every `<root>/<skill>/SKILL.md` beneath it loads.

## Quick start

Install the core (Stage 3 will split docker/fastapi/copilotkit out into optional extras so the bare install is truly slim; today they're still hard install-time deps, but importing the kit pulls **none** of them):

```bash
pip install pux-harness
```

Compile + run:

```python
from pux_harness import compile_org
from langgraph.checkpoint.memory import MemorySaver

graph = compile_org(
    "my_org",
    model=my_chat_model,          # any langchain BaseChatModel (or a model id)
    tools=[my_wan2gp_tool],       # YOUR tools — the agent's real surface
    project_root="./my_app",      # dir containing orgs/ + the root AGENTS.md
    checkpointer=MemorySaver(),
)
out = graph.invoke({"messages": [{"role": "user", "content": "..."}]})
```

`compile_org` returns a deepagents `CompiledStateGraph`. Wire it to whatever transport you use (CopilotKit / AG-UI / plain HTTP) — the kit ships no transport on purpose.

### What you supply

- **`model`** — the supervisor driver (any `BaseChatModel`, or a model id string deepagents can resolve).
- **`tools`** — your app's tools. The kit adds **no** `pux_sandbox_*` tools; an agent's `tools:` frontmatter is whitelisted by exact name against *your* list (unknown names are skipped, so an org authored under the pux harness still compiles here even if it references sandbox tools you don't ship).

### What the kit does *not* bring

The pux Docker sandbox, the pux context/memory/browser-vision middleware, the rubric gate, profile overrides. A consumer that wants any of those installs the matching `pux-harness` extra (Stage 3) or uses the harness directly. The kit uses deepagents' local `FilesystemBackend` so skills resolve on the host filesystem and the agent gets local read/write/shell tools — no container.

## Run the demo

```bash
python examples/run_example.py
```

Compiles the Wan2GP-shaped org under `examples/orgs/wan2gp_demo/` with a scripted (offline) model and prints a stub turn that calls `generate_form`. Swap the scripted model for a real LLM and the stub tool for a Wan2GP driver to make it a real app.

## API

| function | purpose |
|---|---|
| `compile_org(org, *, model, tools, middleware=(), checkpointer=None, project_root=None, addendum=None, skills=None, subagents=None, backend="filesystem")` | the one entry point. Returns a compiled graph. |
| `load_subagents(org, tools, *, project_root, model=None, model_resolver=None)` | build SubAgent dicts from the roster (used internally; exposed for custom flows). |
| `discover_orgs(project_root)` | list org names under `orgs/` + `orgs/specialists/`. |
| `org_agent_slugs(name, project_root)` | the specialist slugs from `org.yaml`. |
| `load_root_prompt(project_root)` / `load_org_prompt(name, project_root)` | the prompt bodies. |
| `build_system_prompt(org, *, project_root, addendum="")` | root + overlay + addendum. |

## Why the kit is a slim core, not a separate package

`pux-harness` does two things: it compiles orgs+skills (portable, Docker-free) **and** it runs them inside a Docker sandbox with a context/memory/browser stack. The compile half is the part an unrelated app wants; the run half is pux-specific baggage. Rather than split that into a second distribution defined by what it *isn't* (which drifts), the compile half lives at `pux_harness.kit` as a positively-named slim core, and the heavy run half attaches as optional extras. A lazy-import + the isolation tripwire (`tests/test_kit_compile.py::test_import_isolation_no_docker_no_heavy_subsystem`) enforce the boundary: importing the kit loads neither `docker` nor any heavy `pux_harness` subsystem.
