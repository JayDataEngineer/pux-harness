"""Pack an org into a standalone portable, runnable archive (``pux pack``).

``pux pack --org <name>`` produces a ``.tar.gz`` containing every primitive the
org needs to run outside the harness — prompts, agent definitions, skills,
policy, profile, shared dependencies, and a manifest. The archive is
self-contained: a consumer can reconstruct the org's agent graph without the
pux harness.

What ships is now **declared, not implicit**: the org-local tree is collected
default-deny through the declarative manifest (``pux_harness.manifest`` —
``package.include`` globs from ``org.yaml``; ``data/``/``.pux/`` permanently
excluded). The legacy ``pux export`` verb + the hardcoded ``_collect_org_files``
allowlist are GONE (permanent contract — see ``tests/export``); there is NO
silent alias. Use ``pux pack``.

Archive layout (mirrors the project tree, scoped to what the org uses)::

    <org-name>/
      AGENTS.md                      # root base prompt
      orgs/<name>/
        AGENTS.md                    # CTO prompt
        org.yaml                     # roster
        profile.yaml                 # (if present)
        policy.yaml                  # (if present)
        agents/*.md                  # specialist definitions
        skills/*/SKILL.md            # org-local skills
        sandbox/*.py                 # org-local scripts
        config/                      # (if present)
        Dockerfile                   # (if present)
      # NOTE: org ``data/`` is INTENTIONALLY EXCLUDED. It is runtime state
      # (auth sessions, market data, campaign state — see the root .gitignore,
      # which treats data/ as non-source), often containing live secrets such as
      # browser-session cookies. It is never a primitive an export reconstructs.
      orgs/_shared/
        agents/*.md                  # shared agents the org resolves
        skills/*/SKILL.md            # shared skills the org resolves
        sandbox/*.py                 # shared sandbox helpers the org references
        clients/*.py                 # shared clients the org references
        tool_servers.yaml            # (if policy declares tool_servers)
      pux_harness/                   # VENDORED slim kit (the portable compiler)
        __init__.py                  # re-exports compile_org
        kit/                         # compile.py, loaders, _bootstrap, _paths, _testing
      run.py                         # standalone runner (--check offline / "task")
      pyproject.toml                 # runtime deps (deepagents, langchain-openai, ...)
      README.md                      # how to run
      manifest.json                  # machine-readable inventory

The ``pux_harness/`` kit + ``run.py`` + ``pyproject.toml`` + ``README.md`` are
the **runtime scaffold (F3)**: they turn the primitives archive into a standalone
RUNNABLE package — a consumer runs the org WITHOUT pux-harness installed
(``pip install . && python run.py``). See ``_build_runtime_scaffold``.
"""
from __future__ import annotations

import json
import os
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import yaml

from pux_harness.agent.orgs import (
    _load_agent_spec,
    _org_path,
    _orgs_dir,
    _parse_list,
    discover_orgs,
    org_agent_slugs,
)
from pux_harness.kit._paths import _PROJECT_ROOT_ENV
from pux_harness.kit._paths import project_root as _default_project_root
from pux_harness.manifest import (
    collect_pack_files,
    load_manifest,
    manifest_metadata,
)
from pux_harness.pack_hooks import (
    PACK_HOOK_REGISTRY,
    HookContext,
    PackHook,
    provenance_from_results,
    run_pack_hooks,
)


# --- runtime scaffold (F3): turn the primitives archive into a RUNNABLE package --
#
# An export is not done when it carries the org's files — it must RUN without the
# harness installed. ``_build_runtime_scaffold`` vendors the slim kit + emits a
# ``pyproject.toml`` (runtime deps), ``run.py`` (entry point), and ``README.md``
# into the archive. A consumer then: ``pip install .`` (deps) → ``python run.py
# --check`` (offline compile smoke) → ``python run.py "task"`` (real model).
#
# The slim kit is import-self-contained (intra-kit + third-party only; no
# ``pux_harness.agent`` leakage — locked by the kit-import-isolation tripwire),
# so vendoring ``pux_harness/kit/`` verbatim makes ``from pux_harness.kit import
# compile_org`` resolve from the unpacked tree. Read from THIS package's kit dir.
_KIT_DIR = Path(__file__).resolve().parent / "kit"
# Runtime kit files (``_testing.py`` IS vendored — its ``ScriptedModel`` powers
# the runner's ``--check`` offline smoke + lets a consumer test their wired graph).
_KIT_RUNTIME_FILES = (
    "__init__.py", "_bootstrap.py", "_paths.py", "compile.py", "loaders.py", "_testing.py",
)

# The package-level __init__ for the VENDORED ``pux_harness``. Re-exports the kit
# entry point so ``from pux_harness import compile_org`` mirrors a full install.
_VENDORED_PKG_INIT = '''"""Vendored slim pux-harness kit — travels with this exported org.

The kit (``pux_harness.kit``) is the portable org+skill compiler: a folder of
org + skills -> a running deepagents agent, no Docker, no server. This top-level
``__init__`` re-exports its one entry point; see ``pux_harness/kit/__init__.py``.
"""
from pux_harness.kit import compile_org  # noqa: F401

__version__ = "0.1.0"
'''

# ``run.py`` — the standalone entry point. ``__ORG__`` / ``__MODEL__`` are sentinel
# replacements (NOT ``.format``) so the template's literal braces (dicts, f-strings)
# need no escaping.
_RUN_PY = '''#!/usr/bin/env python3
"""Standalone runner for the '__ORG__' org — exported from pux-harness.

Usage:
  pip install .                       # install runtime deps (deepagents, langchain-openai, ...)
  python run.py --check               # offline smoke: compile the org (no model/key needed)
  python run.py "your task here"      # run one task against a real model

Model config (via ./.env or shell env):
  PUX_MODEL=__MODEL__              OpenAI-compatible model id (default shown)
  OPENAI_API_KEY=...               auto-read by ChatOpenAI
  OPENAI_BASE_URL=https://...      OpenAI-compatible endpoint (omit for openai.com)
  PUX_PROJECT_ROOT=<dir>           where orgs/ lives (default: this archive dir)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# This archive dir holds orgs/ + the vendored pux_harness/. Resolved from THIS
# file (run.py sits at the archive root) and put first on sys.path so the
# VENDORED kit is used (not any site-packages pux_harness) — the runner then
# works from any cwd.
_ARCHIVE_ROOT = Path(__file__).resolve().parent
if str(_ARCHIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ARCHIVE_ROOT))

from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from pux_harness.kit import bootstrap_env_and_logging, compile_org  # noqa: E402

ORG = "__ORG__"
DEFAULT_MODEL = "__MODEL__"


def _project_root() -> str:
    return os.environ.get("PUX_PROJECT_ROOT") or str(_ARCHIVE_ROOT)


def _build_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("PUX_MODEL", DEFAULT_MODEL),
        temperature=float(os.environ.get("PUX_TEMPERATURE", "0.2")),
        max_tokens=int(os.environ.get("PUX_MAX_TOKENS", "8192")),
    )


def _print_final(result: dict) -> None:
    msgs = result.get("messages") or []
    last = msgs[-1] if msgs else None
    content = getattr(last, "content", last)
    if isinstance(content, list):
        content = "\\n".join(
            str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content
        )
    print(content if content else "(no output)")


def main() -> None:
    bootstrap_env_and_logging()
    args = sys.argv[1:]
    if args and args[0] == "--check":
        from pux_harness.kit._testing import ScriptedModel  # noqa: PLC0415
        graph = compile_org(ORG, model=ScriptedModel(), tools=[], project_root=_project_root())
        print(f"OK: {ORG} compiled -> {type(graph).__name__} (project_root={_project_root()})")
        return
    task = " ".join(args).strip()
    if not task:
        print(f'usage: python run.py "task for the {ORG} org"  (or --check)', file=sys.stderr)
        raise SystemExit(2)
    graph = compile_org(
        ORG, model=_build_model(), tools=[], checkpointer=MemorySaver(),
        project_root=_project_root(),
    )
    _print_final(graph.invoke({"messages": [{"role": "user", "content": task}]}))


if __name__ == "__main__":
    main()
'''

_PYPROJECT = '''[project]
name = "__ORG__-pux"
version = "0.1.0"
description = "Standalone runnable export of the '__ORG__' pux org (vendored slim kit)."
requires-python = ">=3.12,<3.14"
dependencies = [
    "deepagents",
    "langchain",
    "langchain-openai",
    "langgraph",
    "langgraph-checkpoint-memory",
    "pyyaml",
    "python-dotenv>=1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["pux_harness"]
'''

_README = '''# __ORG__ — standalone pux export

A self-contained, **runnable** export of the **__ORG__** org. It carries the
org's prompts, agents, skills, policy, and a **vendored** copy of the slim
`pux_harness.kit` compiler — so it runs **without installing pux-harness**. You
only need the runtime deps (deepagents, langchain-openai, ...).

## Run

```bash
cd __ORG__                 # the archive root (this dir)
pip install .              # installs runtime deps + the vendored kit
python run.py --check      # offline smoke: compiles the org (no key needed)
python run.py "your task"  # run one task against a real model
```

## Configure the model

Drop a `./.env` next to `run.py` (auto-loaded on start):

```
PUX_MODEL=__MODEL__
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
```

`PUX_MODEL` is any id your endpoint serves. `OPENAI_API_KEY` / `OPENAI_BASE_URL`
are read automatically by `ChatOpenAI`. The default (`__MODEL__`) is what this
org used in the harness — override freely.

## Layout

```
__ORG__/
  AGENTS.md             root base prompt
  orgs/__ORG__/...      the org (agents, skills, policy, sandbox)
  orgs/_shared/...      shared agents/skills/sandbox the org resolves
  pux_harness/          vendored slim kit (the portable compiler)
    kit/                compile_org, loaders, bootstrap, _testing, ...
  run.py                runner (entry point)
  pyproject.toml        runtime deps
  manifest.json         machine-readable inventory
```

## What this is NOT

No Docker sandbox, no server, no browser-vision/context middleware, no rubric
gate — the slim kit compiles a plain deepagents agent. For the full harness,
install `pux-harness` and use `compile_org` from there.
'''


def _collect_org_files(org_dir: Path) -> dict[str, Path]:
    """.. removed:: P3

    The hardcoded-dir allowlist (``("agents","skills","sandbox","config")`` +
    core files, ``data`` excluded by hand-comment) is GONE. Org-local
    collection is now **default-deny via the declarative manifest** — see
    :mod:`pux_harness.manifest` (:func:`~pux_harness.manifest.collect_pack_files`,
    driven by ``package.include``/``exclude`` globs in ``org.yaml``).

    This stub exists ONLY so a stale ``from pux_harness.pack import
    _collect_org_files`` raises a loud, explicit error instead of an opaque
    ``ImportError`` — the permanent contract form of "the old allowlist must
    not come back" (re-adding the real collector here trips
    ``tests/export/test_export.py::test_legacy_allowlist_collector_is_removed``).
    """
    raise NotImplementedError(
        "_collect_org_files was removed in P3 (manifest-driven default-deny "
        "pack). Use pux_harness.manifest.collect_pack_files via pack_org()."
    )


def _resolve_shared_agents(org: str) -> dict[str, Path]:
    """Shared agents the org needs (slugs from org.yaml that resolve to _shared)."""
    files: dict[str, Path] = {}
    shared_dir = _orgs_dir() / "_shared" / "agents"
    if not shared_dir.is_dir():
        return files

    org_local_agents = _orgs_dir() / org / "agents"
    if not org_local_agents.is_dir():
        org_local_agents = _orgs_dir() / "specialists" / org / "agents"

    for slug in org_agent_slugs(org):
        # If org-local exists, it was already captured by _collect_org_files
        if org_local_agents.is_dir() and (org_local_agents / f"{slug}.md").is_file():
            continue
        shared_path = shared_dir / f"{slug}.md"
        if shared_path.is_file():
            files[f"orgs/_shared/agents/{slug}.md"] = shared_path

    return files


def _resolve_shared_skills(
    org: str, project_root: Path | None = None,
) -> dict[str, Path]:
    """Shared skills referenced by agents' ``skills:`` frontmatter.

    ``project_root`` defaults to the kit's LIVE resolver (no import-time
    snapshot) — per-call overridable so ``pack_org`` can thread a tmp root."""
    root = project_root if project_root is not None else _default_project_root()
    files: dict[str, Path] = {}
    shared_skills = _orgs_dir() / "_shared" / "skills"
    if not shared_skills.is_dir():
        return files

    # Walk every agent's skills: field
    for slug in org_agent_slugs(org):
        spec = _load_agent_spec(slug, org)
        if spec is None:
            continue
        for raw in _parse_list(spec.get("skills", [])):
            # raw is project-relative like "orgs/_shared/skills"
            if not isinstance(raw, str) or "_shared" not in raw:
                continue
            skills_root = root / raw
            if not skills_root.is_dir():
                continue
            for skill_dir in sorted(skills_root.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if skill_md.is_file():
                    rel = skill_md.relative_to(root)
                    files[str(rel)] = skill_md
                # Include references/ and scripts/ subdirs
                for sub in ("references", "scripts"):
                    sub_dir = skill_dir / sub
                    if sub_dir.is_dir():
                        for f in sorted(sub_dir.rglob("*")):
                            if f.is_file():
                                rel = f.relative_to(root)
                                files[str(rel)] = f

    return files


def _resolve_shared_sandbox(
    org: str, project_root: Path | None = None,
) -> dict[str, Path]:
    """Shared sandbox helpers referenced by policy host_setup hooks.

    ``project_root`` defaults to the kit's LIVE resolver — per-call overridable."""
    root = project_root if project_root is not None else _default_project_root()
    files: dict[str, Path] = {}
    org_dir = _org_path(org)
    policy_path = org_dir / "policy.yaml"
    if not policy_path.is_file():
        return files

    try:
        policy = yaml.safe_load(policy_path.read_text()) or {}
    except yaml.YAMLError:
        return files

    if not isinstance(policy, dict):
        return files

    # ``host_setup`` hooks (``helper_script``) AND ``jobs`` (``script``) both
    # reference shared sandbox files. ``jobs`` is the newer mechanism
    # (sandbox/policy.py JobSpec) and was missed here: dev-bot/general's
    # ``warmup_browser.py`` silently dropped out of the archive, so the exported
    # org's pre-run job FileNotFound'd at serve time. Schema source-of-truth is
    # ``policy_mod.host_setup_hooks`` / ``job_specs`` (sandbox/policy.py); we
    # hand-parse here to stay as lenient as the ``yaml.safe_load`` above —
    # ``policy_mod.load()`` env-substitutes and would raise on an unset
    # ``${VAR}`` elsewhere in the policy (the tool_servers URLs), silently
    # dropping these files (a regression).
    hooks = policy.get("host_setup") or []
    if not isinstance(hooks, list):
        hooks = []
    jobs = policy.get("jobs") or []
    if not isinstance(jobs, list):
        jobs = []

    shared_sandbox = _orgs_dir() / "_shared" / "sandbox"
    shared_clients = _orgs_dir() / "_shared" / "clients"

    # Collect every shared file ref from both mechanisms, then resolve once.
    # Org-local scripts (no ``_shared``) are already captured by
    # _collect_org_files; only the shared ones need explicit bundling.
    file_refs: list[str] = []
    for hook in hooks:
        if isinstance(hook, dict):
            ref = hook.get("helper_script", "")
            if isinstance(ref, str) and ref:
                file_refs.append(ref)
    for job in jobs:
        if isinstance(job, dict):
            ref = job.get("script", "")
            if isinstance(ref, str) and ref:
                file_refs.append(ref)
    for script in file_refs:
        # project-relative, e.g. "orgs/_shared/sandbox/warmup_browser.py"
        if "_shared" not in script:
            continue
        script_path = root / script
        if script_path.is_file():
            files[str(Path(script).parent / script_path.name)] = script_path

    # Also include shared clients if any org-local sandbox script imports them
    org_sandbox = org_dir / "sandbox"
    if org_sandbox.is_dir() and shared_clients.is_dir():
        for py_file in sorted(org_sandbox.glob("*.py")):
            try:
                content = py_file.read_text()
            except Exception:
                continue
            if "from" in content and ("clients" in content or "_shared" in content):
                for client_file in sorted(shared_clients.glob("*.py")):
                    rel = client_file.relative_to(root)
                    files[str(rel)] = client_file
                break  # one check is enough to decide we need clients

    # Include shared sandbox helpers that org sandbox scripts might import
    if org_sandbox.is_dir() and shared_sandbox.is_dir():
        for py_file in sorted(org_sandbox.glob("*.py")):
            try:
                content = py_file.read_text()
            except Exception:
                continue
            if "from" in content and "sandbox" in content:
                for helper in sorted(shared_sandbox.glob("*.py")):
                    rel = helper.relative_to(root)
                    files[str(rel)] = helper
                break

    return files


def _resolve_tool_servers(org: str) -> dict[str, Path]:
    """Shared tool_servers.yaml if the org's policy declares tool_servers."""
    files: dict[str, Path] = {}
    org_dir = _org_path(org)
    policy_path = org_dir / "policy.yaml"
    if not policy_path.is_file():
        return files

    try:
        policy = yaml.safe_load(policy_path.read_text()) or {}
    except yaml.YAMLError:
        return files

    if not isinstance(policy, dict):
        return files

    tool_servers = policy.get("tool_servers")
    if not tool_servers:
        return files

    ts_path = _orgs_dir() / "_shared" / "tool_servers.yaml"
    if ts_path.is_file():
        files["orgs/_shared/tool_servers.yaml"] = ts_path

    return files


def _vendor_kit() -> dict[str, bytes]:
    """The runtime slim-kit files, vendored verbatim into the export.

    A consumer runs the org WITHOUT pux-harness installed: the kit
    (``pux_harness.kit``) is the portable compiler, so it travels in the archive.
    Only the runtime files are copied (the kit is import-self-contained — intra-
    kit + third-party only). Read from THIS package's ``kit/`` dir."""
    vendored: dict[str, bytes] = {}
    for name in _KIT_RUNTIME_FILES:
        vendored[f"pux_harness/kit/{name}"] = (_KIT_DIR / name).read_bytes()
    vendored["pux_harness/__init__.py"] = _VENDORED_PKG_INIT.encode("utf-8")
    return vendored


def _resolve_default_model(org: str) -> str:
    """The org's base-role model id, baked into the runner as the default.

    ``resolve_model_id`` is id-only (no key, no network), so this is safe at
    export time. Falls back to ``glm-5.2`` (the shipped default-tier base) if
    resolution is unavailable — the consumer overrides via ``PUX_MODEL``."""
    try:
        from pux_harness.agent.model import resolve_model_id  # noqa: PLC0415

        return resolve_model_id(role="base", org=org)
    except Exception:  # noqa: BLE001 — export must never die on model resolution
        return "glm-5.2"


def _build_runtime_scaffold(org: str) -> dict[str, bytes]:
    """Every generated file that turns the primitives archive into a RUNNABLE
    package: the vendored slim kit + ``pyproject.toml`` + ``run.py`` + README.

    Keys are archive-relative paths (under ``<org>/``). Values are the file
    bytes (synthetic — not read from a host Path, so they are written via the
    same BytesIO path as the manifest, not the host-file loop)."""
    model = _resolve_default_model(org)
    scaffold = _vendor_kit()
    scaffold["run.py"] = _RUN_PY.replace("__ORG__", org).replace("__MODEL__", model).encode("utf-8")
    scaffold["pyproject.toml"] = _PYPROJECT.replace("__ORG__", org).encode("utf-8")
    scaffold["README.md"] = _README.replace("__ORG__", org).replace("__MODEL__", model).encode("utf-8")
    return scaffold


def _build_manifest(
    org: str,
    files: dict[str, Path],
    scaffold: dict[str, bytes] | None = None,
    manifest: Any = None,
) -> dict[str, Any]:
    """Machine-readable inventory of the packed archive.

    ``files`` are the org primitives (host Path sources); ``scaffold`` are the
    generated runnable-package files (vendored kit + pyproject + run.py + README)
    — routed to their own ``runtime_scaffold`` category so the org inventory
    stays separable from the packaging machinery. ``manifest`` (the parsed
    :class:`~pux_harness.manifest.Manifest`) adds the declared audit surface
    (``manifest_version``/``package``/``capabilities``/``dependencies``) under a
    ``"manifest"`` key — what the org declared it ships/needs/can-do, NOT secrets."""
    org_slugs = org_agent_slugs(org)
    scaffold = scaffold or {}
    scaffold_keys = set(scaffold)

    # Categorize files
    categories: dict[str, list[str]] = {
        "root_prompt": [],
        "org_core": [],
        "org_agents": [],
        "org_skills": [],
        "org_sandbox": [],
        "org_other": [],
        "shared_agents": [],
        "shared_skills": [],
        "shared_sandbox": [],
        "shared_clients": [],
        "shared_tool_servers": [],
        "runtime_scaffold": [],
    }

    for rel in sorted(set(files) | scaffold_keys):
        if rel in scaffold_keys:
            categories["runtime_scaffold"].append(rel)
        elif rel == "AGENTS.md":
            categories["root_prompt"].append(rel)
        elif rel.startswith(f"orgs/{org}/agents/"):
            categories["org_agents"].append(rel)
        elif rel.startswith(f"orgs/{org}/skills/"):
            categories["org_skills"].append(rel)
        elif rel.startswith(f"orgs/{org}/sandbox/"):
            categories["org_sandbox"].append(rel)
        elif rel.startswith(f"orgs/{org}/"):
            categories["org_core"].append(rel)
        elif rel.startswith("orgs/_shared/agents/"):
            categories["shared_agents"].append(rel)
        elif rel.startswith("orgs/_shared/skills/"):
            categories["shared_skills"].append(rel)
        elif rel.startswith("orgs/_shared/sandbox/"):
            categories["shared_sandbox"].append(rel)
        elif rel.startswith("orgs/_shared/clients/"):
            categories["shared_clients"].append(rel)
        elif rel.startswith("orgs/_shared/tool_servers"):
            categories["shared_tool_servers"].append(rel)
        else:
            categories["org_other"].append(rel)

    return {
        "org": org,
        "description": f"Standalone runnable pack of the {org} org",
        "agent_roster": org_slugs,
        "total_files": len(files) + len(scaffold),
        "categories": categories,
        "files": sorted(set(files) | scaffold_keys),
        "manifest": manifest_metadata(manifest) if manifest else None,
    }


# Text files whose CONTENTS may carry stale ``orgs/specialists/`` path refs
# that ``_normalize_specialists_refs`` rewrites (mirrors the tree flattening).
_TEXT_SUFFIXES = (".md", ".yaml", ".yml", ".txt")


def _normalize_specialists_refs(data: bytes) -> bytes:
    """Mirror the tree flattening inside file CONTENTS.

    The archive flattens ``orgs/specialists/<name>/`` to ``orgs/<name>/`` so a
    consumer never reproduces the orchestrator's internal categorization. But
    agent frontmatter and ``policy.yaml`` reference sandbox scripts, skill
    roots, and Dockerfiles by their FULL host path
    (``orgs/specialists/<name>/...``). Copied verbatim, those references dangle
    in the unpacked tree and the org will not recompile — ``kit._resolve_skills``
    raises ``KeyError`` (observed: 7/10 shipped orgs broken on round-trip), and
    policy ``script:``/``dockerfile:`` paths miss at runtime.

    Rewrite ``orgs/specialists/`` -> ``orgs/`` so the archive is self-consistent:
    every content reference resolves at the same flattened path the tree uses.
    Applied to text files only; non-UTF-8 / binary bytes pass through unchanged.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    if "orgs/specialists/" not in text:
        return data
    return text.replace("orgs/specialists/", "orgs/").encode("utf-8")


def pack_org(
    org: str,
    output: Path | None = None,
    project_root: Path | None = None,
    hooks: list[PackHook] | None = None,
    gitleaks_runner: Callable | None = None,
) -> Path:
    """Pack an org as a self-contained ``.tar.gz`` archive (the ``pux pack`` op).

    ``project_root`` is AUTHORITATIVE: the pack resolves the org, its shared
    agents/skills/sandbox, and tool servers against THIS root — even when it is
    a foreign tree (a standalone consumer app that lives next to its own
    inference code, NOT the orchestrator). That is what makes a pack portable:
    ``pack_org`` packages an org from *wherever it lives* without needing a
    copy inside the orchestrator. It defaults to the kit's LIVE resolver
    (``$PUX_PROJECT_ROOT`` or the CWD), mirroring ``kit.compile_org``.

    Org-local collection is **manifest-driven + default-deny**
    (:func:`pux_harness.manifest.collect_pack_files` via ``package.include``
    globs in ``org.yaml``); shared deps are inclusion-by-reference (only what
    the org actually cites). ``data/``/``.pux/`` are never packed.

    Before the tarball is written, the collected files run through the
    **pack-time validation hooks** (``PACK_HOOK_REGISTRY`` — AST + gitleaks, P4):
    a syntax-broken agent function or a leaked secret REFUSES the pack
    (:class:`pux_harness.pack_hooks.PackHookError` — verify-or-die). Pass
    ``hooks=[]`` to skip validation (test/portability seam); ``gitleaks_runner``
    injects the subprocess runner for offline-deterministic tests.

    Implementation note: the downstream resolvers (``discover_orgs``,
    ``_org_path``, ``_resolve_shared_agents``, ``_resolve_tool_servers``) all
    funnel through ``kit._paths.project_root()`` ← ``$PUX_PROJECT_ROOT``. We
    pin that env var to ``project_root`` for the call's duration (restored in a
    ``finally``) so every resolver honors the passed root. Returns the path to
    the written archive.
    """
    root = (
        project_root if project_root is not None else _default_project_root()
    ).resolve()

    prior_env = os.environ.get(_PROJECT_ROOT_ENV)
    os.environ[_PROJECT_ROOT_ENV] = str(root)
    try:
        if org not in discover_orgs():
            raise FileNotFoundError(
                f"org {org!r} not found under {root}; "
                f"discovered: {discover_orgs()}")

        org_dir = _org_path(org)

        # Load the declarative manifest FIRST — it drives default-deny
        # collection of the org-local tree (package.include/exclude globs from
        # org.yaml; defaults when the org declares no ``package:`` block).
        manifest = load_manifest(org_dir, org)

        # 1. Root AGENTS.md (base prompt)
        root_agents_md = root / "AGENTS.md"
        files: dict[str, Path] = {}
        if root_agents_md.is_file():
            files["AGENTS.md"] = root_agents_md

        # 2. Org-local files — MANIFEST-DRIVEN, default-deny (was the
        # _collect_org_files allowlist). data/.pux pruned during the walk.
        files.update(collect_pack_files(org_dir, manifest))

        # 3. Shared dependencies
        files.update(_resolve_shared_agents(org))
        files.update(_resolve_shared_skills(org, root))
        files.update(_resolve_shared_sandbox(org, root))
        files.update(_resolve_tool_servers(org))

        # 4. Pack-time validation hooks (P4): AST + gitleaks gate the pack
        # BEFORE the tarball is written. A syntax-broken agent function or a
        # leaked secret REFUSES the pack (PackHookError; verify-or-die). The
        # hooks validate what the operator/agent AUTHORED (org + shared
        # primitives), not the trusted/generated scaffold. Results seed the
        # manifest's provenance block (the audit surface P5 extends).
        hook_results = run_pack_hooks(
            HookContext(org=org, org_dir=org_dir, files=files, manifest=manifest),
            registry=hooks,
            gitleaks_runner=gitleaks_runner,
        )

        # 5. Runtime scaffold (F3): vendor the slim kit + emit pyproject/run.py/
        # README so the archive is a standalone RUNNABLE package — a consumer
        # runs the org without pux-harness installed.
        scaffold = _build_runtime_scaffold(org)

        # 6. Build manifest inventory (accounts for primitives + the scaffold +
        # the declared audit surface from the parsed manifest) + hook provenance.
        inventory = _build_manifest(org, files, scaffold, manifest)
        inventory["provenance"] = provenance_from_results(hook_results)

        # 7. Write tar.gz
        if output is None:
            output = Path(f"{org}.tar.gz")

        with tarfile.open(output, "w:gz") as tar:
            # Add a top-level directory entry. The mode MUST carry the execute
            # bit (0o755): a DIRTYPE entry at the default 0o644 is created on
            # extract without traverse permission, so every child becomes an
            # EACCES on read — the archive looks fine to `tar tzf` but is
            # unusable when actually unpacked. Set it explicitly.
            info = tarfile.TarInfo(name=f"{org}/")
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            tar.addfile(info)

            for archive_path, host_path in sorted(files.items()):
                try:
                    data = host_path.read_bytes()
                except (PermissionError, OSError):
                    continue
                # Keep content refs consistent with the flattened tree (see
                # _normalize_specialists_refs). Size is computed AFTER rewrite.
                if archive_path.endswith(_TEXT_SUFFIXES):
                    data = _normalize_specialists_refs(data)
                info = tarfile.TarInfo(name=f"{org}/{archive_path}")
                info.size = len(data)
                tar.addfile(info, BytesIO(data))

            # Generated scaffold (vendored kit + pyproject + run.py + README) —
            # synthetic bytes (no host Path), written via the same BytesIO path.
            for archive_path, data in sorted(scaffold.items()):
                info = tarfile.TarInfo(name=f"{org}/{archive_path}")
                info.size = len(data)
                tar.addfile(info, BytesIO(data))

            # Add manifest
            manifest_json = json.dumps(inventory, indent=2).encode()
            info = tarfile.TarInfo(name=f"{org}/manifest.json")
            info.size = len(manifest_json)
            tar.addfile(info, BytesIO(manifest_json))

        return output
    finally:
        if prior_env is None:
            os.environ.pop(_PROJECT_ROOT_ENV, None)
        else:
            os.environ[_PROJECT_ROOT_ENV] = prior_env
