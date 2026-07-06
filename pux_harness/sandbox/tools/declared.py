"""Declared sandbox tools — typed, by-name tools that run IN-CONTAINER.

A per-org ``sandbox/tools/tools.yaml`` DECLARES tools (name + description +
script + typed args). This module synthesizes langchain ``StructuredTool``s
from those declarations: the model calls ``pux_sandbox_<name>(...)`` directly,
and the tool's ``func`` exec's the script IN-CONTAINER via ``DockerExecClient``
(typed ergonomics for the model; in-container execution preserved).

This is the declarative, per-org, no-library-edit equivalent of a REGISTRY
``ToolSpec``. A REGISTRY tool is a library edit that binds harness internals and
runs in-process on the HOST; a DECLARED tool is org-local data whose script
runs INSIDE the sandbox container. It fills the gap between REGISTRY (typed,
host) and raw sandbox scripts (in-container, unstructured shell) — see
``docs/capability-architecture.mdx`` §3e.

Why not the langchain ``@tool`` decorator: that decorates a HOST-side Python
function, which would run host-side — the wrong site. These scripts MUST run
in-container, so the tool's ``func`` exec's into the container (same mechanism
as ``pux_sandbox_python``).

Contract-enforced (``validate_declared_tools``) + auto-audited
(``AuditMiddleware.awrap_tool_call`` records every tool call by name, so
declared tools are audited for free).

Layering: this module stays agent-free (``sandbox/tools`` is a lower layer than
``agent/``). Callers in ``agent/stack.py`` + ``agent/contract.py`` resolve the
org's sandbox dir via ``_org_path`` and pass it in; the container path is
derived from ``project_root()`` + ``WORKSPACE_ROOT`` (both sandbox/kit-layer).
"""

from __future__ import annotations

import json
import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from pux_harness.kit._paths import project_root
from pux_harness.sandbox.backend import WORKSPACE_ROOT
from pux_harness.sandbox.tools._shared import PUX_PREFIX, _result, _tail
from pux_harness.sandbox.tools.registry import (
    LEGACY_TOOL_NAMES,
    NATIVE_FS_TOOLS,
    SPECIALIST_TOOL_NAMES,
)

log = logging.getLogger(__name__)

_TOOLS_SUBPATH = Path("tools") / "tools.yaml"

# type vocab — yaml ``type:`` string -> Python type for the pydantic field.
_PY_TYPE: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

_INVOKE_STYLES = frozenset({"flags", "positional"})
_RETURN_STYLES = frozenset({"text", "json"})

# langchain reserves these arg names (injected as tool-call context); a declared
# tool name re-using them would collide. Bare-name validity, not arg validity.
_RESERVED_NAMES = frozenset({"config", "runtime"})

# snake_case + a leading letter — langchain requires snake_case tool names and
# rejects uppercase. Mirrors the skill-name regex shape used elsewhere.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# --- data model ------------------------------------------------------------

@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: str
    required: bool
    default: Any
    description: str


@dataclass(frozen=True)
class DeclaredToolSpec:
    name: str
    description: str
    script: str
    subcommand: Optional[str]
    invoke: str            # "flags" | "positional"
    timeout: Optional[int]
    returns: str           # "text" | "json"
    args: tuple[ArgSpec, ...]


# --- loading (pure data) ---------------------------------------------------

def load_declared_specs(org_sandbox_dir: Path) -> list[DeclaredToolSpec]:
    """Parse ``<org_sandbox_dir>/tools/tools.yaml`` -> specs.

    Returns ``[]`` when the file is absent (every org without a declaration is
    unaffected — zero behavior change). ``org_sandbox_dir`` is the org's
    ``sandbox/`` directory (e.g. ``…/orgs/specialists/invest/sandbox``); the
    caller (agent layer) resolves it via ``_org_path(org) / "sandbox"``.
    """
    tools_yaml = org_sandbox_dir / _TOOLS_SUBPATH
    if not tools_yaml.is_file():
        return []
    data = yaml.safe_load(tools_yaml.read_text()) or {}
    raw_tools = data.get("tools", []) if isinstance(data, dict) else []
    if not isinstance(raw_tools, list):
        raise ValueError(f"{tools_yaml}: top-level 'tools' must be a list")
    return [_parse_spec(raw, tools_yaml) for raw in raw_tools]


def _parse_spec(raw: Any, src: Path) -> DeclaredToolSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"{src}: each tool entry must be a mapping, got {type(raw).__name__}")
    try:
        name = raw["name"]
        script = raw["script"]
    except KeyError as exc:
        raise ValueError(f"{src}: declared tool missing required key {exc.args[0]!r}") from exc
    raw_args = raw.get("args") or []
    if not isinstance(raw_args, list):
        raise ValueError(f"{src}: tool {name!r} 'args' must be a list")
    args = tuple(
        ArgSpec(
            name=a["name"],
            type=a.get("type", "string"),
            required=bool(a.get("required", False)),
            default=a.get("default"),
            description=a.get("description", ""),
        )
        for a in raw_args
    )
    return DeclaredToolSpec(
        name=name,
        description=raw.get("description", ""),
        script=script,
        subcommand=raw.get("subcommand"),
        invoke=raw.get("invoke", "flags"),
        timeout=raw.get("timeout"),
        returns=raw.get("returns", "text"),
        args=args,
    )


def declared_tool_names(org_sandbox_dir: Path) -> frozenset[str]:
    """Bare declared tool names for ``org_sandbox_dir`` (empty if none).

    Used by the contract (rule 4) so a declared name in an agent ``tools:``
    allowlist is not false-flagged as unknown — the offline check and the
    runtime ``_resolve_tools`` must agree."""
    return frozenset(s.name for s in load_declared_specs(org_sandbox_dir))


# --- building (synthesize StructuredTools) ---------------------------------

def _container_dir(org_sandbox_dir: Path) -> Path:
    """The org sandbox dir as seen IN-CONTAINER.

    The project is bind-mounted 1:1 at ``WORKSPACE_ROOT`` (``/sandbox/workspace``),
    so the container path is ``WORKSPACE_ROOT / <org_sandbox_dir relative to the
    project root>`` — the same mapping ``orgs._resolve_skills`` uses for skills
    roots. Raises ``ValueError`` if the org is not under the project root (the
    container only mounts the project; a non-project org can't reach its scripts)."""
    rel = org_sandbox_dir.relative_to(project_root())
    return Path(WORKSPACE_ROOT) / rel


def _build_args_model(spec: DeclaredToolSpec) -> type:
    """A pydantic model for the tool's args, built dynamically via ``create_model``.

    Deterministic + matches the ``python.py`` BaseModel pattern. Required args
    are non-Optional (``Field(...)``); optional args are ``type | None`` with
    their declared default (or ``None``), so the model may omit them and the
    runner skips ``None`` values when serializing the command."""
    fields: dict[str, Any] = {}
    for arg in spec.args:
        py_type = _PY_TYPE[arg.type]
        if arg.required:
            fields[arg.name] = (py_type, Field(..., description=arg.description))
        else:
            fields[arg.name] = (py_type | None, Field(arg.default, description=arg.description))
    # create_model wants a valid identifier; spec.name is snake_case (validated).
    return create_model(f"_DeclaredArgs_{spec.name}", **fields)


def _stringify(value: Any) -> str:
    return "true" if value is True else "false" if value is False else str(value)


def _build_command(spec: DeclaredToolSpec, container_dir: Path, kwargs: dict[str, Any]) -> str:
    """Serialize the typed kwargs into a ``bash -c`` command that runs the script
    in-container. ``cd`` into the script's dir first so sibling-module/config
    reads (the invest-pipeline shape) resolve naturally."""
    parts: list[str] = ["cd", shlex.quote(str(container_dir)), "&&", "python3", shlex.quote(spec.script)]
    if spec.subcommand:
        parts.append(shlex.quote(spec.subcommand))
    for arg in spec.args:
        val = kwargs.get(arg.name)
        if val is None:
            continue
        if spec.invoke == "positional":
            parts.append(shlex.quote(_stringify(val)))
        else:  # flags (default)
            if arg.type == "boolean":
                if val:
                    parts.append(f"--{arg.name}")  # store_true convention
                # False -> omit
            else:
                parts.append(f"--{arg.name}")
                parts.append(shlex.quote(_stringify(val)))
    return " ".join(parts)


def _make_runner(spec: DeclaredToolSpec, exec_client: Any, container_dir: Path):
    def _run(**kwargs: Any) -> str:
        cmd = _build_command(spec, container_dir, kwargs)
        out, exit_code = exec_client.exec(cmd, timeout=spec.timeout)
        if exit_code != 0:
            return _result({
                "success": False,
                "error": f"script exited {exit_code}",
                "command": cmd,
                "output": _tail(out),
            })
        if spec.returns == "json":
            try:
                parsed = json.loads(out)
            except json.JSONDecodeError as exc:
                return _result({
                    "success": False,
                    "error": f"script output was not valid JSON: {exc}",
                    "command": cmd,
                    "output": _tail(out),
                })
            return _result({"success": True, "command": cmd, "json": parsed, "raw": _tail(out)})
        return _result({"success": True, "command": cmd, "output": _tail(out)})

    return _run


def build_declared_tools(org_sandbox_dir: Path, exec_client: Any) -> list[StructuredTool]:
    """Synthesize a ``StructuredTool`` per declared tool. Empty list when the org
    declares none (byte-identical stack for orgs without ``tools.yaml``)."""
    specs = load_declared_specs(org_sandbox_dir)
    if not specs:
        return []
    container_dir = _container_dir(org_sandbox_dir)
    tools: list[StructuredTool] = []
    for spec in specs:
        tools.append(StructuredTool(
            name=PUX_PREFIX + spec.name,
            description=spec.description or f"Run the sandbox script {spec.script!r}.",
            args_schema=_build_args_model(spec),
            func=_make_runner(spec, exec_client, container_dir),
        ))
    return tools


# --- exec-guard redirect map (declared ⇒ taken out of `execute`) -----------
# The invariant this section serves: once a script is exposed as a typed
# ``pux_sandbox_<name>`` tool, the agent must reach it ONLY through that tool —
# NOT via a raw ``execute("python3 <script> <subcommand> …")`` shell. The typed
# path is schema-validated, runs in-container, and is audited; the raw path is
# none of those and taxes the weak default model (mimo-v2.5) on arg-quoting /
# CLI-recall. ``build_script_redirects`` produces the (pattern, target) pairs
# ``RoutingMiddleware`` matches intercepted ``execute``/bash commands against —
# a match returns a redirect ``ToolMessage`` naming the typed tool WITHOUT
# running the command. The declared tool's OWN ``func`` calls
# ``exec_client.exec(cmd)`` directly (not a tool call), so it is never
# intercepted: agent-via-``execute`` → blocked; declared-tool-internal-exec →
# runs in-container. That distinction is the seam, and it is free at the
# middleware layer (verified by reading ``_make_runner``).
#
# Pure: returns plain ``(re.Pattern, str)`` tuples so ``RoutingMiddleware``
# consumes it with NO import of this module — the agent layer wires the two.


def build_script_redirects(
    specs: "list[DeclaredToolSpec]",
) -> "list[tuple[re.Pattern[str], str]]":
    """Compile each declared spec into a ``(pattern, target_tool)`` redirect.

    Per-``(script, subcommand)`` matching — the correctness subtlety: block
    ONLY the invocation a declared tool exposes, not the whole script.
    ``scan_signals`` wraps ``signals.py score``, so block
    ``python3 … signals.py score`` but leave ``signals.py rank`` / ``validate``
    (un-exposed subcommands) exec-able — the agent has no typed alternative for
    those. A spec with no ``subcommand`` blocks any ``python3 … <script>``
    invocation (the whole script is the tool's surface).

    Pattern shape: ``\\bpython3?\\b[^\\n]*\\b<script>\\b(\\s+<subcommand>\\b)?``.
    The ``python3?`` anchor keeps this about SCRIPT invocation (not a doc
    mention of the bare filename); ``[^\\n]*`` tolerates a ``cd … && python3 …``
    prefix and an absolute/relative path before the basename; ``\\b`` word
    boundaries stop ``signals.py`` matching ``my_signals.py`` and stop
    ``score`` matching ``scoreboard``.

    Returns ``[]`` for an empty spec list (orgs that declare nothing redirect
    nothing — byte-identical routing behavior). Pure: no routing import, no I/O.
    """
    redirects: list[tuple[re.Pattern[str], str]] = []
    for spec in specs:
        script = re.escape(spec.script)
        if spec.subcommand:
            sub = re.escape(spec.subcommand)
            pattern = re.compile(rf"\bpython3?\b[^\n]*\b{script}\b\s+{sub}\b")
        else:
            pattern = re.compile(rf"\bpython3?\b[^\n]*\b{script}\b")
        redirects.append((pattern, PUX_PREFIX + spec.name))
    return redirects


# --- validation (offline contract body; no exec_client) --------------------

def _name_taken(full_name: str) -> bool:
    """Does ``full_name`` (a ``pux_sandbox_*`` name) collide with the REGISTRY
    surface or the legacy denylist? Declared tools share the ``pux_sandbox_``
    prefix, so a collision would shadow a real specialist or revive a legacy name."""
    return full_name in SPECIALIST_TOOL_NAMES or full_name in LEGACY_TOOL_NAMES


def validate_declared_tools(org_sandbox_dir: Path) -> list[str]:
    """Offline validation of an org's declared tools. Returns a list of
    human-readable error strings (empty = clean). No exec_client, no container.

    Checks: yaml parses + schema; each name is snake_case, not reserved
    (``config``/``runtime``), not a native/grader slug, not ``mcp__``-prefixed,
    and does not collide with the REGISTRY/legacy ``pux_sandbox_*`` surface or a
    sibling in the same file; each script file exists on disk relative to the org
    sandbox dir; arg ``type`` is in the vocab; ``invoke``/``returns`` in enums."""
    tools_yaml = org_sandbox_dir / _TOOLS_SUBPATH
    if not tools_yaml.is_file():
        return []
    errors: list[str] = []
    org_name = org_sandbox_dir.parent.name  # the sandbox dir's parent is the org
    try:
        specs = load_declared_specs(org_sandbox_dir)
    except ValueError as exc:
        return [f"{org_name}: tools.yaml: {exc}"]

    seen: set[str] = set()
    for spec in specs:
        tag = f"{org_name}: tool {spec.name!r}"
        if not _NAME_RE.match(spec.name):
            errors.append(f"{tag}: name must be snake_case (lowercase letters/digits/underscores, leading letter)")
        if spec.name in _RESERVED_NAMES:
            errors.append(f"{tag}: name {spec.name!r} is reserved by langchain (config/runtime)")
        if spec.name in NATIVE_FS_TOOLS:
            errors.append(f"{tag}: name {spec.name!r} shadows a native fs/shell tool")
        if spec.name in {"execute", "read_file", "grep"}:  # grader bare slugs
            errors.append(f"{tag}: name {spec.name!r} shadows a grader slug")
        if spec.name.startswith("mcp__"):
            errors.append(f"{tag}: name must not use the reserved 'mcp__' namespace")
        full = PUX_PREFIX + spec.name
        if _name_taken(full):
            errors.append(f"{tag}: surfaces as {full!r} which collides with a REGISTRY/legacy tool")
        if spec.name in seen:
            errors.append(f"{tag}: duplicate declared name in this tools.yaml")
        seen.add(spec.name)

        if not spec.script:
            errors.append(f"{tag}: missing required 'script'")
        else:
            script_path = org_sandbox_dir / spec.script
            if not script_path.is_file():
                errors.append(f"{tag}: script {spec.script!r} not found at {script_path}")
        if spec.invoke not in _INVOKE_STYLES:
            errors.append(f"{tag}: invoke must be one of {sorted(_INVOKE_STYLES)}, got {spec.invoke!r}")
        if spec.returns not in _RETURN_STYLES:
            errors.append(f"{tag}: returns must be one of {sorted(_RETURN_STYLES)}, got {spec.returns!r}")
        if spec.timeout is not None and (not isinstance(spec.timeout, int) or spec.timeout <= 0):
            errors.append(f"{tag}: timeout must be a positive integer, got {spec.timeout!r}")

        arg_names: set[str] = set()
        for arg in spec.args:
            atag = f"{tag} arg {arg.name!r}"
            if not _NAME_RE.match(arg.name):
                errors.append(f"{atag}: name must be snake_case")
            if arg.name in arg_names:
                errors.append(f"{atag}: duplicate arg name")
            arg_names.add(arg.name)
            if arg.type not in _PY_TYPE:
                errors.append(f"{atag}: type must be one of {sorted(_PY_TYPE)}, got {arg.type!r}")
            if arg.required and arg.default is not None:
                errors.append(f"{atag}: 'required' tools may not also declare a default")
    return errors
