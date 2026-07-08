"""Dynamic (level c) sandbox tools — agent-authored, persistent, in-container.

The THIRD tool rung. Where REGISTRY tools (a) are fixed + host-side and DECLARED
tools (b) are operator-authored ``sandbox/tools/tools.yaml``, level (c) is the
run the *agent authors at runtime*: it writes Python under
``orgs/<org>/lib/functions/`` and calls it back later. It is the only rung that
*compounds* — after the org "learns" a function, later turns see only a bounded
result + an index entry, never the function body — the real lever for a
low-power model. See ``docs/dynamic-tools-and-packaging.md`` Part 1.

Rhymes with ``declared.py``: synthesizes langchain ``StructuredTool``s whose
``func`` exec's IN-CONTAINER via the same ``DockerExecClient``. The split that
matters:

* **Authoring is HOST-side** (``pathlib`` writes to
  ``orgs/<org>/lib/functions/<name>.py`` + ``lib/index.yaml``). The project is
  bind-mounted 1:1 at ``/sandbox/workspace`` (``container.py:321``), so a
  host-written file is immediately container-visible AND host-durable —
  persistence is free, and the file is owned by the host uid (NOT root). An
  in-container write would be root-owned on the host (the trap
  ``_scaffold_workspace`` chowns ``memos``/``.pux`` to fix — avoided here by
  writing from the pux process).
* **Execution is IN-CONTAINER**: ``call_function`` exec's
  ``cd <lib_container_dir> && _PUX_DYN_KWARGS=... python3 -c "<runner>"``. The
  runner ``from functions.<name> import run`` then ``run(**kwargs)``; for
  ``python3 -c`` cwd is ``sys.path[0]``, so ``functions.X`` resolves with
  ``lib/functions/__init__.py`` present (created on first use). kwargs ride an
  env var (not shell args) so arbitrary values need no quoting.

Opt-in: ``build_dynamic_tools`` is only called when the org's
``sandbox.dynamic_tools: true`` (``agent.profile.load_dynamic_tools_enabled``,
wired in ``stack.build_stack``). An org without the flag has a byte-identical
stack — zero behavior change.

Not ``.py``-as-agent: the library is tooling the agent EXTENDS; the deepagents
loop is unchanged. Distinct from skills (operator markdown backbone, host) and
from declared tools (operator-authored, in-container): lib is AGENT-authored
executable code.

P1 scope (this module): make / edit / list / call. Pruning (``.archive``) +
graduation (``pux promote-function`` lib→sandbox) are P2; the manifest + pack
hooks that scrub ``lib/functions/*.py`` are P3/P4.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from pux_harness.kit._paths import project_root
from pux_harness.sandbox.backend import WORKSPACE_ROOT
from pux_harness.sandbox.tools._shared import _NoArgs, _result, _tail

log = logging.getLogger(__name__)

# Distinct from ``pux_sandbox_`` (declared/specialist) so the contract's
# collision + admission checks do not conflate the two rungs.
PUX_DYN_PREFIX = "pux_dyn_"

# The four dynamic tools are FIXED (always exactly these) — unlike declared
# tools (operator-chosen names), the dynamic surface is a constant vocabulary.
# Exported so ``contract.py`` can admit them in an agent ``tools:`` allowlist.
DYNAMIC_TOOL_NAMES = frozenset({"make_function", "edit_function", "list_functions", "call_function"})

# Subpaths RELATIVE TO ``org_lib_dir`` (which is already ``orgs/<org>/lib`` —
# the caller in ``stack.build_stack`` resolves it via ``_org_path(org)/"lib"``).
# So functions live at ``<org_lib_dir>/functions`` and the index at its root.
_FUNCTIONS_SUBPATH = Path("functions")
_INDEX_SUBPATH = Path("index.yaml")

# snake_case + leading letter — mirrors ``declared._NAME_RE`` (langchain needs
# snake_case; rejects uppercase). Module filenames derive from this, so it must
# also be filesystem-safe on every host OS.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Bound a runaway dynamic call. Declared tools carry their own per-spec timeout;
# dynamic functions don't declare one, so a single sane default applies.
DEFAULT_DYN_TIMEOUT = 120

# stdout sentinel the runner prints before its JSON envelope, so call_function
# can recover the result even if the function prints debug noise of its own.
_RESULT_MARKER = "===PUX_DYN_RESULT==="

# The in-container runner. Written as a template with a single ``__NAME__``
# placeholder (NOT an f-string) so the literal braces/dicts in the runner source
# don't need doubling. ``\\n`` stays a literal backslash-n in this string, which
# ``python3 -c`` then reads as a newline — the marker sits on its own line and
# the JSON envelope follows it, so ``_extract_result`` can recover the value
# even when the function prints its own debug noise.
_RUNNER_TEMPLATE = '''import json as _json, os as _os, sys as _sys
try:
    from functions.__NAME__ import run as _run
except Exception as _exc:
    _sys.stdout.write(_json.dumps({"ok": False, "error": "import failed: " + repr(_exc)}))
    _sys.exit(1)
try:
    _kw = _json.loads(_os.environ["_PUX_DYN_KWARGS"])
    _out = _run(**_kw)
except Exception as _exc:
    _sys.stdout.write(_json.dumps({"ok": False, "error": "run() raised: " + repr(_exc)}))
    _sys.exit(1)
_sys.stdout.write("\\n''' + _RESULT_MARKER + '''\\n")
_sys.stdout.write(_json.dumps({"ok": True, "value": _out}, default=str))
'''


# --- paths (host + container) ----------------------------------------------

def _functions_dir(org_lib_dir: Path) -> Path:
    return org_lib_dir / _FUNCTIONS_SUBPATH


def _index_path(org_lib_dir: Path) -> Path:
    return org_lib_dir / _INDEX_SUBPATH


def _container_lib_dir(org_lib_dir: Path) -> Path:
    """The org ``lib`` dir as seen IN-CONTAINER.

    Same mapping as ``declared._container_dir``: the project is bind-mounted
    1:1 at ``WORKSPACE_ROOT`` (``/sandbox/workspace``), so the container path is
    ``WORKSPACE_ROOT / <org_lib_dir relative to the project root>``. Pure path
    math — works whether or not the dir exists yet (it is created on first
    ``make_function``)."""
    rel = org_lib_dir.relative_to(project_root())
    return Path(WORKSPACE_ROOT) / rel


# --- index.yaml I/O (source of truth — travels; Store does not) -------------

def load_dynamic_index(org_lib_dir: Path) -> dict[str, dict[str, Any]]:
    """Read ``lib/index.yaml`` -> ``{name: {description, created, usage, ...}}``.

    Returns ``{}`` when the file is absent: an org that has authored nothing
    has an empty library (byte-identical to "no dynamic tools yet"). The index
    is the bookkeeping source of truth — ``list_functions`` reads ONLY this
    (never code), which is what keeps listing cheap + bounded.
    """
    idx = _index_path(org_lib_dir)
    if not idx.is_file():
        return {}
    data = yaml.safe_load(idx.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{idx}: index.yaml must be a mapping at the top level")
    funcs = data.get("functions")
    if funcs is None:
        return {}
    if not isinstance(funcs, dict):
        raise ValueError(f"{idx}: top-level 'functions' must be a mapping")
    return funcs


def save_dynamic_index(org_lib_dir: Path, funcs: dict[str, dict[str, Any]]) -> None:
    """Write ``lib/index.yaml`` (names sorted for stable, reviewable diffs)."""
    idx = _index_path(org_lib_dir)
    idx.parent.mkdir(parents=True, exist_ok=True)
    body = {"functions": dict(sorted(funcs.items()))}
    idx.write_text(yaml.safe_dump(body, sort_keys=False, default_flow_style=False))


def _ensure_lib_skeleton(org_lib_dir: Path) -> None:
    """Create ``lib/functions/`` + its ``__init__.py`` on first use. Idempotent.

    ``__init__.py`` makes ``functions`` an importable package so the runner's
    ``from functions.<name> import run`` resolves against the lib cwd."""
    fns = _functions_dir(org_lib_dir)
    fns.mkdir(parents=True, exist_ok=True)
    init = fns / "__init__.py"
    if not init.exists():
        init.write_text(
            '"""Agent-authored dynamic functions (level c).\n\n'
            "Auto-created by pux_dyn_make_function on first use. Each module\n"
            "exposes ``run(**kwargs)``; call_function exec's\n"
            '``from functions.<name> import run`` in-container.\n"""\n'
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- source validation (footgun prevention; distinct from P4 pack hooks) ----

def _validate_source(name: str, code: str) -> str | None:
    """Reject code that won't run before we write it. Returns an error string
    or ``None`` (clean).

    Two checks: (1) it parses (``SyntaxError`` → clear message + line — the
    weak model can fix + retry); (2) it defines a TOP-LEVEL ``def run`` — the
    entrypoint ``call_function`` invokes. This is the authoring-site guard that
    keeps a broken function from ever reaching ``call_function``; it is NOT the
    pack-time secrets/AST scan (that's P4's gitleaks/ruff hooks over the whole
    ``lib/``)."""
    try:
        tree = ast.parse(code, filename=f"<dynamic:{name}>")
    except SyntaxError as exc:
        loc = f" (line {exc.lineno})" if exc.lineno else ""
        return f"Python syntax error: {exc.msg}{loc}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body):
        return (
            "code must define a top-level `def run(**kwargs)` — that is the "
            "entrypoint call_function invokes. (define `run` at module top "
            "level, not nested inside another function or behind an `if`.)"
        )
    return None


def _extract_result(out: str) -> dict[str, Any] | None:
    """Recover the runner's JSON envelope from captured stdout.

    Robust to a function that prints its own debug output: everything after the
    ``_RESULT_MARKER`` line is the envelope. ``None`` if no marker (the function
    crashed before printing one) or the payload is not valid JSON."""
    if _RESULT_MARKER not in out:
        return None
    tail = out.split(_RESULT_MARKER, 1)[1].strip()
    try:
        parsed = json.loads(tail)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _bump_call(org_lib_dir: Path, name: str, success: bool) -> None:
    """Increment ``usage`` (and ``success`` on a clean run) in ``index.yaml``.

    Defensive: a missing entry is left alone (the pre-call existence check
    should make this unreachable, but a concurrent edit_function must never
    crash the bookkeeping)."""
    funcs = load_dynamic_index(org_lib_dir)
    entry = funcs.get(name)
    if entry is None:
        return
    entry["usage"] = int(entry.get("usage", 0)) + 1
    if success:
        entry["success"] = int(entry.get("success", 0)) + 1
    save_dynamic_index(org_lib_dir, funcs)


# --- args schemas (pydantic via create_model — deterministic, mirrors declared) --

def _make_args_model() -> type:
    return create_model(
        "_DynMakeArgs",
        name=(
            str,
            Field(
                ...,
                description=(
                    "snake_case function name: lowercase letters, digits, "
                    "underscores, leading letter (e.g. compute_rsi)."
                ),
            ),
        ),
        description=(
            str,
            Field(..., description="One-line summary of what the function does; shown by list_functions."),
        ),
        code=(
            str,
            Field(
                ...,
                description=(
                    "Full Python module source. MUST define a top-level "
                    "`def run(**kwargs)` returning a JSON-serializable value. "
                    "Import stdlib + any sandbox-installed packages. run() is "
                    "the only thing call_function invokes."
                ),
            ),
        ),
    )


def _edit_args_model() -> type:
    return create_model(
        "_DynEditArgs",
        name=(str, Field(..., description="Name of an EXISTING function to replace.")),
        code=(
            str,
            Field(
                ...,
                description=(
                    "Full replacement module source. Same contract as "
                    "make_function (top-level `def run(**kwargs)`)."
                ),
            ),
        ),
    )


def _call_args_model() -> type:
    return create_model(
        "_DynCallArgs",
        name=(str, Field(..., description="Name of an EXISTING function to run.")),
        arguments=(
            dict,
            Field(
                default_factory=dict,
                description=(
                    "Keyword arguments for the function's run(), as a JSON "
                    "object (e.g. {\"period\": 14, \"prices\": [1,2,3]}). "
                    "Omit for a function that takes none."
                ),
            ),
        ),
    )


# --- tool closures (each closes over org_lib_dir [+ exec_client]) -----------

def _make_make(org_lib_dir: Path, label: str):
    def _run(name: str, description: str, code: str) -> str:
        if not _NAME_RE.match(name):
            return _result({
                "success": False,
                "error": (
                    f"name {name!r} must be snake_case (lowercase letters/"
                    f"digits/underscores, leading letter)"
                ),
            })
        _ensure_lib_skeleton(org_lib_dir)
        funcs = load_dynamic_index(org_lib_dir)
        if name in funcs:
            return _result({
                "success": False,
                "error": (
                    f"function {name!r} already exists; use pux_dyn_edit_function "
                    f"to change it, or pick a new name."
                ),
            })
        err = _validate_source(name, code)
        if err:
            return _result({"success": False, "error": err})
        target = _functions_dir(org_lib_dir) / f"{name}.py"
        target.write_text(code)
        funcs[name] = {
            "description": description,
            "created": _now_iso(),
            "usage": 0,
            "success": 0,
            "version": 1,
        }
        save_dynamic_index(org_lib_dir, funcs)
        log.info("dynamic[%s]: created function %r", label, name)
        return _result({
            "success": True,
            "name": name,
            "path": str(target.relative_to(project_root())),
        })

    return _run


def _make_edit(org_lib_dir: Path, label: str):
    def _run(name: str, code: str) -> str:
        funcs = load_dynamic_index(org_lib_dir)
        if name not in funcs:
            return _result({
                "success": False,
                "error": (
                    f"function {name!r} does not exist; use "
                    f"pux_dyn_make_function to create it."
                ),
            })
        err = _validate_source(name, code)
        if err:
            return _result({"success": False, "error": err})
        target = _functions_dir(org_lib_dir) / f"{name}.py"
        target.write_text(code)
        entry = funcs[name]
        entry["version"] = int(entry.get("version", 1)) + 1
        entry["edited"] = _now_iso()
        save_dynamic_index(org_lib_dir, funcs)
        log.info("dynamic[%s]: edited function %r -> v%s", label, name, entry["version"])
        return _result({"success": True, "name": name, "version": entry["version"]})

    return _run


def _make_list(org_lib_dir: Path):
    def _run(**kwargs: Any) -> str:
        funcs = load_dynamic_index(org_lib_dir)
        if not funcs:
            return _result({
                "success": True,
                "count": 0,
                "functions": [],
                "note": "no functions yet; use pux_dyn_make_function to create one.",
            })
        # Bounded by index size, NEVER code — the cheap listing that lets the
        # model recall what exists without re-loading bodies.
        summary = [
            {
                "name": n,
                "description": e.get("description", ""),
                "usage": e.get("usage", 0),
                "success": e.get("success", 0),
                "version": e.get("version", 1),
            }
            for n, e in sorted(funcs.items())
        ]
        return _result({"success": True, "count": len(summary), "functions": summary})

    return _run


def _make_call(org_lib_dir: Path, exec_client: Any, label: str):
    def _run(name: str, arguments: dict | None = None) -> str:
        funcs = load_dynamic_index(org_lib_dir)
        if name not in funcs:
            return _result({
                "success": False,
                "error": (
                    f"function {name!r} does not exist; use "
                    f"pux_dyn_list_functions to see available functions."
                ),
            })
        kwargs = arguments or {}
        if not isinstance(kwargs, dict):
            return _result({
                "success": False,
                "error": f"arguments must be a JSON object, got {type(kwargs).__name__}",
            })
        try:
            kwargs_json = json.dumps(kwargs)
        except (TypeError, ValueError) as exc:
            return _result({"success": False, "error": f"arguments are not JSON-serializable: {exc}"})

        runner = _RUNNER_TEMPLATE.replace("__NAME__", name)
        container_dir = _container_lib_dir(org_lib_dir)
        # ``PYTHONDONTWRITEBYTECODE=1`` keeps the container from leaving a
        # root-owned ``__pycache__`` on the (host-visible) lib dir.
        cmd = (
            f"cd {shlex.quote(str(container_dir))} && "
            f"_PUX_DYN_KWARGS={shlex.quote(kwargs_json)} "
            f"PYTHONDONTWRITEBYTECODE=1 "
            f"python3 -c {shlex.quote(runner)}"
        )
        try:
            out, exit_code = exec_client.exec(cmd, timeout=DEFAULT_DYN_TIMEOUT)
        except Exception as exc:  # exec infra failure, not the function's fault
            _bump_call(org_lib_dir, name, success=False)
            return _result({"success": False, "error": f"exec failed: {exc}", "command": cmd})

        if exit_code != 0:
            _bump_call(org_lib_dir, name, success=False)
            return _result({
                "success": False,
                "error": f"function exited with code {exit_code}",
                "command": cmd,
                "output": _tail(out),
            })
        payload = _extract_result(out)
        if payload is None:
            _bump_call(org_lib_dir, name, success=False)
            return _result({
                "success": False,
                "error": (
                    "function did not emit a result marker (it crashed or its "
                    "run() returned before printing) — see output tail"
                ),
                "output": _tail(out),
            })
        if not payload.get("ok"):
            _bump_call(org_lib_dir, name, success=False)
            return _result({
                "success": False,
                "error": payload.get("error", "unknown run() failure"),
                "output": _tail(out),
            })
        _bump_call(org_lib_dir, name, success=True)
        # THE THESIS: only the bounded ``value`` returns to the model — the
        # function body (however large) never enters context.
        return _result({"success": True, "value": payload.get("value")})

    return _run


# --- public builder ---------------------------------------------------------

def build_dynamic_tools(org_lib_dir: Path, exec_client: Any) -> list[StructuredTool]:
    """Synthesize the four ``pux_dyn_*`` StructuredTools for an org's ``lib/``.

    ``org_lib_dir`` is the host path to ``orgs/<org>/lib`` (resolved by the
    caller in ``stack.build_stack`` via ``_org_path(org) / "lib"``; this module
    stays agent-free, mirroring ``declared.py``). Empty list is NOT returned
    here — the caller gates the whole call on ``load_dynamic_tools_enabled``; if
    you reach this builder, all four tools ship.
    """
    label = org_lib_dir.parent.name  # the org dir name, for log/error context
    return [
        StructuredTool(
            name=PUX_DYN_PREFIX + "make_function",
            description=(
                "Create a NEW reusable Python function you can call back later. "
                "Use this to capture a procedure you will repeat, so you do not "
                "re-derive it each turn. `code` must be a complete Python module "
                "defining `def run(**kwargs)` that returns a JSON-serializable "
                "value. Refuses if the name already exists (use edit_function "
                "to change an existing one)."
            ),
            args_schema=_make_args_model(),
            func=_make_make(org_lib_dir, label),
        ),
        StructuredTool(
            name=PUX_DYN_PREFIX + "edit_function",
            description=(
                "Replace the source of an EXISTING function (one you created with "
                "make_function). Same code contract: a top-level "
                "`def run(**kwargs)`. Bumps the function's version."
            ),
            args_schema=_edit_args_model(),
            func=_make_edit(org_lib_dir, label),
        ),
        StructuredTool(
            name=PUX_DYN_PREFIX + "list_functions",
            description=(
                "List the reusable functions you have created, with their "
                "descriptions and call counts. Cheap — does NOT load code. "
                "Call this before call_function to recall what is available."
            ),
            args_schema=_NoArgs,
            func=_make_list(org_lib_dir),
        ),
        StructuredTool(
            name=PUX_DYN_PREFIX + "call_function",
            description=(
                "Run an EXISTING function by name, passing `arguments` (a JSON "
                "object of keyword arguments) to its `run(**kwargs)`. Returns "
                "the function's result value (bounded — the function body stays "
                "out of your context, only its return value comes back)."
            ),
            args_schema=_call_args_model(),
            func=_make_call(org_lib_dir, exec_client, label),
        ),
    ]
