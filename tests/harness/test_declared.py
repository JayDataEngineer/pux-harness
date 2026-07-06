"""Declared sandbox tools — typed, by-name tools synthesized from per-org
``sandbox/tools/tools.yaml`` whose ``func`` exec's a script IN-CONTAINER.

These tests prove the auto-translate path the weak-model reliability fix rests on:
the model calls ``pux_sandbox_<name>(typed, args)`` directly instead of emitting
untyped ``python3 sandbox/x.py --rank 5`` shell. Covers:

- loading + the data model (``test_load_*``);
- the synthesized StructuredTool name/schema (``test_build_*``);
- command serialization — flags / positional / subcommand / boolean / None
  (``test_command_*``) — the crux of "the harness builds the shell, not the model";
- the runtime resolver relaxation — a declared name in ``tool_map`` resolves
  instead of false-raising (``test_resolve_*``);
- the offline validator — clean yaml passes; every malformation fails loud
  (``test_validate_*``).

Mirrors ``test_registry.py`` (the sibling tool-surface test) for conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from pux_harness.agent.orgs import _resolve_tools
from pux_harness.sandbox.tools import declared as D
from pux_harness.sandbox.tools._shared import PUX_PREFIX
from pux_harness.sandbox.tools.declared import (
    ArgSpec,
    DeclaredToolSpec,
    build_declared_tools,
    declared_tool_names,
    load_declared_specs,
    validate_declared_tools,
)


# --- fakes -----------------------------------------------------------------

class _FakeExec:
    """Records every exec(command, timeout) and returns a canned (out, code)."""

    def __init__(self, out: str = "ok\n", exit_code: int = 0):
        self.calls: list[tuple[str, Any]] = []
        self._out = out
        self._exit = exit_code

    def exec(self, command: str, *, timeout: int | None = None) -> tuple[str, int]:
        self.calls.append((command, timeout))
        return (self._out, self._exit)


def _spec(
    name: str = "scan_signals", script: str = "signals.py", *,
    subcommand: str | None = None, invoke: str = "flags", returns: str = "text",
    args: tuple[ArgSpec, ...] = (),
    timeout: int | None = None,
) -> DeclaredToolSpec:
    return DeclaredToolSpec(
        name=name, description="d", script=script, subcommand=subcommand,
        invoke=invoke, timeout=timeout, returns=returns, args=args,
    )


def _write_tools_yaml(org_sandbox_dir: Path, body: str) -> None:
    (org_sandbox_dir / "tools").mkdir(parents=True, exist_ok=True)
    (org_sandbox_dir / "tools" / "tools.yaml").write_text(body)


# --- loading ---------------------------------------------------------------

def test_load_returns_empty_when_no_yaml(tmp_path: Path):
    """No tools.yaml -> [] (every org without a declaration is unaffected)."""
    assert load_declared_specs(tmp_path) == []


def test_load_parses_full_shape(tmp_path: Path):
    """Every field round-trips: types, required, defaults, subcommand, invoke."""
    (tmp_path / "echo.py").write_text("# real script\n")
    _write_tools_yaml(tmp_path, """
tools:
  - name: scan_signals
    description: "Scan a ticker."
    script: echo.py
    subcommand: scan
    invoke: flags
    timeout: 90
    returns: json
    args:
      - {name: ticker, type: string, required: true, description: "sym"}
      - {name: rank, type: integer, required: false, default: 5}
      - {name: verbose, type: boolean, required: false, default: false}
""")
    specs = load_declared_specs(tmp_path)
    assert len(specs) == 1
    s = specs[0]
    assert s.name == "scan_signals"
    assert s.script == "echo.py"
    assert s.subcommand == "scan"
    assert s.invoke == "flags"
    assert s.timeout == 90
    assert s.returns == "json"
    assert len(s.args) == 3
    assert s.args[0] == ArgSpec("ticker", "string", True, None, "sym")
    assert s.args[1] == ArgSpec("rank", "integer", False, 5, "")
    assert s.args[2] == ArgSpec("verbose", "boolean", False, False, "")
    assert declared_tool_names(tmp_path) == frozenset({"scan_signals"})


# --- building --------------------------------------------------------------

def test_build_empty_when_no_yaml(tmp_path: Path):
    assert build_declared_tools(tmp_path, exec_client=_FakeExec()) == []


def test_build_synthesizes_named_structured_tool(tmp_path: Path, monkeypatch):
    """A declared tool surfaces as ``pux_sandbox_<name>`` with a schema whose
    required field is required and whose optional field carries its default."""
    (tmp_path / "echo.py").write_text("# real\n")
    _write_tools_yaml(tmp_path, """
tools:
  - name: scan_signals
    description: "Scan."
    script: echo.py
    args:
      - {name: ticker, type: string, required: true}
      - {name: rank, type: integer, required: false, default: 5}
""")
    monkeypatch.setattr(D, "project_root", lambda: tmp_path)  # tmp_path IS the project root
    tools = build_declared_tools(tmp_path, exec_client=_FakeExec())
    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, StructuredTool)
    assert tool.name == PUX_PREFIX + "scan_signals"
    schema = tool.args_schema.model_json_schema()["properties"]
    assert set(schema) == {"ticker", "rank"}
    # required propagates to the JSON schema
    required = tool.args_schema.model_json_schema().get("required", [])
    assert "ticker" in required and "rank" not in required


# --- command serialization (the crux) --------------------------------------

def _runner_tool(spec: DeclaredToolSpec, exec_client: _FakeExec) -> StructuredTool:
    """Build a tool with a fixed container dir (bypasses project_root resolution
    so command-shape tests need no monkeypatch)."""
    container = Path("/sandbox/workspace/orgs/x/sandbox")
    return StructuredTool(
        name=PUX_PREFIX + spec.name, description=spec.description or "d",
        args_schema=D._build_args_model(spec), func=D._make_runner(spec, exec_client, container),
    )


def test_command_flags_style():
    """flags (default): --name value per present arg; boolean True -> --flag."""
    spec = _spec(args=(
        ArgSpec("ticker", "string", True, None, ""),
        ArgSpec("rank", "integer", False, 5, ""),
        ArgSpec("verbose", "boolean", False, False, ""),
    ))
    fx = _FakeExec()
    tool = _runner_tool(spec, fx)
    tool.invoke({"ticker": "AAPL", "rank": 5, "verbose": True})
    cmd = fx.calls[0][0]
    assert cmd == (
        "cd /sandbox/workspace/orgs/x/sandbox && python3 signals.py "
        "--ticker AAPL --rank 5 --verbose"
    ), cmd


def test_command_boolean_false_omitted_and_none_skipped():
    """boolean False -> flag omitted; a None optional -> arg skipped (no --name)."""
    spec = _spec(args=(
        ArgSpec("ticker", "string", True, None, ""),
        ArgSpec("verbose", "boolean", False, False, ""),
        ArgSpec("rank", "integer", False, None, ""),
    ))
    fx = _FakeExec()
    tool = _runner_tool(spec, fx)
    tool.invoke({"ticker": "AAPL", "verbose": False, "rank": None})
    cmd = fx.calls[0][0]
    assert cmd == "cd /sandbox/workspace/orgs/x/sandbox && python3 signals.py --ticker AAPL", cmd


def test_command_positional_style_and_subcommand():
    """positional: values in declared order; subcommand: a leading word."""
    spec = _spec(invoke="positional", subcommand="scan", args=(
        ArgSpec("ticker", "string", True, None, ""),
        ArgSpec("rank", "integer", False, None, ""),
    ))
    fx = _FakeExec()
    tool = _runner_tool(spec, fx)
    tool.invoke({"ticker": "AAPL", "rank": 3})
    cmd = fx.calls[0][0]
    assert cmd == (
        "cd /sandbox/workspace/orgs/x/sandbox && python3 signals.py scan AAPL 3"
    ), cmd


def test_command_quotes_shell_unsafe_values():
    """A value with spaces is shlex-quoted — the model can't break out of argv."""
    spec = _spec(args=(ArgSpec("q", "string", True, None, ""),))
    fx = _FakeExec()
    tool = _runner_tool(spec, fx)
    tool.invoke({"q": "a b'; rm -rf /"})
    cmd = fx.calls[0][0]
    assert "--q " in cmd
    # the dangerous value is single-quoted, not interpolated raw
    assert "rm -rf /" in cmd and "'a b';" not in cmd


def test_result_envelope_success_and_failure():
    """exit 0 -> success envelope with output; non-zero -> error envelope with
    the exit code + command."""
    spec = _spec(args=(ArgSpec("ticker", "string", True, None, ""),))

    ok = _FakeExec(out="result\n", exit_code=0)
    ret_ok = _runner_tool(spec, ok).invoke({"ticker": "AAPL"})
    assert '"success": true' in ret_ok and "result" in ret_ok

    bad = _FakeExec(out="traceback\n", exit_code=2)
    ret_bad = _runner_tool(spec, bad).invoke({"ticker": "AAPL"})
    assert '"success": false' in ret_bad and "exited 2" in ret_bad and "signals.py" in ret_bad


def test_result_json_parse_and_malformed():
    """returns: json parses stdout; malformed stdout -> error envelope."""
    spec_ok = _spec(returns="json", args=(ArgSpec("ticker", "string", True, None, ""),))
    fx_ok = _FakeExec(out='{"hits": 3}', exit_code=0)
    ret = _runner_tool(spec_ok, fx_ok).invoke({"ticker": "AAPL"})
    assert '"json"' in ret and '"hits"' in ret and "3" in ret

    spec_bad = _spec(returns="json", args=(ArgSpec("ticker", "string", True, None, ""),))
    fx_bad = _FakeExec(out="not json", exit_code=0)
    ret_bad = _runner_tool(spec_bad, fx_bad).invoke({"ticker": "AAPL"})
    assert '"success": false' in ret_bad and "not valid JSON" in ret_bad


# --- runtime resolver relaxation -------------------------------------------

def _mk_tool(name: str) -> StructuredTool:
    return StructuredTool(name=name, description="d", args_schema=BaseModel, func=lambda **_: "")


def test_resolve_admits_declared_tool_in_map():
    """A declared name (classify_slug -> None) NOW resolves when its
    ``pux_sandbox_*`` key is in the map. Before this fix it false-raised."""
    tool = _mk_tool(PUX_PREFIX + "scan_signals")
    resolved = _resolve_tools(["scan_signals"], {PUX_PREFIX + "scan_signals": tool})
    assert resolved == [tool]


def test_resolve_still_raises_on_unknown_name():
    with pytest.raises(KeyError):
        _resolve_tools(["totally_made_up"], {PUX_PREFIX + "scan_signals": _mk_tool("x")})


def test_resolve_still_raises_on_specialist_not_in_map():
    """A known specialist that isn't enabled for this agent still fails loud."""
    with pytest.raises(KeyError):
        _resolve_tools(["python"], {})  # python is a real specialist; absent here


# --- validator (offline contract body) -------------------------------------

def _valid_tree(tmp_path: Path) -> Path:
    """A clean org sandbox dir: valid yaml + a real script on disk."""
    org_dir = tmp_path / "orgs" / "acme" / "sandbox"
    org_dir.mkdir(parents=True)
    (org_dir / "echo.py").write_text("# real script\n")
    _write_tools_yaml(org_dir, """
tools:
  - name: scan_signals
    description: "Scan."
    script: echo.py
    args:
      - {name: ticker, type: string, required: true}
""")
    return org_dir


def test_validate_clean_passes(tmp_path: Path):
    assert validate_declared_tools(_valid_tree(tmp_path)) == []


def test_validate_empty_when_no_yaml(tmp_path: Path):
    assert validate_declared_tools(tmp_path) == []


@pytest.mark.parametrize("name, hint", [
    ("Scan_Signals", "snake_case"),      # uppercase
    ("9signals", "snake_case"),          # leading digit
    ("config", "reserved"),              # langchain reserved
    ("runtime", "reserved"),
    ("execute", "shadows"),              # native fs/shell
    ("read_file", "shadows"),
    ("mcp__foo", "reserved"),            # mcp namespace
    ("python", "collides"),              # REGISTRY specialist -> pux_sandbox_python
    ("bash", "collides"),                # LEGACY_TOOL_NAMES -> pux_sandbox_bash
])
def test_validate_rejects_bad_names(tmp_path: Path, name: str, hint: str):
    org_dir = tmp_path / "orgs" / "acme" / "sandbox"
    org_dir.mkdir(parents=True)
    (org_dir / "echo.py").write_text("# real\n")
    _write_tools_yaml(org_dir, f"""
tools:
  - name: {name}
    description: "d"
    script: echo.py
    args:
      - {{name: ticker, type: string, required: true}}
""")
    errs = validate_declared_tools(org_dir)
    assert errs, f"expected {name!r} ({hint}) to be rejected, got no errors"
    # the message(s) must name the offending tool. ``execute``/``read_file``
    # legitimately trip TWO checks (native + grader shadow — both partitions of
    # REGISTRY contain those slugs), so assert presence, not a single error.
    assert any(name in e for e in errs), errs


def test_validate_rejects_duplicate_names(tmp_path: Path):
    org_dir = tmp_path / "orgs" / "acme" / "sandbox"
    org_dir.mkdir(parents=True)
    (org_dir / "echo.py").write_text("# real\n")
    _write_tools_yaml(org_dir, """
tools:
  - {name: dup, description: "d", script: echo.py, args: []}
  - {name: dup, description: "d", script: echo.py, args: []}
""")
    errs = validate_declared_tools(org_dir)
    assert any("duplicate" in e for e in errs), errs


def test_validate_rejects_missing_script_file(tmp_path: Path):
    org_dir = tmp_path / "orgs" / "acme" / "sandbox"
    org_dir.mkdir(parents=True)
    _write_tools_yaml(org_dir, """
tools:
  - {name: scan_signals, description: "d", script: ghost.py, args: []}
""")
    errs = validate_declared_tools(org_dir)
    assert any("ghost.py" in e and "not found" in e for e in errs), errs


def test_validate_rejects_bad_arg_type_and_bad_enums(tmp_path: Path):
    org_dir = tmp_path / "orgs" / "acme" / "sandbox"
    org_dir.mkdir(parents=True)
    (org_dir / "echo.py").write_text("# real\n")
    _write_tools_yaml(org_dir, """
tools:
  - name: scan_signals
    description: "d"
    script: echo.py
    invoke: weird
    returns: csv
    args:
      - {name: ticker, type: filepath, required: true}
""")
    errs = validate_declared_tools(org_dir)
    joined = " ".join(errs)
    assert "filepath" in joined           # bad arg type
    assert "invoke" in joined and "weird" in joined
    assert "returns" in joined and "csv" in joined


def test_validate_rejects_required_with_default(tmp_path: Path):
    org_dir = tmp_path / "orgs" / "acme" / "sandbox"
    org_dir.mkdir(parents=True)
    (org_dir / "echo.py").write_text("# real\n")
    _write_tools_yaml(org_dir, """
tools:
  - name: scan_signals
    description: "d"
    script: echo.py
    args:
      - {name: ticker, type: string, required: true, default: "X"}
""")
    errs = validate_declared_tools(org_dir)
    assert any("default" in e for e in errs), errs
