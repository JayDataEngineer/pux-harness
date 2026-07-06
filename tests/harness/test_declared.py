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

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from pux_harness.agent.orgs import _resolve_tools
from pux_harness.sandbox.tools import declared as D
from pux_harness.sandbox.tools._shared import PUX_PREFIX
from pux_harness.sandbox.tools.declared import (
    ArgSpec,
    DeclaredToolSpec,
    build_declared_tools,
    build_script_redirects,
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


# --- exec-guard redirect map (declared ⇒ taken out of `execute`) -----------
# ``build_script_redirects`` compiles each declared spec into a (pattern, target)
# pair ``RoutingMiddleware`` matches intercepted ``execute``/bash commands
# against. The correctness crux is per-(script, subcommand) scoping: a declared
# ``scan_signals`` (wraps ``signals.py score``) must block raw exec of
# ``signals.py score`` but LEAVE ``signals.py rank``/``validate`` exec-able.


def test_redirects_empty_for_no_specs():
    """No declared tools -> no redirects (byte-identical routing for orgs that
    declare nothing)."""
    assert build_script_redirects([]) == []


def test_redirects_target_is_prefixed_tool_name():
    """Each redirect's target is ``pux_sandbox_<name>`` — the typed tool the
    agent should call instead of the raw script."""
    redirs = build_script_redirects([_spec(name="scan_signals", subcommand="score")])
    assert len(redirs) == 1
    pattern, target = redirs[0]
    assert target == PUX_PREFIX + "scan_signals"
    assert pattern.search("python3 signals.py score --ticker AAPL") is not None


def test_redirects_match_real_command_shapes():
    """The pattern tolerates the shapes the model actually emits via ``execute``:
    a ``cd <dir> && python3 …`` prefix, an absolute/relative path before the
    basename, flags after the subcommand, and the bare ``python3 … script`` form."""
    redirs = build_script_redirects([_spec(subcommand="score")])
    pattern, _ = redirs[0]
    for cmd in (
        "python3 signals.py score --ticker AAPL",
        "cd /sandbox/workspace/orgs/x/sandbox && python3 signals.py score --ticker AAPL",
        "python3 orgs/specialists/invest/sandbox/signals.py score --ticker AAPL",
        "python3 ./signals.py score",
    ):
        assert pattern.search(cmd) is not None, f"should match: {cmd!r}"


def test_redirects_do_not_match_a_different_subcommand():
    """Per-(script, subcommand) scoping: ``signals.py rank``/``validate`` are
    NOT exposed by ``scan_signals`` (which wraps only ``score``), so they stay
    exec-able — the agent has no typed alternative for those."""
    pattern, _ = build_script_redirects([_spec(subcommand="score")])[0]
    assert pattern.search("python3 signals.py rank") is None
    assert pattern.search("python3 signals.py validate") is None
    assert pattern.search("python3 signals.py") is None  # no subcommand at all


def test_redirects_do_not_match_a_different_script():
    """A redirect for ``signals.py`` must not match ``portfolio.py`` even with
    the same subcommand — the pattern is anchored on the script filename."""
    pattern, _ = build_script_redirects([_spec(subcommand="score")])[0]
    assert pattern.search("python3 portfolio.py score") is None


def test_redirects_word_boundaries_stop_false_matches():
    """``\\b`` keeps the redirect surgical: ``my_signals.py`` is not ``signals.py``
    (the ``_`` before ``signals`` is a word char → no boundary), and ``score``
    is not ``scoreboard`` (no boundary after ``score``)."""
    pattern, _ = build_script_redirects([_spec(subcommand="score")])[0]
    assert pattern.search("python3 my_signals.py score") is None
    assert pattern.search("python3 signals.py scoreboard") is None


def test_redirects_no_subcommand_blocks_whole_script():
    """A spec with no ``subcommand`` exposes the whole script, so EVERY
    ``python3 … <script>`` invocation is redirected — including subcommand-style
    calls, since the tool owns the script's entire surface."""
    pattern, _ = build_script_redirects([_spec(subcommand=None)])[0]
    assert pattern.search("python3 signals.py --ticker AAPL") is not None
    assert pattern.search("python3 signals.py rank") is not None
    assert pattern.search("cd x && python3 signals.py") is not None


def test_redirects_one_per_spec_and_order_preserved():
    """Multiple declared tools -> one redirect each, in declaration order (the
    order ``RoutingMiddleware`` iterates; first match wins in practice but order
    is deterministic for the redirect message)."""
    specs = [
        _spec(name="scan_signals", script="signals.py", subcommand="score"),
        _spec(name="rank_tickers", script="ranker.py", subcommand=None),
    ]
    redirs = build_script_redirects(specs)
    assert [t for _, t in redirs] == [
        PUX_PREFIX + "scan_signals",
        PUX_PREFIX + "rank_tickers",
    ]
    # the second spec (no subcommand) still matches its own script.
    assert redirs[1][0].search("python3 ranker.py --top 10") is not None


# --- exec-guard: RoutingMiddleware redirects declared-script exec ----------
# End-to-end proof of the invariant: a script exposed as a typed tool must be
# reached via that tool, NOT via raw ``execute``. ``RoutingMiddleware`` fed the
# declared redirect map returns a redirect ``ToolMessage`` WITHOUT awaiting the
# handler (the command never runs). Mirrors the ``test_audit.py`` request/
# handler pattern.

def _routing_mw(specs):
    """A RoutingMiddleware armed with the redirect map for ``specs``."""
    from pux_harness.context.sandbox_routing import RoutingMiddleware
    return RoutingMiddleware(declared_redirects=build_script_redirects(specs))


def _exec_req(command: str):
    """A fake execute/bash tool-call request carrying ``command`` + a thread_id."""
    from types import SimpleNamespace
    return SimpleNamespace(
        tool_call={"name": "execute", "id": "c1", "args": {"command": command}},
        state={"configurable": {"thread_id": "t1"}},
    )


def _recording_handler(ran: list):
    """Async handler that records the call + returns a 'RAN' ToolMessage."""
    async def _h(_req):
        ran.append(True)
        return ToolMessage(content="RAN", name="execute", tool_call_id="c1")
    return _h


def test_routing_redirects_declared_script_exec_without_running_it():
    """The exec-guard: an ``execute("python3 signals.py score …")`` targeting a
    declared script is REDIRECTED — returns a ToolMessage naming the typed tool
    and the handler is NEVER awaited (the command does not run)."""
    mw = _routing_mw([_spec(name="scan_signals", subcommand="score")])
    ran: list = []
    out = asyncio.run(mw.awrap_tool_call(
        _exec_req("python3 signals.py score --ticker AAPL"),
        handler=_recording_handler(ran)))
    assert ran == [], "declared-script exec must NOT run the command"
    assert isinstance(out, ToolMessage)
    assert out.name == "execute"
    target = PUX_PREFIX + "scan_signals"
    assert target in out.content, out.content
    assert out.tool_call_id == "c1"


def test_routing_allows_non_declared_script_exec():
    """A script NOT exposed as a typed tool stays exec-able — the guard is
    surgical, not a blanket ``execute`` block. Handler IS awaited."""
    mw = _routing_mw([_spec(name="scan_signals", subcommand="score")])
    ran: list = []
    out = asyncio.run(mw.awrap_tool_call(
        _exec_req("python3 portfolio.py rebalance"),
        handler=_recording_handler(ran)))
    assert ran == [True], "non-declared script exec must run normally"
    assert out.content == "RAN"


def test_routing_allows_unexposed_subcommand_of_declared_script():
    """Per-(script, subcommand) scoping at the MIDDLEWARE level:
    ``signals.py rank`` is NOT exposed by ``scan_signals`` (wraps only
    ``score``), so it runs — the agent has no typed alternative for it."""
    mw = _routing_mw([_spec(name="scan_signals", subcommand="score")])
    ran: list = []
    asyncio.run(mw.awrap_tool_call(
        _exec_req("cd /sandbox/workspace/orgs/x/sandbox && python3 signals.py rank"),
        handler=_recording_handler(ran)))
    assert ran == [True], "an un-exposed subcommand must stay exec-able"


def test_routing_redirect_real_command_shapes():
    """The redirect fires on the shapes the model actually emits: a
    ``cd <dir> && python3 …`` prefix and an absolute path before the basename."""
    mw = _routing_mw([_spec(name="scan_signals", subcommand="score")])
    for cmd in (
        "cd /sandbox/workspace/orgs/specialists/invest/sandbox && "
        "python3 signals.py score --ticker AAPL",
        "python3 orgs/specialists/invest/sandbox/signals.py score",
    ):
        ran: list = []
        out = asyncio.run(mw.awrap_tool_call(_exec_req(cmd), handler=_recording_handler(ran)))
        assert ran == [], f"should redirect (not run): {cmd!r}"
        assert PUX_PREFIX + "scan_signals" in out.content


def test_routing_deny_still_wins_over_redirect_for_network():
    """Network egress deny takes precedence: a command with ``curl`` is denied
    (the ``_DENY_MSG``), not redirected, even if it also names a declared
    script. Both block the command; deny is the conservative choice."""
    from pux_harness.context.sandbox_routing import _DENY_MSG
    mw = _routing_mw([_spec(name="scan_signals", subcommand="score")])
    out = asyncio.run(mw.awrap_tool_call(
        _exec_req("python3 signals.py score && curl http://exfiltrate.example"),
        handler=_recording_handler([])))
    assert out.content == _DENY_MSG


def test_routing_sync_wrap_mirrors_async_redirect():
    """The sync ``wrap_tool_call`` (used by synchronous runners) redirects the
    same as the async path — no async-only gap."""
    mw = _routing_mw([_spec(name="scan_signals", subcommand="score")])
    ran: list = []
    out = mw.wrap_tool_call(
        _exec_req("python3 signals.py score"),
        handler=lambda _r: _recording_handler(ran)(_r))  # sync caller; redirect never awaits
    assert ran == [], "sync path must also not run a redirected command"
    assert PUX_PREFIX + "scan_signals" in out.content


def test_routing_no_redirects_is_byte_identical_to_before():
    """Default ``declared_redirects=[]`` (an org that declares nothing) leaves
    routing byte-identical: a normal script exec is allowed (handler awaited)."""
    from pux_harness.context.sandbox_routing import RoutingMiddleware
    mw = RoutingMiddleware()  # no declared_redirects
    assert mw.declared_redirects == []
    ran: list = []
    asyncio.run(mw.awrap_tool_call(
        _exec_req("python3 anything.py go"),
        handler=_recording_handler(ran)))
    assert ran == [True]


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
