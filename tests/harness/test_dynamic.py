"""Dynamic (level c) sandbox tools — agent-authored, persistent, in-container.

The FOUR tool rung: ``pux_dyn_make_function``/``edit_function``/``list_functions``/
``call_function`` let the agent author Python under ``orgs/<org>/lib/functions/``
and call it back later. These tests prove:

- the synthesized surface (4 fixed tools, ``pux_dyn_`` prefix) — ``test_build_*``;
- host-side authoring writes the module + ``__init__.py`` + ``index.yaml``
  (``test_make_*``) and guards name/dupe/syntax/no-``run`` at authoring time;
- edit bumps version + rejects the missing; list is bounded to the index
  (``test_edit_*`` / ``test_list_*``);
- call builds the in-container command shape (cd, env-var kwargs,
  ``PYTHONDONTWRITEBYTECODE``) and parses the marker-delimited envelope, bumping
  usage/success in the index (``test_call_*``);
- THE THESIS (``test_thesis_*``): a function's BODY — however large — never
  enters the result the model sees; only its bounded return value does. That is
  the per-turn context drop after the org "learns" a function.

Mirrors ``test_declared.py`` for conventions (``_FakeExec``, ``tmp_path``).
``project_root`` is monkeypatched to ``tmp_path`` so the host/container path math
(``relative_to``) resolves against the scratch lib dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pux_harness.sandbox.tools import dynamic as D
from pux_harness.sandbox.tools._shared import _NoArgs


# --- fakes -----------------------------------------------------------------

class _FakeExec:
    """Records execute(command, timeout); returns a canned ExecuteResponse-like."""

    def __init__(self, out: str = "", exit_code: int = 0):
        self.calls: list[tuple[str, Any]] = []
        self._out = out
        self._exit = exit_code

    def execute(self, command: str, *, timeout: int | None = None):
        from collections import namedtuple
        self.calls.append((command, timeout))
        _Resp = namedtuple("_Resp", ["output", "exit_code", "truncated"])
        return _Resp(self._out, self._exit, False)


_GOOD = "def run(**kwargs):\n    return sum(kwargs['nums'])\n"


@pytest.fixture
def org_lib(tmp_path: Path, monkeypatch) -> Path:
    """A scratch ``lib/`` dir with dynamic.project_root patched to tmp_path.

    In production the lib dir is always under the real project root; the patch
    makes the ``relative_to(project_root())`` path math resolve against the
    scratch dir so no repo path is hard-coded."""
    monkeypatch.setattr(D, "project_root", lambda: tmp_path)
    lib = tmp_path / "lib"
    return lib


def _tools(lib: Path, exec_client: _FakeExec | None = None) -> dict[str, Any]:
    return {t.name: t for t in D.build_dynamic_tools(lib, exec_client or _FakeExec())}


def _invoke(tool, **kw) -> dict:
    return json.loads(tool.invoke(kw))


# --- build / surface -------------------------------------------------------

def test_build_dynamic_tools_fixed_surface(org_lib):
    tools = D.build_dynamic_tools(org_lib, _FakeExec())
    names = sorted(t.name for t in tools)
    assert names == sorted("pux_dyn_" + n for n in D.DYNAMIC_TOOL_NAMES)
    # every tool has a real args schema (list_functions -> _NoArgs)
    assert all(t.args_schema is not None for t in tools)
    list_tool = next(t for t in tools if t.name == "pux_dyn_list_functions")
    assert list_tool.args_schema is _NoArgs


def test_dynamic_tool_names_is_the_four(org_lib):
    assert D.DYNAMIC_TOOL_NAMES == frozenset(
        {"make_function", "edit_function", "list_functions", "call_function"}
    )


# --- make_function (host-side authoring) -----------------------------------

def test_make_writes_module_init_and_index(org_lib):
    t = _tools(org_lib)
    r = _invoke(t["pux_dyn_make_function"], name="add_nums", description="sum a list", code=_GOOD)
    assert r["success"] is True
    assert r["name"] == "add_nums"
    assert r["path"] == "lib/functions/add_nums.py"
    assert (org_lib / "functions" / "add_nums.py").read_text() == _GOOD
    # __init__.py ensured on first use (makes `functions` an importable package)
    assert (org_lib / "functions" / "__init__.py").is_file()
    # index.yaml is the bookkeeping source of truth
    idx = D.load_dynamic_index(org_lib)
    assert idx["add_nums"]["description"] == "sum a list"
    assert idx["add_nums"]["usage"] == 0
    assert idx["add_nums"]["version"] == 1


def test_make_rejects_duplicate(org_lib):
    t = _tools(org_lib)
    assert _invoke(t["pux_dyn_make_function"], name="f", description="x", code=_GOOD)["success"]
    r = _invoke(t["pux_dyn_make_function"], name="f", description="x", code=_GOOD)
    assert r["success"] is False
    assert "already exists" in r["error"]
    assert "edit_function" in r["error"]  # points the model at the right tool


@pytest.mark.parametrize("bad", ["UPPER", "1num", "has-dash", "weird!"])
def test_make_rejects_non_snake_name(org_lib, bad):
    t = _tools(org_lib)
    r = _invoke(t["pux_dyn_make_function"], name=bad, description="x", code=_GOOD)
    assert r["success"] is False and "snake_case" in r["error"]


def test_make_rejects_syntax_error(org_lib):
    t = _tools(org_lib)
    r = _invoke(t["pux_dyn_make_function"], name="broken", description="x",
                code="def run(**kw):\n   return(\n")
    assert r["success"] is False
    assert "syntax" in r["error"].lower()
    # nothing written on rejection
    assert not (org_lib / "functions" / "broken.py").exists()
    assert "broken" not in D.load_dynamic_index(org_lib)


def test_make_rejects_code_without_run(org_lib):
    t = _tools(org_lib)
    r = _invoke(t["pux_dyn_make_function"], name="norun", description="x",
                code="def helper(**kw):\n    return 1\n")
    assert r["success"] is False
    assert "def run" in r["error"]


# --- edit_function ---------------------------------------------------------

def test_edit_bumps_version(org_lib):
    t = _tools(org_lib)
    _invoke(t["pux_dyn_make_function"], name="f", description="x", code=_GOOD)
    new = "def run(**kwargs):\n    return max(kwargs['nums'])\n"
    r = _invoke(t["pux_dyn_edit_function"], name="f", code=new)
    assert r["success"] is True and r["version"] == 2
    assert (org_lib / "functions" / "f.py").read_text() == new
    # usage/success preserved across the edit
    idx = D.load_dynamic_index(org_lib)
    assert idx["f"]["version"] == 2 and "edited" in idx["f"]


def test_edit_rejects_missing(org_lib):
    t = _tools(org_lib)
    r = _invoke(t["pux_dyn_edit_function"], name="ghost", code=_GOOD)
    assert r["success"] is False and "does not exist" in r["error"]


def test_edit_rejects_syntax_error(org_lib):
    t = _tools(org_lib)
    _invoke(t["pux_dyn_make_function"], name="f", description="x", code=_GOOD)
    r = _invoke(t["pux_dyn_edit_function"], name="f", code="def run(**kw):\n   return(\n")
    assert r["success"] is False and "syntax" in r["error"].lower()
    # original body untouched on rejection
    assert (org_lib / "functions" / "f.py").read_text() == _GOOD


# --- list_functions (cheap, bounded to index) ------------------------------

def test_list_empty(org_lib):
    t = _tools(org_lib)
    r = _invoke(t["pux_dyn_list_functions"])
    assert r["success"] is True and r["count"] == 0 and r["functions"] == []


def test_list_is_bounded_to_index_not_code(org_lib):
    t = _tools(org_lib)
    body = "def run(**kw):\n    # SECRET_MARKER_DO_NOT_LEAK\n    return 1\n"
    _invoke(t["pux_dyn_make_function"], name="f", description="d", code=body)
    payload = t["pux_dyn_list_functions"].invoke({})
    assert "SECRET_MARKER_DO_NOT_LEAK" not in payload  # body never in list
    parsed = json.loads(payload)
    assert parsed["functions"][0]["description"] == "d"


# --- call_function (in-container exec + bookkeeping) -----------------------

def test_call_command_shape(org_lib):
    """The harness builds the shell — cd into lib, kwargs via env var, no
    bytecode cache, python3 -c runner. The model never quotes arbitrary
    values."""
    fx = _FakeExec(out="\n" + D._RESULT_MARKER + "\n" + json.dumps({"ok": True, "value": 6}))
    t = _tools(org_lib, fx)
    _invoke(t["pux_dyn_make_function"], name="add", description="x", code=_GOOD)
    _invoke(t["pux_dyn_call_function"], name="add", arguments={"nums": [1, 2, 3]})
    assert len(fx.calls) == 1
    cmd, timeout = fx.calls[0]
    # cd into the org lib dir (container path) then run python3 -c <runner>
    cd_target = cmd.split(" &&", 1)[0]
    assert cd_target.startswith("cd ") and cd_target.rstrip().endswith("lib")
    assert "_PUX_DYN_KWARGS=" in cmd
    assert "PYTHONDONTWRITEBYTECODE=1" in cmd
    assert "python3 -c " in cmd
    assert timeout == D.DEFAULT_DYN_TIMEOUT
    # kwargs travel as JSON (single-quoted by shlex), not positional shell args
    assert json.dumps({"nums": [1, 2, 3]}) in cmd


def test_call_success_parses_envelope_and_bumps(org_lib):
    fx = _FakeExec(out="debug noise\n" + "\n" + D._RESULT_MARKER + "\n"
                   + json.dumps({"ok": True, "value": [1, 2, 3]}))
    t = _tools(org_lib, fx)
    _invoke(t["pux_dyn_make_function"], name="f", description="x", code=_GOOD)
    r = _invoke(t["pux_dyn_call_function"], name="f", arguments={"nums": [1, 2, 3]})
    assert r["success"] is True
    assert r["value"] == [1, 2, 3]
    idx = D.load_dynamic_index(org_lib)
    assert idx["f"]["usage"] == 1 and idx["f"]["success"] == 1


def test_call_crash_no_marker_bumps_usage_only(org_lib):
    """Function crashed before printing the marker — bounded failure, usage
    counts but success does not."""
    fx = _FakeExec(out="Traceback (most recent call last):\n  boom\n", exit_code=0)
    t = _tools(org_lib, fx)
    _invoke(t["pux_dyn_make_function"], name="f", description="x", code=_GOOD)
    r = _invoke(t["pux_dyn_call_function"], name="f", arguments={})
    assert r["success"] is False
    assert "marker" in r["error"]
    idx = D.load_dynamic_index(org_lib)
    assert idx["f"]["usage"] == 1 and idx["f"]["success"] == 0


def test_call_nonzero_exit(org_lib):
    fx = _FakeExec(out="err", exit_code=2)
    t = _tools(org_lib, fx)
    _invoke(t["pux_dyn_make_function"], name="f", description="x", code=_GOOD)
    r = _invoke(t["pux_dyn_call_function"], name="f", arguments={})
    assert r["success"] is False and "exited with code 2" in r["error"]
    idx = D.load_dynamic_index(org_lib)
    assert idx["f"]["usage"] == 1 and idx["f"]["success"] == 0


def test_call_rejects_missing(org_lib):
    t = _tools(org_lib)
    r = _invoke(t["pux_dyn_call_function"], name="ghost", arguments={})
    assert r["success"] is False and "does not exist" in r["error"]


# --- THE THESIS: bounded results (the context-drop mechanism) --------------

def test_thesis_big_body_small_result_never_leaks(org_lib):
    """A function with a HUGE body that returns a tiny value: the model sees
    only the value, never the body. This is the mechanism that makes per-turn
    context DROP after the org "learns" a function — the whole point of level c."""
    sentinel = "X" * 50_000  # 50KB body the model must never re-read
    body = (
        "COMMENT = '" + sentinel + "'\n"
        "def run(**kwargs):\n"
        "    # " + sentinel + "\n"
        "    return 42\n"
    )
    fx = _FakeExec(out="\n" + D._RESULT_MARKER + "\n" + json.dumps({"ok": True, "value": 42}))
    t = _tools(org_lib, fx)
    _invoke(t["pux_dyn_make_function"], name="big", description="huge body, tiny result", code=body)

    call_payload = t["pux_dyn_call_function"].invoke({"name": "big", "arguments": {}})
    # the bounded value IS present...
    assert '"value": 42' in call_payload
    # ...but the 50KB body NEVER enters the result the model sees
    assert sentinel not in call_payload
    assert len(call_payload) < 500  # result is tiny despite a 50KB function


def test_thesis_list_never_leaks_bodies(org_lib):
    """After learning several functions, list_functions stays bounded by the
    index (descriptions + counts), never the cumulative code."""
    t = _tools(org_lib)
    for i in range(5):
        body = "SECRET_%d = '%s'\ndef run(**kw):\n    return %d\n" % (i, "Z" * 10_000, i)
        _invoke(t["pux_dyn_make_function"], name=f"f{i}", description=f"func {i}", code=body)
    payload = t["pux_dyn_list_functions"].invoke({})
    for i in range(5):
        assert ("Z" * 10_000) not in payload  # no body leaks
    parsed = json.loads(payload)
    assert parsed["count"] == 5
    assert len(payload) < 2000  # bounded by index, not 50KB of code


# --- index I/O -------------------------------------------------------------

def test_index_round_trip_sorted(org_lib):
    D._ensure_lib_skeleton(org_lib)
    funcs = {
        "zeta": {"description": "z", "usage": 0, "success": 0, "version": 1},
        "alpha": {"description": "a", "usage": 3, "success": 2, "version": 1},
    }
    D.save_dynamic_index(org_lib, funcs)
    # names sorted on write -> stable, reviewable diffs
    text = (org_lib / "index.yaml").read_text()
    assert text.index("alpha:") < text.index("zeta:")
    loaded = D.load_dynamic_index(org_lib)
    assert loaded["alpha"]["usage"] == 3


def test_load_index_absent_returns_empty(org_lib):
    # no lib dir at all yet
    assert D.load_dynamic_index(org_lib / "nested") == {}


# --- runner template against REAL python3 (prove, don't assert) -------------

def test_runner_executes_against_real_python3(org_lib):
    """The riskiest piece is the in-container runner template (``python3 -c``
    doing ``from functions.<name> import run``). Prove it end-to-end against
    REAL python3 — not a ``_FakeExec`` — in the exact shape ``call_function``
    builds: cwd = lib dir, kwargs via env var, ``PYTHONDONTWRITEBYTECODE=1``.
    Host python3 shares the container's cwd/``sys.path[0]`` semantics, so this
    covers import + run + marker + JSON + the bytecache guard; only container
    egress ACLs (not exercised by a pure compute fn) wait on a live container."""
    import os
    import subprocess

    D._ensure_lib_skeleton(org_lib)
    (org_lib / "functions" / "add.py").write_text(
        "def run(**kwargs):\n    return sum(kwargs['nums'])\n"
    )
    runner = D._RUNNER_TEMPLATE.replace("__NAME__", "add")
    env = {
        **os.environ,
        "_PUX_DYN_KWARGS": json.dumps({"nums": [10, 20, 30]}),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        ["python3", "-c", runner], cwd=str(org_lib), env=env,
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"runner failed: {proc.stderr}"
    assert D._extract_result(proc.stdout) == {"ok": True, "value": 60}
    # the bytecache guard keeps a root-owned __pycache__ off the host-visible lib
    assert not (org_lib / "functions" / "__pycache__").exists()


# --- graduation (promote_function: lib -> sandbox, c->b, git-tracked) --------

def test_promote_moves_to_sandbox_and_tracks_path(org_lib):
    """Graduation moves the module lib/functions/<n>.py -> sandbox/functions/<n>.py
    and the index entry lib/index.yaml -> sandbox/index.yaml. The returned path is
    under sandbox/ — that is what makes the function travel via Git AND Pack."""
    t = _tools(org_lib)
    _invoke(t["pux_dyn_make_function"], name="rsi", description="compute rsi", code=_GOOD)
    res = D.promote_function(org_lib, "rsi")
    assert res["success"] is True
    assert res["path"] == "sandbox/functions/rsi.py"
    # lib copy gone + de-indexed
    assert not (org_lib / "functions" / "rsi.py").exists()
    assert "rsi" not in D.load_dynamic_index(org_lib)
    # sandbox copy present + indexed, with a promoted stamp + package init
    sb = org_lib.parent / "sandbox"
    assert (sb / "functions" / "rsi.py").read_text() == _GOOD
    assert (sb / "functions" / "__init__.py").is_file()
    sb_idx = D.load_dynamic_index(sb)
    assert sb_idx["rsi"]["description"] == "compute rsi"
    assert "promoted" in sb_idx["rsi"]


def test_call_promoted_resolves_sandbox_root(org_lib):
    """A promoted function is still callable — call_function resolves its owning
    root (sandbox, since lib no longer has it), cds into the SANDBOX container
    dir, and bumps the SANDBOX index (not lib)."""
    fx = _FakeExec(out="\n" + D._RESULT_MARKER + "\n" + json.dumps({"ok": True, "value": 7}))
    t = _tools(org_lib, fx)
    _invoke(t["pux_dyn_make_function"], name="f", description="x", code=_GOOD)
    D.promote_function(org_lib, "f")
    r = _invoke(t["pux_dyn_call_function"], name="f", arguments={"nums": [1, 2, 3]})
    assert r["success"] is True and r["value"] == 7
    cmd, _ = fx.calls[0]
    cd_target = cmd.split(" &&", 1)[0]
    assert cd_target.rstrip().endswith("sandbox")  # cd into the SANDBOX root
    sb = org_lib.parent / "sandbox"
    assert D.load_dynamic_index(sb)["f"]["usage"] == 1
    assert "f" not in D.load_dynamic_index(org_lib)  # lib index untouched


def test_edit_refuses_promoted(org_lib):
    """Promoted = operator-owned source; the agent may not rewrite it. edit_function
    refuses with a pointer at the tracked source, and leaves it untouched."""
    t = _tools(org_lib)
    _invoke(t["pux_dyn_make_function"], name="f", description="x", code=_GOOD)
    D.promote_function(org_lib, "f")
    r = _invoke(t["pux_dyn_edit_function"], name="f", code=_GOOD)
    assert r["success"] is False and "PROMOTED" in r["error"]
    sb = org_lib.parent / "sandbox"
    assert (sb / "functions" / "f.py").read_text() == _GOOD  # untouched


def test_list_merges_lib_and_sandbox_with_source(org_lib):
    t = _tools(org_lib)
    _invoke(t["pux_dyn_make_function"], name="a", description="lib fn", code=_GOOD)
    _invoke(t["pux_dyn_make_function"], name="b", description="to promote", code=_GOOD)
    D.promote_function(org_lib, "b")
    r = _invoke(t["pux_dyn_list_functions"])
    assert r["count"] == 2
    by = {f["name"]: f for f in r["functions"]}
    assert by["a"]["source"] == "lib"
    assert by["b"]["source"] == "sandbox"


def test_promote_missing_errors(org_lib):
    res = D.promote_function(org_lib, "ghost")
    assert res["success"] is False and "does not exist" in res["error"]


def test_promote_twice_errors(org_lib):
    t = _tools(org_lib)
    _invoke(t["pux_dyn_make_function"], name="f", description="x", code=_GOOD)
    assert D.promote_function(org_lib, "f")["success"] is True
    res = D.promote_function(org_lib, "f")
    assert res["success"] is False and "already promoted" in res["error"]


# --- pruning (archive_function: lib -> lib/.archive/, reversible) ------------

def test_archive_retires_and_keeps_source(org_lib):
    """Archiving drops the function from the active surface (list/call lose it)
    but PRESERVES the source under lib/.archive/ so the operator can restore it."""
    t = _tools(org_lib)
    _invoke(t["pux_dyn_make_function"], name="old", description="x", code=_GOOD)
    res = D.archive_function(org_lib, "old")
    assert res["success"] is True
    assert ".archive/old." in res["archived_to"]
    assert not (org_lib / "functions" / "old.py").exists()
    assert "old" not in D.load_dynamic_index(org_lib)
    assert _invoke(t["pux_dyn_list_functions"])["count"] == 0
    archived = list((org_lib / ".archive").glob("old.*.py"))
    assert len(archived) == 1 and archived[0].read_text() == _GOOD  # reversible


def test_archive_missing_errors(org_lib):
    res = D.archive_function(org_lib, "ghost")
    assert res["success"] is False and "does not exist" in res["error"]


# --- promoted runner against REAL python3 (graduation proof; prove, don't assert) --

def test_promoted_runner_executes_against_real_python3(org_lib):
    """A PROMOTED function lives at sandbox/functions/<name>.py with the runner
    cwd = the SANDBOX root. Prove import+run+marker resolves against THAT root
    (not lib) end-to-end against real python3 — the graduation thesis: a
    git-tracked, operator-owned function is callable in-container exactly like an
    agent-authored one, just from the tracked location. Mirrors the lib proof."""
    import os
    import subprocess

    # make_function writes the module + the index entry promote_function keys off
    # (writing the file alone leaves no index entry, so promotion would no-op).
    t = _tools(org_lib)
    _invoke(t["pux_dyn_make_function"], name="add", description="sum", code=_GOOD)
    res = D.promote_function(org_lib, "add")  # lib/functions/add.py -> sandbox/functions/add.py
    assert res["success"] is True
    sb = org_lib.parent / "sandbox"
    runner = D._RUNNER_TEMPLATE.replace("__NAME__", "add")
    env = {
        **os.environ,
        "_PUX_DYN_KWARGS": json.dumps({"nums": [10, 20, 30]}),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        ["python3", "-c", runner], cwd=str(sb), env=env,
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"promoted runner failed: {proc.stderr}"
    assert D._extract_result(proc.stdout) == {"ok": True, "value": 60}
    assert not (sb / "functions" / "__pycache__").exists()
