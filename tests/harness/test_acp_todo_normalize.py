"""Lock in the _normalize_todos stall fix.

The deterministic stall in .pux/stall.log (session a0af08…, org coder) was
GLM-5.2 emitting write_todos with bare-string entries → the base handler's
``todo.get("content")`` crashed with AttributeError mid-stream, killing the
turn. _normalize_todos coerces every shape the model improvises so the base
handler only ever sees dicts. If a refactor drops the override, this test
goes red."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from pux_harness.acp import _normalize_todos


def test_exact_crash_input_bare_strings():
    """The literal shape that crashed stall.log: list of bare strings."""
    out = _normalize_todos(["write design doc", "review code"])
    assert out == [
        {"content": "write design doc", "status": "pending"},
        {"content": "review code", "status": "pending"},
    ]
    # The exact access pattern that crashed (server.py:446) must now survive:
    for todo in out:
        assert isinstance(todo, dict)
        _ = todo.get("content", "")  # would have raised AttributeError before


def test_well_formed_dicts_pass_through_unchanged():
    out = _normalize_todos([{"content": "x", "status": "in_progress"}])
    assert out == [{"content": "x", "status": "in_progress"}]


def test_dict_missing_content_gets_empty_string():
    out = _normalize_todos([{"status": "completed"}])
    assert out == [{"status": "completed", "content": ""}]


def test_non_dict_non_str_entries_are_stringified():
    # Models have emitted None / ints / floats inside the todos list.
    out = _normalize_todos([None, 42, 3.14])
    assert out == [
        {"content": "None", "status": "pending"},
        {"content": "42", "status": "pending"},
        {"content": "3.14", "status": "pending"},
    ]


def test_top_level_non_list_is_salvaged_not_crashed():
    # Schema violation: the whole payload is a string, not a list.
    out = _normalize_todos("just do the thing")
    assert out == [{"content": "just do the thing", "status": "pending"}]


def test_none_is_empty_list():
    assert _normalize_todos(None) == []


def test_mixed_shapes_in_one_payload():
    out = _normalize_todos(["bare string", {"content": "dict"}, {"status": "pending"}, 7])
    assert all(isinstance(t, dict) and "content" in t for t in out)
    assert out[0]["content"] == "bare string"
    assert out[1] == {"content": "dict"}
    assert out[2] == {"status": "pending", "content": ""}
    assert out[3]["content"] == "7"


def test_base_handler_access_pattern_survives_on_every_output():
    """Every output entry MUST support .get('content', '') — the exact line
    that crashed. Run against a representative mixed payload."""
    out = _normalize_todos(["s", {"content": "d"}, {"status": "x"}, None, 1, "t"])
    for todo in out:
        assert isinstance(todo.get("content", ""), str)
        assert isinstance(todo.get("status", "pending"), str)
