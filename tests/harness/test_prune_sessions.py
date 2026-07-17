"""Prove pux_harness.sandbox.prune bounds session history correctly."""
import json, os, sys, time, tempfile, pathlib
from datetime import datetime, timezone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from pux_harness.sandbox.prune import prune_pux_dir


def _ts(days_ago: float) -> str:
    return datetime.fromtimestamp(
        time.time() - days_ago * 86400, tz=timezone.utc
    ).isoformat().replace("+00:00", "Z")


def _make_tree(root: pathlib.Path):
    sessions = root / "sessions"; sessions.mkdir(parents=True)
    wild = root / "wild-logs"; wild.mkdir()
    # OLD files (30 days) — should be swept
    old_meta = sessions / "old.meta.json"; old_meta.write_text("{}")
    old_wild = wild / "old.log"; old_wild.write_text("x")
    _setmtime(old_meta, 30); _setmtime(old_wild, 30)
    # NEW files (1 day) — must be kept
    new_meta = sessions / "new.meta.json"; new_meta.write_text("{}")
    _setmtime(new_meta, 1)
    # run_events.jsonl: 3 old lines (30d) + 1 new line (1d)
    ev = root / "run_events.jsonl"
    ev.write_text("\n".join([
        json.dumps({"run_id": "a", "ts": _ts(30)}),
        json.dumps({"run_id": "b", "ts": _ts(30)}),
        json.dumps({"run_id": "c", "ts": _ts(30)}),
        json.dumps({"run_id": "d", "ts": _ts(1)}),
    ]) + "\n")
    return ev


def _setmtime(p, days_ago):
    ts = time.time() - days_ago * 86400
    os.utime(p, (ts, ts))


def test_dry_run_changes_nothing_but_reports():
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td); ev = _make_tree(root)
        before = ev.read_text()
        report = prune_pux_dir(root, days=14, dry_run=True)
        assert report["dry_run"] is True
        assert len(report["meta"]["swept"]) == 1          # old.meta.json
        assert report["meta"]["kept"] == 1                 # new.meta.json
        assert report["events"]["removed_lines"] == 3
        assert report["events"]["kept_lines"] == 1
        assert len(report["wildlogs"]["swept"]) == 1
        # Nothing actually deleted:
        assert (root / "sessions" / "old.meta.json").exists()
        assert ev.read_text() == before


def test_apply_sweeps_old_keeps_new():
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td); ev = _make_tree(root)
        report = prune_pux_dir(root, days=14, dry_run=False)
        assert not (root / "sessions" / "old.meta.json").exists()
        assert (root / "sessions" / "new.meta.json").exists()
        assert not (root / "wild-logs" / "old.log").exists()
        # JSONL rewritten: only the 1 new line survives
        kept = [json.loads(l) for l in ev.read_text().splitlines() if l.strip()]
        assert len(kept) == 1
        assert kept[0]["run_id"] == "d"


def test_days_zero_sweeps_everything():
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td); _make_tree(root)
        report = prune_pux_dir(root, days=0, dry_run=False)
        assert report["meta"]["kept"] == 0
        assert report["events"]["kept_lines"] == 0


def test_missing_dirs_are_handled_cleanly():
    with tempfile.TemporaryDirectory() as td:
        report = prune_pux_dir(pathlib.Path(td), days=14, dry_run=False)
        assert report["meta"]["swept"] == []
        assert report["events"]["removed_lines"] == 0
        assert report["wildlogs"]["swept"] == []


def test_events_line_without_ts_falls_back_to_file_mtime():
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td); _make_tree(root)
        ev = root / "run_events.jsonl"
        # Append a line with no ts; set its container (the file) to old.
        # The whole file is new (just written), so to test the fallback we
        # rewrite with a single no-ts line and age the file.
        ev.write_text(json.dumps({"run_id": "no-ts"}) + "\n")
        _setmtime(ev, 30)
        report = prune_pux_dir(root, days=14, dry_run=False)
        assert report["events"]["removed_lines"] == 1  # aged out via file mtime
        assert report["events"]["kept_lines"] == 0


def test_categories_scoped():
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td); _make_tree(root)
        report = prune_pux_dir(root, days=14, dry_run=False, categories=("meta",))
        assert "events" not in report and "wildlogs" not in report
        assert (root / "wild-logs" / "old.log").exists()  # untouched
