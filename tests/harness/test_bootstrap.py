"""Shared process bootstrap — ``pux_harness.kit.bootstrap_env_and_logging``.

This is the SEAM that makes pux seamless in a foreign codebase: a consumer
drops ``./.env`` next to its code, runs pux, and its keys land in
``os.environ`` without an ``export``. It is vendored with the slim kit into
every export, so the SAME function serves ``pux serve``/``pux direct``/
``pux acp`` AND an exported runner. These tests pin its two contracts:

1. **``.env`` load** — launch-CWD-anchored (``find_dotenv(usecwd=True)``); an
   existing shell export is NEVER overridden (``override=False`` default); no
   ``.env`` is a no-op, not an error.
2. **stderr logging pin** — opt-in (``pin_stderr=True``); when off, the root
   logger is left untouched (HTTP entrypoints keep their own log config).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from pux_harness.kit import bootstrap_env_and_logging


def test_dotenv_loaded_into_environ(tmp_path: Path, monkeypatch) -> None:
    # Foreign-codebase launch: the key is in ./.env, NOT in the shell env.
    monkeypatch.delenv("PUX_BOOT_TEST_KEY", raising=False)
    (tmp_path / ".env").write_text("PUX_BOOT_TEST_KEY=from-dotenv-file\n")
    monkeypatch.chdir(tmp_path)

    assert "PUX_BOOT_TEST_KEY" not in os.environ  # precondition
    bootstrap_env_and_logging()
    assert os.environ["PUX_BOOT_TEST_KEY"] == "from-dotenv-file"


def test_dotenv_does_not_override_existing_export(tmp_path: Path, monkeypatch) -> None:
    # A real shell export WINS over .env (no silent clobber) — the documented
    # load_dotenv(override=False) default, pinned so an explicit setting holds.
    monkeypatch.setenv("PUX_BOOT_TEST_KEY", "from-shell-export")
    (tmp_path / ".env").write_text("PUX_BOOT_TEST_KEY=from-dotenv-file\n")
    monkeypatch.chdir(tmp_path)

    bootstrap_env_and_logging()
    assert os.environ["PUX_BOOT_TEST_KEY"] == "from-shell-export"


def test_no_dotenv_is_noop_not_error(tmp_path: Path, monkeypatch) -> None:
    # A CWD with no .env (e.g. a sandbox/CI dir) must not raise — pux still runs.
    monkeypatch.delenv("PUX_BOOT_TEST_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / ".env").exists()

    bootstrap_env_and_logging()  # no .env, no crash
    assert "PUX_BOOT_TEST_KEY" not in os.environ


def test_pin_stderr_true_strips_stdout_binds_stderr() -> None:
    # ACP path: a library may have auto-configured a stdout handler; pin_stderr
    # re-binds the root logger to stderr (stdout IS the JSON-RPC wire).
    root = logging.getLogger()
    root.addHandler(logging.StreamHandler(sys.stdout))
    try:
        bootstrap_env_and_logging(pin_stderr=True)
        streams = [getattr(h, "stream", None) for h in root.handlers]
        assert sys.stdout not in streams
        assert sys.stderr in streams
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)


def test_pin_stderr_false_leaves_logging_untouched() -> None:
    # HTTP path (serve/direct): pin_stderr=False must NOT reconfigure logging —
    # uvicorn keeps its own handlers. A pre-seeded stdout handler survives.
    root = logging.getLogger()
    seeded = logging.StreamHandler(sys.stdout)
    root.addHandler(seeded)
    try:
        bootstrap_env_and_logging(pin_stderr=False)
        assert seeded in root.handlers  # untouched
    finally:
        root.removeHandler(seeded)


def test_idempotent_double_call(tmp_path: Path, monkeypatch) -> None:
    # cli.main() calls bootstrap(pin_stderr=False); ``pux acp`` then calls it
    # again with pin_stderr=True. The second call must not raise or double-load.
    monkeypatch.delenv("PUX_BOOT_TEST_KEY", raising=False)
    (tmp_path / ".env").write_text("PUX_BOOT_TEST_KEY=once\n")
    monkeypatch.chdir(tmp_path)

    bootstrap_env_and_logging()
    bootstrap_env_and_logging(pin_stderr=True)  # second call is harmless
    assert os.environ["PUX_BOOT_TEST_KEY"] == "once"
