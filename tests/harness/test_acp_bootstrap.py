"""ACP stdio bootstrap — the ``.env`` + stderr-logging contract for ``pux acp``.

The editor launches ``pux acp`` in a shell that has NOT sourced the user's key
exports; the API key lives in ``./.env``. Two guarantees ``run_acp`` must make
BEFORE building the graph (which reads the key off ``os.environ[api_key_env]``):

1. **``.env`` is loaded** — a key written to a project ``.env`` lands in
   ``os.environ`` so ``agent.model.get_model`` doesn't ``KeyError``; an existing
   shell export is NEVER overridden (no silent clobber).
2. **Logging is pinned to stderr** — stdout is the JSON-RPC wire format, so the
   root logger's handlers must all stream to ``sys.stderr`` (``force=True``
   strips any pre-seeded stdout handler a library may have added).

A wiring test pins that ``run_acp`` calls the bootstrap BEFORE anything that
reads env / opens the server (the helper is the single chokepoint).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from pux_harness import acp


def test_dotenv_loaded_into_environ(tmp_path: Path, monkeypatch) -> None:
    # Editor-style launch: the key is in ./.env, NOT in the shell environment.
    monkeypatch.delenv("PUX_ACP_TEST_KEY", raising=False)
    (tmp_path / ".env").write_text("PUX_ACP_TEST_KEY=from-dotenv-file\n")
    monkeypatch.chdir(tmp_path)

    assert "PUX_ACP_TEST_KEY" not in os.environ  # precondition
    acp._bootstrap_env_and_logging()
    assert os.environ["PUX_ACP_TEST_KEY"] == "from-dotenv-file"


def test_dotenv_does_not_override_existing_export(tmp_path: Path, monkeypatch) -> None:
    # A real shell export WINS over .env (no silent override) — the documented
    # load_dotenv(override=False) default, pinned so serve/direct semantics hold.
    monkeypatch.setenv("PUX_ACP_TEST_KEY", "from-shell-export")
    (tmp_path / ".env").write_text("PUX_ACP_TEST_KEY=from-dotenv-file\n")
    monkeypatch.chdir(tmp_path)

    acp._bootstrap_env_and_logging()
    assert os.environ["PUX_ACP_TEST_KEY"] == "from-shell-export"


def test_logging_pinned_to_stderr() -> None:
    # Pre-seed a stdout handler (simulating a lib that auto-configured one) and
    # prove force=True strips it + binds the root logger to stderr. stdout is
    # the ACP wire format — no log handler may stream to it.
    root = logging.getLogger()
    root.addHandler(logging.StreamHandler(sys.stdout))
    try:
        acp._bootstrap_env_and_logging()
        streams = [getattr(h, "stream", None) for h in root.handlers]
        assert sys.stdout not in streams
        assert sys.stderr in streams
    finally:
        # basicConfig(force=True) replaced the seeded handler; reset for siblings.
        for h in list(root.handlers):
            root.removeHandler(h)


def test_run_acp_invokes_bootstrap_before_server(monkeypatch) -> None:
    # Wiring: run_acp must call the bootstrap, then discover_orgs, then the
    # async server — in that order. Proves the helper isn't dead code without
    # standing up a live stdio server (no real env / thread store needed).
    calls: list[str] = []

    def _fake_bootstrap() -> None:
        calls.append("bootstrap")

    def _fake_run(coro) -> None:
        calls.append("asyncio.run")
        coro.close()  # never awaited — close so pytest doesn't warn

    monkeypatch.setattr(acp, "_bootstrap_env_and_logging", _fake_bootstrap)
    monkeypatch.setattr(acp, "discover_orgs", lambda: {"general"})
    monkeypatch.setattr(acp, "asyncio", SimpleNamespace(run=_fake_run))

    acp.run_acp("general")
    assert calls[0] == "bootstrap"  # before any env read / server build
    assert calls == ["bootstrap", "asyncio.run"]
