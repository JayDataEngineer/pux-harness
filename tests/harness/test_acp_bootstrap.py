"""ACP wiring — ``run_acp`` calls the SHARED bootstrap with the stderr pin.

The ``.env``-load + stderr-pin contracts themselves live on the shared helper
(``pux_harness.kit.bootstrap_env_and_logging``) and are pinned in
``tests/harness/test_bootstrap.py``. What is ACP-specific — and what this file
pins — is the WIRING: ``run_acp`` must invoke that helper with
``pin_stderr=True`` (stdout is the JSON-RPC wire) BEFORE any env read / server
build. Proves the helper isn't dead code on the stdio path without standing up
a live server.
"""
from __future__ import annotations

from types import SimpleNamespace

from pux_harness import acp


def test_run_acp_invokes_bootstrap_with_stderr_pin_before_server(monkeypatch) -> None:
    calls: list[tuple] = []

    def _fake_bootstrap(*, pin_stderr: bool = False) -> None:
        calls.append(("bootstrap", pin_stderr))

    def _fake_run(coro) -> None:
        calls.append(("asyncio.run", None))
        coro.close()  # never awaited — close so pytest doesn't warn

    monkeypatch.setattr(acp, "bootstrap_env_and_logging", _fake_bootstrap)
    monkeypatch.setattr(acp, "discover_orgs", lambda: {"general"})
    monkeypatch.setattr(acp, "asyncio", SimpleNamespace(run=_fake_run))

    acp.run_acp("general")

    # First call is the bootstrap, WITH the stderr pin, before the async server.
    assert calls[0] == ("bootstrap", True)
    assert calls == [("bootstrap", True), ("asyncio.run", None)]
