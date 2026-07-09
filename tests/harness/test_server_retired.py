"""Permanent contract failure: ``pux_harness.server`` is RETIRED (Aegra phase D).

``no-legacy-left-behind``: once Aegra (the OSS langgraph-api / LangGraph-Platform
drop-in) was PROVEN to serve the full Agent Protocol surface — the parity gate
in ``tests/upstream/`` AND live prod (``server.py`` was already DISABLED,
``pux-aegra.service`` reboot-safe on :9988, cloud E2E green) — the hand-rolled
``pux_harness.server`` runtime was DELETED. Aegra is the single AP runtime owner.

These tests make any re-introduction of ``server.py`` (or the ``pux serve``
subcommand) FAIL the suite. The old form is a PERMANENT contract failure, not a
dormant fallback: there is one AP runtime (Aegra), no ``pux serve``, no fallback
launcher. ``server.py`` stays recoverable only via git history (``git show``).

See ``docs/AEGRA_PROD.md`` (phase D), ``[[server-py-retired]]``,
``[[aegra-prod-cutover-shipped]]``, ``[[no-legacy-left-behind]]``.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys


def test_server_module_retired() -> None:
    """``pux_harness.server`` must NOT be importable — Aegra owns the AP lane.

    ``find_spec`` does not execute the module; it returns ``None`` when no such
    module exists. If someone re-adds ``server.py``, this asserts non-None and
    fails loud."""
    spec = importlib.util.find_spec("pux_harness.server")
    assert spec is None, (
        "pux_harness.server was RETIRED in Aegra phase D (single-owner: Aegra "
        "serves the Agent Protocol surface natively). Re-introducing the "
        "hand-rolled AP runtime violates no-legacy-left-behind. See "
        "docs/AEGRA_PROD.md."
    )


def test_serve_subcommand_retired() -> None:
    """``pux serve`` must no longer be a subcommand.

    The Agent Protocol HTTP server is now Aegra (prod:
    ``scripts/start_pux_aegra.sh``) or ``langgraph dev`` / ``aegra dev`` (dev) —
    NOT a ``pux`` verb. ``pux`` only builds orgs in-process (acp / tui / direct);
    client verbs (dispatch / run / ...) talk to whatever AP server is up."""
    r = subprocess.run(
        [sys.executable, "-m", "pux_harness", "serve"],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0, "`pux serve` should be rejected (retired)"
    assert "invalid choice" in r.stderr, (
        "`pux serve` must be rejected by argparse as an unknown subcommand. "
        f"stderr: {r.stderr!r}"
    )


def test_serve_not_in_help_subcommands() -> None:
    """Belt-and-suspenders: ``serve`` is absent from the registered subcommand
    set printed by ``--help`` (catches a re-add that somehow didn't route through
    argparse's invalid-choice path)."""
    r = subprocess.run(
        [sys.executable, "-m", "pux_harness", "--help"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "serve" not in _subcommand_set(r.stdout), (
        "`serve` must not appear in the pux subcommand list. "
        f"help: {r.stdout!r}"
    )


def _subcommand_set(help_stdout: str) -> set[str]:
    """The registered subcommands, parsed from argparse's ``{a,b,c}`` group."""
    import re  # noqa: PLC0415

    match = re.search(r"\{([^}]*)\}", help_stdout)
    return set(match.group(1).split(",")) if match else set()
