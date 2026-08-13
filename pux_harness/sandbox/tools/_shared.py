"""Generic helpers for sandbox-executing LangChain tools — ZERO pux imports.

This module is portable alongside the specialist tools: any project with a
deepagents ``BaseSandbox`` (OpenShell, LocalShell, E2B, …) can import from
here and build the same toolset pux uses internally. No ``pux_harness.*``
imports — that coupling lives in ``_pux.py`` (org/skill config) and
``registry.py`` (the contract layer).

Extracted from the original monolithic ``tools.py`` to break circular-dependency
risk: every specialist file imports from ``_shared``; ``_shared`` imports
NOTHING from this package.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from deepagents.backends.sandbox import BaseSandbox

log = logging.getLogger(__name__)


def _exec(
    sandbox: "BaseSandbox", command: str, timeout: int | None = None,
) -> tuple[str, int]:
    """Run ``command`` in ``sandbox``, return ``(output, exit_code)``.

    Bridges deepagents ``BaseSandbox.execute()`` (returns ``ExecuteResponse``)
    to the ``(output, exit_code)`` tuple the specialist tools unpack. Timeouts
    are caught and surfaced as ``exit_code=124`` (matching the legacy
    ``ExecTimeout`` behaviour the tools previously relied on).
    """
    try:
        r = (sandbox.execute(command) if timeout is None
             else sandbox.execute(command, timeout=timeout))
        return r.output, r.exit_code
    except (TimeoutError, subprocess.TimeoutExpired):
        return f"timeout after {timeout}s", 124


def _result(obj: dict) -> str:
    """Serialize a tool-result dict to the exact JSON the Go bridge surfaced
    (2-space indent + sorted map keys at every level)."""
    return json.dumps(obj, indent=2, sort_keys=True)


def _tail(text: str, n: int = 800) -> str:
    """Last ``n`` chars of ``text`` — keeps stderr tails out of result
    envelopes without leaking megabytes. Mirrors the Go ``tailOutput`` helper."""
    return text if len(text) <= n else "..." + text[len(text) - n:]


class _NoArgs(BaseModel):
    """Schema for argument-less tools (list_skills, browser_tabs, etc.)."""
