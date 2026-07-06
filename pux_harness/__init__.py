"""pux-harness: deepagents-based Pux orchestration.

The slim, Docker-free org+skill compiler is a first-class surface:
``compile_org`` is re-exported here so a standalone consumer (e.g. a Wan2GP +
CopilotKit app) does ``from pux_harness import compile_org`` and gets the
portable core — the kit pulls neither ``docker`` nor any heavy
``pux_harness.<subsystem>`` (sandbox/browser/context/server) at import time.
"""
from __future__ import annotations

from pux_harness.kit import compile_org

__all__ = ["compile_org"]
