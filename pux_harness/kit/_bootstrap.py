"""Process bootstrap shared by every pux entrypoint AND by exported runners.

This is the seam that makes pux **seamless in a foreign codebase**: a consumer
app (or an exported org runner) drops a ``.env`` next to its code, runs pux, and
its keys are picked up WITHOUT the user having to ``export`` them into the shell
that launched the process. It lives in the slim kit (not the heavy runtime) so
it is vendored into every export — the runner emitted for an exported org calls
this at its top, and ``pux serve``/``pux direct``/``pux acp`` call it from their
own entrypoints. One function, every consumer.

Two contracts:

1. **Load ``./.env``.** ``agent.model.get_model`` reads the provider key straight
   off ``os.environ[api_key_env]`` (a hard ``KeyError`` with no fallback).
   Editors/CI/embedded apps launch pux WITHOUT sourcing the user's shell exports,
   so a key that ``export``-shells see lives only in ``./.env`` here. We anchor
   the search on the launch CWD via ``find_dotenv(usecwd=True)`` — the bare
   ``load_dotenv()`` default uses ``usecwd=False``, which searches upward from
   THIS module's source file and would find pux's own repo ``.env``, NOT the
   foreign project's. ``override=False`` (the default) means a real shell export
   wins, so this never clobbers an explicit setting.

2. **Pin logging to stderr** (only when ``pin_stderr=True`` — the ACP stdio path,
   where stdout IS the JSON-RPC wire and a stray log line corrupts the stream).
   ``force=True`` re-binds the root logger before deepagents/langchain/acp can
   auto-configure a stdout handler. HTTP entrypoints (serve/agui) pass
   ``pin_stderr=False`` so uvicorn keeps its own log config.

Idempotent: ``load_dotenv`` never overrides an already-set var and
``basicConfig(force=True)`` re-binds cleanly, so ``pux acp`` calling this again
after ``cli.main`` already did (with ``pin_stderr=False``) is harmless — the
second call just adds the stderr pin.
"""
from __future__ import annotations

import logging
import sys

from dotenv import find_dotenv, load_dotenv


def bootstrap_env_and_logging(*, pin_stderr: bool = False) -> None:
    """Load ``./.env`` (launch-CWD-anchored); optionally pin logging to stderr.

    Call as the FIRST thing an entrypoint does, before any env read. ``usecwd``
    is load-bearing — see the module docstring.
    """
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    if pin_stderr:
        logging.basicConfig(stream=sys.stderr, force=True)
