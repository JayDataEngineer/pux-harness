"""Prepare/warmup middleware — the serve-lane owner of the pre-agent ``prepare()`` seam.

Why this exists
---------------
``prepare()`` (``sandbox/container.py``) runs an org's declared policy
``jobs:`` (e.g. ``warmup_browser`` — force Chrome CDP attach so the first real
browser tool call doesn't absorb the cold-start) and, in serve-class
transports, the universal ``warmup_webhook`` probe (prove the run-completion
event endpoint is reachable from this sandbox before the agent loop). It is
warn-and-continue: a failed job is logged, never blocks the run.

Historically the two HTTP/CLI entry points called it themselves:

* ``main.py`` (``pux direct``) — ``prepare(org, exec_client=shared_exec())``
  before ``agent.ainvoke()``.
* ``server.py`` (``pux serve`` fallback) — ``prepare(org, universal_warmup=True)``
  offloaded to a worker thread inside ``POST /runs``.

Aegra (pux's prod Agent Protocol runtime — a langgraph-api/LangGraph-Platform
drop-in) owns the run loop itself: there is NO pux entry point between
"receive run" and "invoke the graph", so neither of those two call sites is
reached. Without a hook, ``warmup_browser`` / ``warmup_webhook`` silently stop
firing under Aegra (known cutover delta). This middleware is the fix: it runs
``prepare()`` from the graph's own ``before_agent`` hook, which Aegra (and any
langgraph-api runtime) DOES drive. See ``[[aegra-prod-cutover-shipped]]``,
``[[browser-warmup]]``, ``[[run-event-stream]]``.

Single owner, no double-fire
----------------------------
The middleware is GATED on ``RuntimeFacts.prepare_warmup`` (default False).
Only the Aegra runtime factory (``runtime/upstream.py``) sets it True. The
``direct`` and ``server.py`` lanes leave it False and keep their OWN explicit
``prepare()`` call — so each runtime has exactly ONE prepare source and no run
double-fires. Tests build the graph with the default ``RuntimeFacts()`` → the
middleware no-ops (``_build_prepare`` returns ``None``) → Docker is never
touched from a test invoke.

``universal_warmup`` is gated on ``transport != "direct"`` to match the
historical lane behavior: serve/Aegra/acp/tui/mcp probe the run-completion
endpoint; ``direct`` does not (no serve up in direct mode — the probe would
retry ~15s before failing). See ``orgs/_shared/sandbox/warmup_webhook.py``.

Why ``abefore_agent`` + ``asyncio.to_thread``
---------------------------------------------
``prepare()`` does blocking Docker I/O (``ensure`` container + ``exec``
warmup scripts). The prod path (Aegra / langgraph-api) streams runs
asynchronously, so the hook runs as ``abefore_agent``. Calling ``prepare()``
inline would stall the SINGLE event loop — during which a webhook-less client's
``GET /events`` poll could time out and mistake the runtime for dead (the
exact reason ``server.py`` offloaded with ``asyncio.to_thread``). Offloading
keeps the loop serving ``/events``, ``/ok``, and new runs while prep runs in
parallel.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

_log = logging.getLogger(__name__)


class PrepareWarmupMiddleware(AgentMiddleware):
    """Run ``prepare()`` once at agent start (the ``before_agent`` hook).

    Construct with the org name and whether to run the universal
    ``warmup_webhook`` probe (serve-class transports only). The hook itself is
    best-effort: any exception from ``prepare()`` is swallowed so a prep
    failure can NEVER break the agent run — matching ``prepare()``'s own
    warn-and-continue contract for the jobs it runs internally.
    """

    def __init__(self, org: str, *, universal_warmup: bool) -> None:
        self.org = org
        self.universal_warmup = universal_warmup

    def _run_prepare(self) -> None:
        """One ``prepare()`` call; never raises into the agent run."""
        from pux_harness.sandbox.container import prepare  # noqa: PLC0415

        try:
            results = prepare(self.org, universal_warmup=self.universal_warmup)
        except Exception:  # noqa: BLE001 — prep must never block the agent run
            _log.warning(
                "prepare(%s) raised; continuing to agent", self.org, exc_info=True
            )
            return

        if not results:
            return
        failed = [r for r in results if r.get("status") != "ok"]
        summary = f"{len(results)} job(s)" + (
            f" ({len(failed)} failed)" if failed else " — all ok"
        )
        (_log.warning if failed else _log.info)("prepare(%s): %s", self.org, summary)
        for r in failed:
            _log.warning(
                "prepare job %s: %s — %s",
                r.get("name"),
                r.get("status"),
                (r.get("error") or "")[:160],
            )

    # -- hooks ----------------------------------------------------------------
    # Parameter NAMES are load-bearing. langgraph's ``RunnableCallable`` (which
    # langchain's agent factory wraps every ``before_agent`` hook in) passes
    # ``state`` as the single positional arg and injects ``runtime`` as a KEYWORD
    # arg, detected by EXACT parameter name — it introspects the signature and
    # only fills params named ``runtime``/``config``/``error`` (see
    # ``langgraph/_internal/_runnable.py`` ``RunnableCallable.ainvoke`` +
    # ``func_accepts``). Naming these ``_state``/``_runtime`` (underscores) makes
    # the framework skip the runtime injection → ``TypeError: missing 1 required
    # positional argument`` at agent start (caught live under Aegra). Match the
    # abstract ``AgentMiddleware`` signature + ``SessionGuideMiddleware`` exactly.

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync path (rare — graph runs stream async). No loop to stall."""
        self._run_prepare()
        return None

    async def abefore_agent(
        self, state: Any, runtime: Any
    ) -> dict[str, Any] | None:
        """Async path (the prod one: Aegra / server.py stream).

        Offload the blocking Docker I/O to a worker thread so the event loop
        keeps serving /events polls + new runs while prep runs in parallel.
        """
        await asyncio.to_thread(self._run_prepare)
        return None
