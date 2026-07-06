"""The unified context-saving middleware.

One ``AgentMiddleware`` that does BOTH jobs in a single ``wrap_tool_call``
pass — so every tool result is read exactly once, timed once, and written
through one store:

1. **Capture** — record a structured ``tool_call`` (or ``error``) event:
   tool name, truncated args, success, elapsed seconds, and a 300-char
   output preview. These events feed the snapshot builder (cross-session
   rehydration) and the ``ctx_search`` retrieval surface.
2. **Offload** — when a tool returns more than ``threshold`` chars of plain
   text, park the FULL bytes in the store as a blob and replace the
   ``ToolMessage`` the model sees with a short preview + a ``ctx:<id>``
   retrieval handle. This is *proactive*: it keeps large results out of the
   context window BEFORE they accumulate, complementing deepagents' reactive
   ``SummarizationMiddleware`` (which only evicts on overflow).

This merges the old ``ContextOffloadMiddleware`` + ``EventCaptureMiddleware``
(two passes, two stores) into one. It rides the
MAIN agent AND every subagent — ``build_context_layer()`` is the single seam
both ``graph.py`` and ``orgs._build_sub`` import, so the whole agent tree
gets capture + offload + retrieval with zero duplication.

Retrieval tools (``ctx_recall`` / ``ctx_search``) are **exempt** from both
jobs: their purpose is to inject content, so re-stashing their output would
trap the agent the instant it retrieves a large stash (proven in E2E
before the exemption), and their output needs no event preview.
"""
from __future__ import annotations

import json
import time
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from pux_harness.context.events import EventStore, StashResult, shared_event_store

# Defaults: ~2k tokens (chars/4). Tuned to catch directory dumps, big greps,
# log tails, and JSON blobs without trimming ordinary tool replies.
DEFAULT_THRESHOLD = 8000
DEFAULT_PREVIEW = 1500

# The retrieval surface — ctx_recall/ctx_search exist to bring (a slice of)
# stashed content back INTO context. Offloading OR preview-logging their
# output would defeat that (and re-stash a big recall instantly). So they
# are exempt from BOTH capture-detail and offload.
_RETRIEVAL_TOOLS = frozenset({"ctx_recall", "ctx_search"})


def _is_text_tm(result: Any) -> bool:
    """Only offload a real ToolMessage with plain-string content. Multimodal
    content (lists of blocks), Command returns, and non-ToolMessage results pass
    through untouched — offloading structured/image output would lose it."""
    return isinstance(result, ToolMessage) and isinstance(result.content, str)


def _stub(stash: StashResult, content: str, tool: str, *, preview: int) -> str:
    """The replacement message the model sees instead of the full blob.

    Phrased as a plain TRUNCATED TOOL RESULT with a one-line continuation
    pointer — deliberately NOT a system notice: no ``[bracketed]`` banner, no
    "offload/stashed/context" jargon, no rationale. A skeptical live model
    reads ``[ctx-offload] ... stashed so it doesn't crowd the working context``
    as a management artifact and second-guesses whether later content is
    "real" (observed: MiMo-2.5 transcribed a unique tail correctly but then
    argued it was an injected context-management marker). Phrased as a normal
    truncation, the model just calls ``ctx_recall`` for the rest — the right
    action is the path of least resistance, no reasoning required.

    The ``ctx:<id>`` handle in the pointer IS the structural offload signal
    (a non-truncated tool result never carries one); tests key on it rather
    than on any phrasing."""
    head = content[:preview]
    ellipsis = "…" if len(content) > preview else ""
    return (
        f"{tool} returned {stash.chars} chars (first {preview} shown). "
        f"For the complete output, call ctx_recall({stash.handle!r}).\n\n"
        f"{head}{ellipsis}"
    )


def _offload(
    result: Any, store: EventStore, tool_name: str, *,
    threshold: int, preview: int, thread_id: str = "",
) -> Any:
    """If ``result`` is an oversized text ToolMessage, stash + replace. Else
    return unchanged. Pure (no I/O beyond the store); called by both the sync
    and async wrap hooks so behavior is identical either way.

    ``threshold <= 0`` is a kill-switch: offload nothing (handy for tests + a
    future env-flag to disable the feature without unwiring the middleware).

    Retrieval tools (``ctx_recall``/``ctx_search``) are exempt — see
    ``_RETRIEVAL_TOOLS``."""
    if (
        threshold <= 0
        or tool_name in _RETRIEVAL_TOOLS
        or not _is_text_tm(result)
        or len(result.content) <= threshold
    ):
        return result
    stash = store.stash_blob(result.content, tool=tool_name, thread_id=thread_id)
    return ToolMessage(
        content=_stub(stash, result.content, tool_name, preview=preview),
        tool_call_id=result.tool_call_id,
        name=result.name,
    )


class ContextMiddleware(AgentMiddleware):
    """Capture every tool call as an event AND offload oversized results to a
    blob — one ``wrap_tool_call`` pass, one store.

    Knobs:
    - ``threshold`` (default 8000 chars): results this size or larger are
      stashed behind a ``ctx:<id>`` handle. Set ``<= 0`` to disable offload.
    - ``preview`` (default 1500 chars): how much of a stashed result to keep
      inline in the stub message.
    - ``enabled`` (default True): master switch — when False, capture AND
      offload are both skipped (the handler runs bare).
    """

    def __init__(
        self,
        store: EventStore | None = None,
        *,
        threshold: int = DEFAULT_THRESHOLD,
        preview: int = DEFAULT_PREVIEW,
        enabled: bool = True,
    ) -> None:
        self.store = store or shared_event_store()
        self.threshold = threshold
        self.preview = preview
        self.enabled = enabled

    # -- request introspection (works for langchain's ToolCallRequest or a
    # SimpleNamespace stand-in in tests) -----------------------------------

    def _tool_name(self, request: Any) -> str:
        tc = getattr(request, "tool_call", None) or {}
        if isinstance(tc, dict):
            name = tc.get("name")
            return str(name) if name is not None else "tool"
        return "tool"

    def _tool_args_summary(self, request: Any) -> str:
        tc = getattr(request, "tool_call", None) or {}
        args = tc.get("args", {}) if isinstance(tc, dict) else {}
        raw = json.dumps(args, ensure_ascii=False, default=str)
        return raw[:500] if len(raw) > 500 else raw

    def _thread_id(self, request: Any) -> str:
        state = getattr(request, "state", None) or {}
        if isinstance(state, dict):
            return state.get("configurable", {}).get("thread_id", "")
        return ""

    def _capture(
        self, *, tool: str, args_summary: str, thread_id: str,
        result: Any, elapsed: float,
    ) -> None:
        """Record the tool_call event with a 300-char output preview (skipped
        for retrieval tools + error-status results)."""
        success = True
        output_preview = ""
        if isinstance(result, ToolMessage):
            content = result.content if isinstance(result.content, str) else ""
            success = result.status != "error"
            if tool not in _RETRIEVAL_TOOLS:
                output_preview = content[:300] if content else ""
        self.store.capture(
            "tool_call",
            {
                "tool": tool,
                "args": args_summary,
                "success": success,
                "elapsed_s": round(elapsed, 3),
                "output_preview": output_preview,
            },
            thread_id=thread_id,
        )

    # -- sync ------------------------------------------------------------------

    def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled:
            return handler(request)

        tool = self._tool_name(request)
        args_summary = self._tool_args_summary(request)
        thread_id = self._thread_id(request)
        t0 = time.time()

        try:
            result = handler(request)
        except Exception as exc:
            elapsed = time.time() - t0
            self.store.capture(
                "error",
                {
                    "tool": tool,
                    "args": args_summary,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "elapsed_s": round(elapsed, 3),
                },
                thread_id=thread_id,
            )
            self.store.flush()
            raise

        elapsed = time.time() - t0
        # Capture the event from the ORIGINAL result (what the tool actually
        # returned) BEFORE offload swaps the message the model will see — the
        # event is a record of activity, the offload is a separate concern.
        self._capture(
            tool=tool, args_summary=args_summary, thread_id=thread_id,
            result=result, elapsed=elapsed,
        )
        result = _offload(
            result, self.store, tool,
            threshold=self.threshold, preview=self.preview, thread_id=thread_id,
        )
        self.store.flush()
        return result

    # -- async (the production path — the server/runner use ainvoke) ----------

    async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled:
            return await handler(request)

        tool = self._tool_name(request)
        args_summary = self._tool_args_summary(request)
        thread_id = self._thread_id(request)
        t0 = time.time()

        try:
            result = await handler(request)
        except Exception as exc:
            elapsed = time.time() - t0
            self.store.capture(
                "error",
                {
                    "tool": tool,
                    "args": args_summary,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "elapsed_s": round(elapsed, 3),
                },
                thread_id=thread_id,
            )
            self.store.flush()
            raise

        elapsed = time.time() - t0
        self._capture(
            tool=tool, args_summary=args_summary, thread_id=thread_id,
            result=result, elapsed=elapsed,
        )
        result = _offload(
            result, self.store, tool,
            threshold=self.threshold, preview=self.preview, thread_id=thread_id,
        )
        self.store.flush()
        return result
