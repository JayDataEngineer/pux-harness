"""Prompt + turn-end capture middleware (context-mode parity gaps 4 + 5).

context-mode (the MCP plugin) captures every user prompt and every assistant
turn-end via its ``UserPromptSubmit`` and ``Stop`` host hooks. The harness had
no equivalent — user prompts and final assistant responses weren't in the
``EventStore``, so a resume-snapshot rebuild lost them and ``ctx_search``
couldn't find them.

This module closes that gap with one ``AgentMiddleware`` using two langgraph
lifecycle hooks:

* ``before_model`` — scan ``state.messages`` for ``HumanMessage``s newer than
  the last seen index for this thread; capture each as a ``user_message`` event.
  This is the UserPromptSubmit equivalent. In a tool-calling loop the user
  message is added ONCE before the first model call, so each prompt is recorded
  exactly once (no re-capture on subsequent iterations).
* ``after_agent`` — scan backward for the last ``AIMessage`` without pending
  ``tool_calls`` (the actual turn-end response, not an intermediate tool-call
  decision); capture it as a ``turn_end`` event. This is the Stop equivalent.

Both hooks thread-scope via ``state.configurable.thread_id`` and track a
per-thread watermark (``self._last_seen``) so a long-lived server handling many
threads doesn't conflate their message indexes.

Why one middleware and not two: ``before_model`` + ``after_agent`` share the
same store, the same thread-id resolver, and the same watermark — splitting them
would duplicate that plumbing. The two event types keep the data cleanly
queryable via ``ctx_search``.

Event shapes:

* ``user_message`` — ``{"content": str}``, priority P2 (user decisions are
  high-signal — same tier as ``decision_made``).
* ``turn_end`` — ``{"content": str, "tool_calls": int}``, priority P3 (audit
  trail; less critical than the user prompt that triggered it).
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from pux_harness.context.events import EventStore, P2, P3, shared_event_store

# Cap stored message content — prompts/responses can be long, and the snapshot
# builder budget is ≤2KB. The full text is recoverable via ctx_search snippets
# only if it's indexed; we store the full preview up to this cap so search hits
# return useful context. Agents that need the verbatim tail should keep it in
# their working window or offload via ctx_index.
_MAX_MSG_CHARS = 2_000


class PromptCaptureMiddleware(AgentMiddleware):
    """Capture user prompts + final assistant turns into the ``EventStore``.

    Set ``enabled=False`` to disable (tests, specific orgs). The store defaults
    to the process-wide shared event store; tests pass their own
    ``EventStore(tmp_path)``."""

    def __init__(
        self,
        store: EventStore | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.store = store or shared_event_store()
        self.enabled = enabled
        # Per-thread message-index watermark: the last index we scanned.
        # Messages at index < watermark are already captured.
        self._last_seen: dict[str, int] = {}

    # -- request introspection (same shape as session_guide + context middleware)

    def _thread_id(self, state: Any) -> str:
        if isinstance(state, dict):
            return state.get("configurable", {}).get("thread_id", "")
        # langgraph state objects expose configurable via attribute too.
        cfg = getattr(state, "configurable", None)
        if isinstance(cfg, dict):
            return cfg.get("thread_id", "")
        return ""

    @staticmethod
    def _messages(state: Any) -> list[Any]:
        if isinstance(state, dict):
            return list(state.get("messages", []))
        msgs = getattr(state, "messages", None)
        return list(msgs) if msgs is not None else []

    @staticmethod
    def _content_str(msg: Any) -> str:
        """Flatten a message's content to a plain string (it may be a list of
        blocks for multimodal messages)."""
        c = getattr(msg, "content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            # Concatenate text blocks; skip image/tool blocks (no useful text).
            parts = []
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return str(c)

    @staticmethod
    def _has_pending_tool_calls(msg: Any) -> bool:
        """True iff this AIMessage is an intermediate tool-call decision (not a
        turn-end response). Tool-call decisions have non-empty ``tool_calls``;
        the final response does not."""
        tc = getattr(msg, "tool_calls", None)
        return bool(tc)

    def _watermark(self, thread_id: str) -> int:
        return self._last_seen.get(thread_id, 0)

    def _bump(self, thread_id: str, idx: int) -> None:
        # Monotonic — only advance forward (a thread reuse shouldn't reset it).
        prev = self._last_seen.get(thread_id, 0)
        if idx > prev:
            self._last_seen[thread_id] = idx

    # -- before_model: UserPromptSubmit equivalent -----------------------------

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Capture any ``HumanMessage``s newer than this thread's watermark."""
        if not self.enabled:
            return None
        thread_id = self._thread_id(state)
        if not thread_id:
            return None  # no thread → can't scope, drop (parity with session_guide)

        msgs = self._messages(state)
        wm = self._watermark(thread_id)
        for i in range(wm, len(msgs)):
            msg = msgs[i]
            if isinstance(msg, HumanMessage):
                content = self._content_str(msg)[:_MAX_MSG_CHARS]
                self.store.capture(
                    "user_message",
                    {"content": content},
                    priority=P2,
                    thread_id=thread_id,
                    category="user",
                )
        self._bump(thread_id, len(msgs))
        self.store.flush()
        return None  # no state mutation — capture-only

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    # -- after_agent: Stop equivalent -----------------------------------------

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Capture the final ``AIMessage`` (no pending tool_calls) as a
        ``turn_end`` event. Scans backward from the newest message."""
        if not self.enabled:
            return None
        thread_id = self._thread_id(state)
        if not thread_id:
            return None

        msgs = self._messages(state)
        for i in range(len(msgs) - 1, max(self._watermark(thread_id) - 1, -1), -1):
            if i < 0:
                break
            msg = msgs[i]
            if isinstance(msg, AIMessage) and not self._has_pending_tool_calls(msg):
                content = self._content_str(msg)[:_MAX_MSG_CHARS]
                self.store.capture(
                    "turn_end",
                    {
                        "content": content,
                        "tool_calls": 0,
                    },
                    priority=P3,
                    thread_id=thread_id,
                    category="assistant",
                )
                self._bump(thread_id, len(msgs))
                self.store.flush()
                return None
        # No final AIMessage found (e.g. agent ended on a tool error) — nothing
        # to capture. Don't bump: a later hook may yet see the response.
        return None

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)
