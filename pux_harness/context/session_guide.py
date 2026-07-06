"""Cross-session rehydration via Session Guide (Phase 12).

When a thread resumes, this middleware:
1. Queries recent events from the EventStore
2. Builds a structured snapshot via ``snapshot.build_snapshot()``
3. Stores it in the ``session_resume`` table
4. On next session start, claims the latest unconsumed snapshot
5. Injects it as a ``<session_knowledge>`` directive in the system prompt

The agent immediately knows what files were being edited, what tasks were
in progress, what errors occurred, and what decisions were made — without
re-reading the full message history.

Modeled after mksglu/context-mode's SessionStart + PreCompact lifecycle:
- PreCompact (our wrap_model_call detecting compaction) → build + store snapshot
- Session Start (our before_agent) → claim + inject snapshot
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import SystemMessage

from pux_harness.context.events import EventStore, shared_event_store
from pux_harness.context.snapshot import build_snapshot

# Tag used to inject the session guide into the system prompt.
_SESSION_KNOWLEDGE_TAG = "session_knowledge"


class SessionGuideMiddleware(AgentMiddleware):
    """Build and inject session guide for cross-session rehydration.

    Lifecycle:
    - ``before_agent``: if a resume snapshot exists, inject it into the
      system prompt as a ``<session_knowledge>`` directive.
    - ``wrap_model_call``: detect compaction (message count drop) and
      build + store a new snapshot from recent events.

    Set ``enabled=False`` to disable.
    """

    def __init__(
        self,
        store: EventStore | None = None,
        *,
        enabled: bool = True,
        search_tool: str = "ctx_search",
    ) -> None:
        self.store = store or shared_event_store()
        self.enabled = enabled
        self.search_tool = search_tool
        self._injected_session: str | None = None

    def _thread_id(self, state: Any) -> str:
        if isinstance(state, dict):
            return state.get("configurable", {}).get("thread_id", "")
        return ""

    def _build_and_store(self, thread_id: str) -> str | None:
        """Build a snapshot from recent events and store it. Returns the snapshot XML."""
        if not thread_id:
            return None

        events = self.store.recent(thread_id=thread_id, limit=200)
        if not events:
            return None

        snapshot_xml = build_snapshot(
            events,
            thread_id=thread_id,
            search_tool=self.search_tool,
        )
        self.store.upsert_resume(thread_id, snapshot_xml, len(events))
        return snapshot_xml

    def _claim_and_inject(self, thread_id: str) -> str | None:
        """Claim the latest unconsumed resume snapshot. Returns the XML or None."""
        if not thread_id:
            return None

        row = self.store.claim_latest_unconsumed_resume(exclude_session=thread_id)
        if row is None:
            return None

        self._injected_session = row.get("session_id", "")
        return row.get("snapshot", "")

    # -- before_agent: inject on session start/resume --------------------------

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Inject session guide if a resume snapshot is available."""
        if not self.enabled:
            return None

        thread_id = self._thread_id(state)
        if not thread_id:
            return None

        snapshot = self._claim_and_inject(thread_id)
        if not snapshot:
            return None

        # Build the injection directive.
        directive = (
            f"\n\n<{_SESSION_KNOWLEDGE_TAG}>\n"
            f"Previous session state (injected on resume):\n"
            f"{snapshot}\n"
            f"Use ctx_search to retrieve full details for any section.\n"
            f"</{_SESSION_KNOWLEDGE_TAG}>"
        )

        # Append to the system prompt in state.
        # State may be a dict (LangGraph) or an object with messages.
        if isinstance(state, dict):
            messages = state.get("messages", [])
            if messages and isinstance(messages[0], SystemMessage):
                # Append to existing system message.
                existing = messages[0].content
                if isinstance(existing, str):
                    messages[0] = SystemMessage(content=existing + directive)
                else:
                    # Content is a list of blocks — append a text block.
                    blocks = list(existing) if isinstance(existing, list) else [existing]
                    blocks.append({"type": "text", "text": directive})
                    messages[0] = SystemMessage(content=blocks)
            else:
                # Prepend a new system message.
                messages.insert(0, SystemMessage(content=directive.lstrip()))
            return {"messages": messages}

        return None

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    # -- wrap_model_call: build snapshot on compaction --------------------------

    def wrap_model_call(self, request, handler):  # type: ignore[no-untyped-def]
        """Detect compaction by monitoring message count, build snapshot."""
        if not self.enabled:
            return handler(request)

        # Store the current message count so we can detect compaction next turn.
        state = getattr(request, "state", None)
        thread_id = self._thread_id(state) if state else ""

        # Build snapshot on every model call (cheap — just reads recent events).
        # The snapshot is upserted, so only the latest matters.
        if thread_id:
            self._build_and_store(thread_id)

        return handler(request)

    async def awrap_model_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled:
            return await handler(request)

        state = getattr(request, "state", None)
        thread_id = self._thread_id(state) if state else ""

        if thread_id:
            self._build_and_store(thread_id)

        return await handler(request)
