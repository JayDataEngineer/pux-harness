"""Prove the ``pux direct`` recursion handler surfaces partial state.

When a run hits the step cap (``GraphRecursionError``), ``_run`` recovers the
last checkpointed state via ``_recover_partial`` instead of dying on a bare
traceback (#88). This is the verify-or-die belt for that handler: it proves
the CORE langgraph contract the recovery relies on — state is persisted at the
end of every super-step, so ``aget_state`` returns the partial messages after
the cap fires — using a REAL minimal agent + a real in-memory checkpointer
(no stubs on the langgraph side; only the model is scripted).
"""
from __future__ import annotations

import asyncio

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError


class _LoopingModel(BaseChatModel):
    """Always calls ``loop_tool`` — forces an infinite model→tool loop that
    exhausts the recursion limit (a faithful stand-in for a runaway gather)."""

    @property
    def _llm_type(self) -> str:
        return "looping"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001,ANN002,ANN003
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001,ANN002,ANN003
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "loop_tool", "args": {}, "id": "c"}],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def loop_tool() -> str:
    """A no-op tool the looping model calls forever."""
    return "again"


def _looping_agent() -> object:
    return create_agent(
        model=_LoopingModel(),
        tools=[loop_tool],
        checkpointer=InMemorySaver(),
    )


def test_recover_partial_returns_checkpointed_messages():
    """After ``GraphRecursionError``, ``_recover_partial`` returns the messages
    checkpointed before the cap — not an empty list. This is the langgraph
    persistence contract the handler relies on; proving it with a real agent +
    checkpointer is the belt beyond correct-by-inspection."""
    from pux_harness.main import _recover_partial

    agent = _looping_agent()
    config = {"configurable": {"thread_id": "recursion-test"}, "recursion_limit": 4}
    with pytest.raises(GraphRecursionError):
        asyncio.run(
            agent.ainvoke(
                {"messages": [{"role": "user", "content": "loop forever"}]},
                config=config,
            )
        )

    messages, todos = asyncio.run(_recover_partial(agent, config))
    # The user message + at least one model/tool round-trip were checkpointed
    # before the cap fired — recovery surfaces them, not empties.
    assert messages, "expected checkpointed partial messages after recursion cap"
    assert len(messages) >= 2
    # This agent has no todo state; todos degrades to None, not a crash.
    assert todos is None


def test_recover_partial_handles_empty_state():
    """A thread that was never invoked yields no checkpoint. ``_recover_partial``
    must return ``( [], None )`` — the direct lane must not IndexError on an
    empty partial (the bug the empty-messages guard in ``_run`` closes)."""
    from pux_harness.main import _recover_partial

    agent = _looping_agent()
    config = {"configurable": {"thread_id": "never-run"}, "recursion_limit": 4}
    messages, todos = asyncio.run(_recover_partial(agent, config))
    assert messages == []
    assert todos is None
