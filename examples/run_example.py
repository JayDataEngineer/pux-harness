"""Standalone demo: compile + run the Wan2GP-shaped org with the slim kit core.

Run from anywhere::

    python examples/run_example.py

This file is a self-contained "how to use the kit from a different project":
it imports ONLY the slim ``pux_harness`` kit core (``compile_org``) — no Docker,
no sandbox, no server, no context/memory stack — and needs no real API key (the
supervisor model is a tiny scripted stub). A real app swaps the scripted model
for an LLM and supplies its own tools (e.g. a Wan2GP driver).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field, PrivateAttr

from pux_harness import compile_org

EXAMPLE_ROOT = Path(__file__).resolve().parent


# --- the app's own tool (the agent's real surface) -------------------------

class _FormArgs(BaseModel):
    prompt: str = Field(..., description="the text prompt")
    width: int = 512
    height: int = 512
    steps: int = 20


def _generate_form(prompt: str, width: int = 512, height: int = 512, steps: int = 20) -> str:
    """A stub for the host app's Wan2GP driver — returns the resolved form."""
    return (
        f"Wan2GP form: prompt={prompt!r} size={width}x{height} steps={steps}"
    )


generate_form = StructuredTool.from_function(
    _generate_form, name="generate_form", args_schema=_FormArgs,
    description="Emit a Wan2GP generation form from resolved parameters.",
)


# --- a scripted supervisor model (no API key needed) -----------------------

class _StubModel(BaseChatModel):
    """Calls ``generate_form`` once, then ends the turn.

    A real app passes a ``ChatOpenAI``/``ChatAnthropic`` model here; the scripted
    stub just makes this demo runnable offline. ``bind_tools`` returns ``self``
    because the canned response ignores the bound schema.
    """

    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "stub"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_StubModel":
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        self._calls += 1
        if self._calls == 1:
            msg = AIMessage(
                content="",
                tool_calls=[{
                    "name": "generate_form",
                    "args": {"prompt": "a neon koi fish, cinematic", "width": 1024, "steps": 35},
                    "id": "call-1",
                }],
            )
        else:
            msg = AIMessage(content="Done — the form above is ready to drive Wan2GP.")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def main() -> None:
    graph = compile_org(
        "wan2gp_demo",
        model=_StubModel(),
        tools=[generate_form],
        project_root=EXAMPLE_ROOT,
        checkpointer=MemorySaver(),
    )
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "make me a 1024 detail render of a neon koi"}]},
        config={"configurable": {"thread_id": "demo"}},
    )
    for msg in result["messages"]:
        role = getattr(msg, "type", getattr(msg, "role", "?"))
        content = getattr(msg, "content", "")
        print(f"[{role}] {content}")


if __name__ == "__main__":
    main()
