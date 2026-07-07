"""Human-in-the-loop ``ask_user`` tool — the one transport-aware HITL surface.

Three transports, one tool (rely-on-upstream + transport-aware design):

* **Web (``serve`` / ``agui``)** — the tool body calls langgraph's native
  ``interrupt()``. The graph pauses; the AG-UI bridge emits ``on_interrupt``;
  CopilotKit's ``useInterrupt`` card renders + calls ``resolve(text)``; that
  text resumes the graph and becomes this tool's return value. Proven to pause
  + resume inside a deepagents tool body (``/tmp/spike_interrupt_in_tool.py``).
* **Editor (``acp`` / ``direct`` / ``tui``)** — the editor's permission popover
  has no free-text input, so an interrupt the client can't resume would dead-
  end. Instead the tool poses the question as its result + a supervisor prompt
  suffix tells the agent to END its turn; the user's next ``session/prompt`` is
  the answer (the turn-based conversational substitute).
* **MCP / autonomous** — the tool is DROPPED at construction (``build_stack``);
  the model can't call a tool that isn't there. No silent no-op.

One tool, body branches on the transport string. Construction-gated in
``build_stack`` (opt-in via ``profile.yaml`` ``ask_user: true`` AND not
``mcp_active`` AND not ``autonomous``).
"""
from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt

ASK_USER_NAME = "ask_user"

# The supervisor prompt suffix that makes the editor (turn-based) path reliable:
# the agent must STOP after asking, not barrel on. Appended to the supervisor
# system prompt only when ask_user is active for this org.
ASK_USER_PROMPT_SUFFIX = (
    "When you call `ask_user` and it returns a question for the user, you have "
    "posed your question — END your turn immediately and wait for the user's "
    "reply. Do NOT call further tools or continue working until they answer."
)

# Transports that can RESUME a paused graph (a human is on the other end of a
# stream): the web surfaces. Everything else gets the turn-based conversational
# substitute (the editor permission popover has no free-text field).
_INTERRUPT_TRANSPORTS = frozenset({"serve", "agui"})


def ask_user_turn_based(transport: str) -> bool:
    """``True`` for transports where ask_user poses the question + ends the turn
    (acp / direct / tui / …). ``False`` for the web surfaces (``serve`` /
    ``agui``), where the tool ``interrupt()``s and the reply resumes it. The
    supervisor prompt suffix (``ASK_USER_PROMPT_SUFFIX``) is appended ONLY when
    this is ``True`` — over the web the interrupt pause already gates the reply,
    so an "end your turn" instruction would be stale (by the time the tool
    returns, the user HAS replied)."""
    return transport not in _INTERRUPT_TRANSPORTS


def make_ask_user_tool(transport: str) -> BaseTool:
    """Build the ``ask_user`` tool bound to a transport.

    ``transport`` is a plain string (not ``RuntimeFacts``) to keep this module
    import-free of ``stack.py`` (it would otherwise be a cycle: ``stack``
    imports this to build the tool). The construction gate — opt-in + not
    mcp/autonomous — lives in ``build_stack``; by the time this is called the
    tool SHOULD exist.
    """
    use_interrupt = transport in _INTERRUPT_TRANSPORTS

    @tool
    def ask_user(
        question: str,
        options: list[str] | None = None,
        default: str | None = None,
    ) -> str:
        """Ask the human a question and wait for their reply.

        Use this whenever you need a decision, a clarification, or a choice only
        the user can make before you can continue. ``options`` is the set of
        valid answers (offer them when the choice is bounded); ``default`` is the
        safe fallback if they decline. Over the web this blocks for their reply;
        over an editor it poses the question and pauses for the next turn.
        """
        if use_interrupt:
            # Native langgraph interrupt: pause the graph. The web client
            # (CopilotKit ``useInterrupt`` over AG-UI) renders a card and calls
            # ``resolve(text)``; that text resumes us + becomes our result.
            reply = interrupt(
                {
                    "question": question,
                    "options": list(options) if options else [],
                    "default": default,
                }
            )
            return str(reply)
        # Editor / direct / tui: no resumable free-text interrupt. Pose the
        # question; the supervisor prompt suffix ends the turn + the user's
        # next message is the answer.
        opts = " / ".join(options) if options else "(reply with any answer)"
        suffix = f" [default: {default}]" if default else ""
        return (
            f"❓ {question}\n"
            f"Options: {opts}{suffix}\n"
            f"— end your turn and wait for the user's reply."
        )

    return ask_user
