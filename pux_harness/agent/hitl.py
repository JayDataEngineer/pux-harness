"""Human-in-the-loop ``ask_user`` tool — the one transport-aware HITL surface.

Three transports, one tool (rely-on-upstream + transport-aware design):

* **Web (``serve`` / ``agui``)** — the tool body calls langgraph's native
  ``interrupt()``. The graph pauses; the AG-UI bridge emits ``on_interrupt``;
  CopilotKit's ``useInterrupt`` card renders + calls ``resolve(text)``; that
  text resumes the graph and becomes this tool's return value.
* **Editor (``acp``)** — the tool raises a langgraph ``interrupt`` with an
  ``{"ask_user": ...}`` payload. The ACP server's ``_handle_interrupts`` detects
  it, presents the question (and options, if any) to the client as a chat
  message, then ENDS the turn — mechanically identical to ending a session. The
  interrupt persists in the thread checkpoint; the user's NEXT freeform message
  ("A", "B", or any text) is the resume signal: the server resumes the graph
  with it, and that text becomes this tool's return value. Works on
  Zed/Toad/Hermes TODAY — no ``elicitation`` capability needed; options are
  merely *presented*, the reply is freeform.
* **MCP / autonomous** — the tool is DROPPED at construction (``build_stack``);
  the model can't call a tool that isn't there. No silent no-op.

``direct`` / ``tui`` have no resumable interrupt channel, so the tool poses the
question as its result + a supervisor prompt suffix ends the turn (the user's
next message is the answer, correlated by the agent from context).

One tool, body branches on the transport string. Construction-gated in
``build_stack`` (opt-in via ``profile.yaml`` ``ask_user: true`` AND not
``mcp_active`` AND not ``autonomous``).
"""
from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt

ASK_USER_NAME = "ask_user"

# The supervisor prompt suffix that makes the turn-based (``direct`` / ``tui``)
# path reliable: the agent must STOP after asking, not barrel on. Appended to
# the supervisor system prompt only when ask_user is active AND the transport is
# turn-based (NOT ``acp``/``serve``/``agui``, which halt via a real interrupt).
ASK_USER_PROMPT_SUFFIX = (
    "When you call `ask_user` and it returns a question for the user, you have "
    "posed your question — END your turn immediately and wait for the user's "
    "reply. Do NOT call further tools or continue working until they answer."
)

# Transports that HALT via a resumable langgraph ``interrupt`` (a human is on the
# other end of a stream/session): the web surfaces + the editor. ``direct`` /
# ``tui`` get the turn-based conversational substitute instead.
_RESUMABLE_TRANSPORTS = frozenset({"serve", "agui", "acp"})


def ask_user_turn_based(transport: str) -> bool:
    """``True`` for transports where ask_user poses the question + ends the turn
    WITHOUT a resumable interrupt (``direct`` / ``tui``). ``False`` for
    ``serve`` / ``agui`` / ``acp``, where the tool ``interrupt()``s and the
    reply resumes it. The supervisor prompt suffix (``ASK_USER_PROMPT_SUFFIX``)
    is appended ONLY when this is ``True`` — over a resumable transport the
    interrupt pause already gates the reply, so an "end your turn" instruction
    would be stale (by the time the tool returns, the user HAS replied)."""
    return transport not in _RESUMABLE_TRANSPORTS


def make_ask_user_tool(transport: str) -> BaseTool:
    """Build the ``ask_user`` tool bound to a transport.

    ``transport`` is a plain string (not ``RuntimeFacts``) to keep this module
    import-free of ``stack.py`` (it would otherwise be a cycle: ``stack``
    imports this to build the tool). The construction gate — opt-in + not
    mcp/autonomous — lives in ``build_stack``; by the time this is called the
    tool SHOULD exist.
    """
    is_acp = transport == "acp"
    is_web = transport in ("serve", "agui")

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
        safe fallback if they decline. Over the web / an editor this blocks for
        their reply; over ``direct``/``tui`` it poses the question and pauses
        for the next turn.
        """
        if is_acp:
            # Editor: raise a langgraph interrupt the ACP server detects. It
            # presents the question (+options), ENDS the turn (the interrupt
            # persists in the checkpoint), and the user's NEXT freeform message
            # resumes the graph as this tool's return value. Mechanically: end
            # session -> user signal resumes the thread. No capability probing,
            # no permission/elicitation split — one uniform path.
            reply = interrupt(
                {
                    "ask_user": {
                        "question": question,
                        "options": list(options) if options else [],
                        "default": default,
                    }
                }
            )
            decisions = (reply or {}).get("decisions", []) if isinstance(reply, dict) else []
            for decision in decisions:
                if isinstance(decision, dict) and decision.get("type") == "ask_user":
                    return str(decision.get("answer", ""))
            # No answer injected (e.g. client cancelled): fall back to default.
            return str(default if default is not None else "")
        if is_web:
            # Web: native langgraph interrupt. CopilotKit ``useInterrupt`` over
            # AG-UI renders a card and calls ``resolve(text)``; that text
            # resumes us + becomes our result.
            reply = interrupt(
                {
                    "question": question,
                    "options": list(options) if options else [],
                    "default": default,
                }
            )
            return str(reply)
        # direct / tui: no resumable interrupt. Pose the question; the
        # supervisor prompt suffix ends the turn + the user's next message is
        # the answer.
        opts = " / ".join(options) if options else "(reply with any answer)"
        suffix = f" [default: {default}]" if default else ""
        return (
            f"❓ {question}\n"
            f"Options: {opts}{suffix}\n"
            f"— end your turn and wait for the user's reply."
        )

    return ask_user
