"""InterpreterHintsMiddleware — when the ``eval`` tool (CodeInterpreterMiddleware's
sandboxed QuickJS JS REPL) returns an ``<error type="...">`` result, append a
companion ``HumanMessage`` that classifies the failure, explains the root cause,
and names the fallback so a strong orchestrator can either fix the script or
abandon ``eval`` and drive the individual tools directly.

THE PROBLEM
    ``CodeInterpreterMiddleware`` (langchain-quickjs) is a pure tool-injector —
    it exposes an ``eval`` tool whose result is a plain-text XML-ish string:
        success:  ``<result kind="...">value</result>``
        error:    ``<error type="SyntaxError">message\\nstack</error>``
    On failure the model gets the raw error string with NO guidance on how to
    recover. A strong orchestrator that writes a bad dispatch script (syntax
    error, timeout, OOM, exceeded the PTC call budget) sees an opaque
    ``<error>`` block and often either (a) retries the SAME broken script,
    burning tokens, or (b) gives up on the whole task instead of falling back
    to the simpler per-tool calling mode that always works.

THE FIX (same pattern as BrowserVisionMiddleware)
    A ``wrap_tool_call`` middleware that detects ``<error type="...">`` in an
    ``eval`` ToolMessage result and emits a ``Command([text_tm, human_hint])``
    — the text ToolMessage keeps its tool_call_id so the reducer still pairs it
    with the pending tool call; the companion HumanMessage carries:

        [eval failed — <classification>]
        Why: <root-cause hint>
        ErrorType: <type>
        Fix the script and re-call eval, OR skip eval and call the tools
        directly: glob, grep, ls, read_file, task.

    So the model knows WHAT broke, WHY, and that it has a clean escape hatch.

WHY NOT tool_retry
    ``tool_retry`` retries the SAME tool call on transient failures. eval
    failures are NOT transient — a syntax error or a logic bug produces the
    identical error on the next call. The model needs to FIX the script,
    which requires knowing WHAT broke. This middleware provides exactly that
    diagnostic; blind retry would waste a round for nothing.

ZERO HAPPY-PATH COST
    The middleware does ONE regex scan (``<error type=``) on ``eval`` results
    only — every other tool name short-circuits before the scan, and eval
    SUCCESS results pass through untouched (no hint, no Command, no overhead).

THE 7 ERROR TYPES (from langchain-quickjs ``_repl.py``)
    ``SyntaxError``            — JS parse error (e.name from quickjs)
    ``Timeout``                — 5s wall-clock exceeded
    ``OutOfMemory``            — 64 MiB heap exhausted
    ``PTCCallBudgetExceeded``  — too many glob/grep/ls/read_file calls in one eval
    ``Deadlock``               — top-level Promise never resolved
    ``ConcurrentEval``         — overlapping evals on the same context (prompting bug)
    runtime types              — TypeError/ReferenceError/RangeError (e.error_type from quickjs)
    Plus ``MarshalError`` is surfaced as a JS-level error when the return value
    can't be converted to a Python value (circular ref, non-serializable handle).
"""
from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

# The eval tool name — CodeInterpreterMiddleware defaults to ``"eval"`` and our
# ``_build_interpreter`` does not override it.
_EVAL_TOOL_NAME = "eval"

# PTC tools the interpreter exposes inside eval (glob/grep/ls/read_file) plus
# the ``task`` global. These are the fallback the hint steers toward.
_DEFAULT_FALLBACK_TOOLS = ["glob", "grep", "ls", "read_file", "task"]

# One regex scan to detect an error block. Captures the ``type`` attribute.
# Compiled once at import; DOTALL so a multi-line type attribute wouldn't
# defeat it (the upstream format never spans lines, but be safe).
_ERROR_RE = re.compile(r'<error\s+type="([^"]+)"', re.DOTALL)


# --- error classification table -------------------------------------------------
# Maps the ``error_type`` string from the eval result to a (classification, hint)
# pair. Runtime JS errors (TypeError, ReferenceError, RangeError, ...) are caught
# by the fallthrough entry — they carry the JS exception name as the type.
def _classify(error_type: str) -> tuple[str, str]:
    et = error_type.strip()
    if et == "SyntaxError":
        return (
            "JS syntax error",
            "The script has invalid JS syntax. Common gotcha: `return {a: 1}` "
            "is parsed as a block + label, not an object — wrap in parens: "
            "`return ({a: 1})`. Check for unclosed brackets, bad commas, "
            "unmatched template literals.",
        )
    if et == "Timeout":
        return (
            "execution timeout (5s budget exceeded)",
            "The script ran past the wall-clock budget — likely an infinite "
            "loop or a very slow operation. Reduce iterations, add a break "
            "condition, or split into smaller eval calls.",
        )
    if et == "OutOfMemory":
        return (
            "heap exhausted (64 MiB limit)",
            "The script exceeded the QuickJS heap — likely accumulating a "
            "large array or string. Stream/chunk results instead of building "
            "one giant structure; narrow the glob before reading.",
        )
    if et == "PTCCallBudgetExceeded":
        return (
            "too many PTC tool calls",
            "The script called glob/grep/ls/read_file past the call budget. "
            "Batch with Promise.all, narrow globs before reading, or do fewer "
            "round-trips. Each task() dispatch also counts toward the budget.",
        )
    if et == "Deadlock":
        return (
            "Promise never resolved",
            "A top-level Promise never settled — likely a missing `await` or "
            "an unresolved task() call. Ensure every async branch resolves.",
        )
    if et == "ConcurrentEval":
        return (
            "overlapping evals",
            "Two evals ran on the same REPL context simultaneously. This is a "
            "prompting bug — serialize: run one eval at a time per thread.",
        )
    if et in ("MarshalError",):
        return (
            "unserializable return value",
            "The return value couldn't be converted to a plain value (circular "
            "reference, non-serializable handle). Return a plain object, a "
            "primitive, or JSON.stringify it before returning.",
        )
    # Runtime JS exceptions: TypeError, ReferenceError, RangeError, SyntaxError
    # variants, etc. The type IS the JS error name — surface it verbatim.
    return (
        f"JS runtime exception ({et})",
        f"A {et} was thrown at runtime. Common causes: null/undefined property "
        f"access, wrong argument type, index out of bounds. The error message "
        f"above names the failing expression.",
    )


def _build_hint(error_type: str, fallback_tools: list[str]) -> str:
    classification, hint = _classify(error_type)
    fallback = ", ".join(fallback_tools)
    return (
        f"[eval failed — {classification}]\n"
        f"Why: {hint}\n"
        f"ErrorType: {error_type}\n"
        f"Fix the script and re-call eval, OR skip eval and call the tools "
        f"directly: {fallback}."
    )


def _enrich_eval_result(
    result: Any,
    *,
    fallback_tools: list[str],
) -> Any:
    """Return ``Command([text_tm, human_hint])`` iff ``result`` is an ``eval``
    ToolMessage carrying an ``<error type="...">`` block. Otherwise return
    ``result`` unchanged (success or non-eval — zero overhead)."""
    if not isinstance(result, ToolMessage):
        return result
    if result.name != _EVAL_TOOL_NAME:
        return result
    content = result.content if isinstance(result.content, str) else ""
    if not content:
        return result
    m = _ERROR_RE.search(content)
    if not m:
        return result  # success — no hint needed
    error_type = m.group(1)
    human = HumanMessage(content=[{"type": "text", "text": _build_hint(
        error_type, fallback_tools
    )}])
    return Command(update={"messages": [result, human]})


class InterpreterHintsMiddleware(AgentMiddleware):
    """Append a classified-failure hint + fallback as a companion HumanMessage
    after an ``eval`` tool error. See module docstring for the full rationale."""

    def __init__(
        self,
        *,
        fallback_tools: list[str] | None = None,
    ) -> None:
        self._fallback_tools = (
            list(fallback_tools) if fallback_tools else list(_DEFAULT_FALLBACK_TOOLS)
        )

    @staticmethod
    def _is_eval_call(request: Any) -> bool:
        tc = getattr(request, "tool_call", None) or {}
        name = tc.get("name") if isinstance(tc, dict) else None
        return name == _EVAL_TOOL_NAME

    def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self._is_eval_call(request):
            return handler(request)  # not eval — no scan
        return _enrich_eval_result(
            handler(request),
            fallback_tools=self._fallback_tools,
        )

    async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self._is_eval_call(request):
            return await handler(request)
        result = await handler(request)
        return _enrich_eval_result(
            result,
            fallback_tools=self._fallback_tools,
        )
