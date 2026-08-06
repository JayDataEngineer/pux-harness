"""Full-prefix Anthropic prompt caching — caches every resent token.

Extends upstream ``AnthropicPromptCachingMiddleware`` with a third explicit
``cache_control`` breakpoint on the last message's final content block, so the
rolling conversation history (the bulk of resent tokens) is cached across turns
— not just the static system prompt + tools.
"""
from __future__ import annotations

from typing import Any

from langchain_anthropic.middleware.prompt_caching import (
    AnthropicPromptCachingMiddleware,
    _tag_system_message,
    _tag_tools,
)


class FullPrefixCachingMiddleware(AnthropicPromptCachingMiddleware):
    """Caches EVERY resent token — system prompt, tools, AND the rolling
    conversation history.

    Stock ``AnthropicPromptCachingMiddleware`` tags the system prompt (breakpoint
    1) and last tool (breakpoint 2), then passes ``cache_control`` in
    ``model_settings`` expecting the transport to expand it into a third
    breakpoint on the message tail.  BUT ``ChatAnthropic``'s DIRECT API path
    does NOT expand that kwarg — the growing message history was never cached.

    This subclass overrides ``_apply_caching`` to EXPLICITLY tag the last
    message's final content block with ``cache_control`` (breakpoint 3):

    1.  System prompt last content block  — static prefix
    2.  Last tool definition              — static tools
    3.  Last message last content block   — rolling prefix (all prior turns)

    On turn N+1 the prefix up to turn N's last message is a cache READ (90%
    discount); only the 2 newest messages are at full price.  3 of Anthropic's
    4 allowed breakpoints — well within limits.

    No-op for non-``ChatAnthropic`` models (``_should_apply_caching`` returns
    ``False``).
    """

    def _apply_caching(self, request: Any) -> Any:  # type: ignore[name-defined]  # noqa: F821
        overrides: dict[str, Any] = {}
        cc = self._cache_control

        # Breakpoint 1: system message's last content block.
        system_message = _tag_system_message(request.system_message, cc)
        if system_message is not request.system_message:
            overrides["system_message"] = system_message

        # Breakpoint 2: last tool definition.
        tools = _tag_tools(request.tools, cc)
        if tools is not request.tools:
            overrides["tools"] = tools

        # Breakpoint 3: last message's last content block — THE addition that
        # caches the rolling conversation history.
        messages = list(request.messages)
        if messages:
            last = messages[-1]
            content = last.content
            if isinstance(content, str):
                if content:
                    messages[-1] = last.model_copy(
                        update={"content": [{"type": "text", "text": content, "cache_control": cc}]}
                    )
                    overrides["messages"] = messages
            elif isinstance(content, list) and content:
                new_content = list(content)
                last_block = new_content[-1]
                base = last_block if isinstance(last_block, dict) else {}
                new_content[-1] = {**base, "cache_control": cc}
                messages[-1] = last.model_copy(update={"content": new_content})
                overrides["messages"] = messages

        return request.override(**overrides)
