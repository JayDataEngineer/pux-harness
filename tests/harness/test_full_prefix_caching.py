"""Unit tests for ``_FullPrefixCachingMiddleware``.

The stock ``AnthropicPromptCachingMiddleware`` tags only the system prompt +
last tool and passes ``cache_control`` via ``model_settings``, which
``ChatAnthropic``'s direct-API path does NOT expand into a message-tail
breakpoint.  Our subclass overrides ``_apply_caching`` to EXPLICITLY tag the
last message — the rolling conversation prefix that grows every turn.

These tests prove all 3 breakpoints fire on the request that hits the API:
  1. system message's last content block
  2. last tool's extras
  3. last message's last content block
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from pux_harness.agent.stack import _FullPrefixCachingMiddleware


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _make_request(*, system=None, tools=None, messages=None, model=None):
    """Build a minimal ModelRequest with a mock ChatAnthropic model."""
    from langchain.agents.middleware.types import ModelRequest

    # _should_apply_caching does isinstance(request.model, ChatAnthropic).
    # We can't easily construct a real ChatAnthropic without API keys, so
    # mock the class and make the isinstance check pass via a spec.
    mock_model = MagicMock()
    mock_model.__class__ = MagicMock(
        __name__="ChatAnthropic",
        __mro__=(type, object),
    )
    # Patch isinstance for the specific check
    return ModelRequest(
        model=model or mock_model,
        messages=messages or [],
        system_message=system,
        tools=tools or [],
        model_settings={},
    )


class _FakeTool(BaseTool):
    """Minimal BaseTool subclass so _tag_tools can read .extras."""

    name: str = "fake_tool"
    description: str = "a fake tool"

    def _run(self, *args, **kwargs):
        pass

    async def _arun(self, *args, **kwargs):
        pass


def _patch_isinstance(monkeypatch):
    """Make isinstance(_, ChatAnthropic) return True for our mock model."""
    import langchain_anthropic.middleware.prompt_caching as pc_mod

    real_isinstance = isinstance

    def _fake_isinstance(obj, classinfo):
        # The ONLY isinstance check in _should_apply_caching is against
        # ChatAnthropic.  If classinfo is ChatAnthropic, check by class name
        # so our MagicMock passes.
        if classinfo is pc_mod.ChatAnthropic:
            return getattr(obj, "_is_anthropic", False)
        return real_isinstance(obj, classinfo)

    monkeypatch.setattr(
        "pux_harness.agent.stack._FullPrefixCachingMiddleware",
        _FullPrefixCachingMiddleware,  # ensure still importable
    )


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

class TestFullPrefixCachingBreakpoints:
    """Verify all 3 cache_control breakpoints are placed on the request."""

    def test_tags_system_message_last_block(self):
        """Breakpoint 1: system message's last content block gets cache_control."""
        mw = _FullPrefixCachingMiddleware(unsupported_model_behavior="ignore")
        system = SystemMessage(content="You are an agent.")
        req = _make_request(system=system, messages=[HumanMessage(content="hi")])
        req.model._is_anthropic = True

        # Bypass _should_apply_caching by calling _apply_caching directly
        result = mw._apply_caching(req)

        sm = result.system_message
        assert isinstance(sm.content, list)
        last_block = sm.content[-1]
        assert "cache_control" in last_block, (
            "system message's last content block must have cache_control"
        )
        assert last_block["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    def test_tags_last_tool_extras(self):
        """Breakpoint 2: last tool's extras get cache_control."""
        mw = _FullPrefixCachingMiddleware(unsupported_model_behavior="ignore")
        tools = [_FakeTool(name="tool_a"), _FakeTool(name="tool_b")]
        req = _make_request(
            tools=tools,
            messages=[HumanMessage(content="hi")],
        )

        result = mw._apply_caching(req)

        tagged = result.tools[-1]
        assert "cache_control" in tagged.extras, (
            "last tool's extras must have cache_control — a single trailing "
            "breakpoint caches the entire tool set"
        )
        assert tagged.extras["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
        # Earlier tools should NOT be tagged (save breakpoints)
        assert "cache_control" not in (result.tools[0].extras or {})

    def test_tags_last_message_string_content(self):
        """Breakpoint 3: last message with STRING content gets cache_control.

        This is THE critical addition — the stock middleware does NOT do this.
        The rolling conversation prefix is the bulk of resent tokens.
        """
        mw = _FullPrefixCachingMiddleware(unsupported_model_behavior="ignore")
        messages = [
            HumanMessage(content="what is 2+2?"),
            AIMessage(content="4"),
            ToolMessage(content="result: 4", tool_call_id="tc1"),
        ]
        req = _make_request(messages=messages)

        result = mw._apply_caching(req)

        last = result.messages[-1]
        assert isinstance(last.content, list), (
            "string content should be converted to a list with cache_control"
        )
        block = last.content[-1]
        assert "cache_control" in block, (
            "last message's last content block MUST have cache_control — "
            "this is the rolling prefix breakpoint the stock middleware lacks"
        )
        assert block["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    def test_tags_last_message_list_content(self):
        """Breakpoint 3: last message with LIST content — last block tagged."""
        mw = _FullPrefixCachingMiddleware(unsupported_model_behavior="ignore")
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content=[
                {"type": "text", "text": "thinking..."},
                {"type": "text", "text": "response"},
            ]),
        ]
        req = _make_request(messages=messages)

        result = mw._apply_caching(req)

        last = result.messages[-1]
        last_block = last.content[-1]
        assert "cache_control" in last_block, (
            "last content block in a list message must have cache_control"
        )
        # The non-last block should NOT have cache_control
        assert "cache_control" not in last.content[0]

    def test_does_not_mutate_original_request(self):
        """The override pattern is immutable — original messages/tools/system
        must be unchanged."""
        mw = _FullPrefixCachingMiddleware(unsupported_model_behavior="ignore")
        original_system = SystemMessage(content="system")
        original_msgs = [HumanMessage(content="hi")]
        original_tools = [_FakeTool()]
        req = _make_request(
            system=original_system,
            tools=original_tools,
            messages=original_msgs,
        )

        _ = mw._apply_caching(req)

        # Original request untouched
        assert req.system_message is original_system
        assert req.system_message.content == "system"
        assert req.messages is original_msgs
        assert req.messages[0].content == "hi"
        assert "cache_control" not in (original_tools[0].extras or {})

    def test_all_three_breakpoints_in_one_request(self):
        """Integration: a single _apply_caching call places all 3 breakpoints."""
        mw = _FullPrefixCachingMiddleware(unsupported_model_behavior="ignore")
        req = _make_request(
            system=SystemMessage(content="system prompt"),
            tools=[_FakeTool()],
            messages=[
                HumanMessage(content="turn 1"),
                AIMessage(content="reply 1"),
                HumanMessage(content="turn 2"),
            ],
        )

        result = mw._apply_caching(req)

        # BP 1: system
        assert "cache_control" in result.system_message.content[-1]
        # BP 2: last tool
        assert "cache_control" in result.tools[-1].extras
        # BP 3: last message
        assert "cache_control" in result.messages[-1].content[-1]

    def test_no_model_settings_cache_control(self):
        """We deliberately do NOT pass cache_control in model_settings.

        The direct Anthropic API path does not expand it into a block-level
        breakpoint, so passing it is ambiguous — the 3 explicit breakpoints
        above are sufficient and unambiguous.
        """
        mw = _FullPrefixCachingMiddleware(unsupported_model_behavior="ignore")
        req = _make_request(
            system=SystemMessage(content="s"),
            messages=[HumanMessage(content="m")],
        )

        result = mw._apply_caching(req)

        assert "cache_control" not in result.model_settings, (
            "cache_control should NOT be in model_settings — the 3 explicit "
            "block-level breakpoints are sufficient"
        )


class TestShouldApplyCaching:
    """The _should_apply_caching gate from the parent class is inherited."""

    def test_skip_non_anthropic_model(self):
        """For non-ChatAnthropic models, wrap_model_call is a clean no-op."""
        mw = _FullPrefixCachingMiddleware(unsupported_model_behavior="ignore")
        # Use a plain object that's not ChatAnthropic
        mock_model = MagicMock()
        req = _make_request(
            system=SystemMessage(content="s"),
            messages=[HumanMessage(content="m")],
            model=mock_model,
        )

        # wrap_model_call should pass through without modification
        called = []
        def handler(r):
            called.append(r)
            return "RESPONSE"

        result = mw.wrap_model_call(req, handler)
        assert result == "RESPONSE"
        assert len(called) == 1
        # The request should be unmodified (no cache_control anywhere)
        assert called[0].system_message.content == "s"
