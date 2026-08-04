"""Hermetic tests for the provider reasoning adapter
(``pux_harness.agent.reasoning``).

Drives ``_convert_chunk_to_generation_chunk`` DIRECTLY with synthetic raw
provider chunk dicts — no network, no real model. This is the unit proof for
the model-side half of reasoning streaming; the wire proof (reasoning →
``AgentThoughtChunk`` over ACP) lives in ``tests/integration/test_acp_e2e.py``.

Mirrors the call convention ``langchain_openai`` uses internally
(``base.py:1617``): ``default_chunk_class=AIMessageChunk``,
``base_generation_info={}``.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessageChunk

from pux_harness.agent.reasoning import (
    ReasoningChatOpenAI,
    _extract_reasoning,
)


def _model() -> ReasoningChatOpenAI:
    # Construction is lazy — no network, no key validation until a request.
    return ReasoningChatOpenAI(
        model="fake-model", base_url="http://fake/v1", api_key="fake"
    )


def _convert(model: ReasoningChatOpenAI, delta: dict, **chunk_extra: object):
    """Run the adapter on a synthetic chunk with the given ``choices[0].delta``
    and return the resulting ``ChatGenerationChunk`` (or ``None``)."""
    chunk = {"choices": [{"delta": delta, **chunk_extra}]}
    return model._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})


# --- shipped extractor shapes ------------------------------------------------


def test_captures_delta_reasoning_content() -> None:
    """The proven shape: DeepSeek / MiMo / OpenRouter stream
    ``delta.reasoning_content``. The adapter preserves it onto the canonical
    ``additional_kwargs["reasoning_content"]`` field."""
    gen = _convert(_model(), {"reasoning_content": "Thinking hard…"})
    assert gen is not None
    assert gen.message.additional_kwargs.get("reasoning_content") == "Thinking hard…"


def test_captures_delta_reasoning() -> None:
    """Second shipped shape: some OpenAI-compat variants use ``delta.reasoning``."""
    gen = _convert(_model(), {"reasoning": "reasoning here…"})
    assert gen is not None
    assert gen.message.additional_kwargs.get("reasoning_content") == "reasoning here…"


def test_both_shapes_concatenate() -> None:
    """The registry runs EVERY extractor; if (unusually) both fire on one chunk
    the results concatenate — proving the registry, not a single hardcoded key."""
    gen = _convert(
        _model(), {"reasoning_content": "A", "reasoning": "B"}
    )
    assert gen is not None
    assert gen.message.additional_kwargs.get("reasoning_content") == "AB"


# --- regression: byte-identical when there's nothing to adapt ----------------


def test_no_reasoning_no_pollution() -> None:
    """A plain content delta MUST NOT add a ``reasoning_content`` key — non-
    reasoning providers stay byte-identical to ``ChatOpenAI`` (no-fallbacks)."""
    gen = _convert(_model(), {"content": "answer"})
    assert gen is not None
    assert "reasoning_content" not in gen.message.additional_kwargs
    assert gen.message.content == "answer"


def test_content_and_reasoning_both_survive() -> None:
    """super()'s content path is intact: a chunk with BOTH content + reasoning
    keeps the content on ``message.content`` AND the reasoning on
    ``additional_kwargs`` — the adapter only ADDS, never replaces."""
    gen = _convert(
        _model(), {"content": "the answer", "reasoning_content": "the thought"}
    )
    assert gen is not None
    assert gen.message.content == "the answer"
    assert gen.message.additional_kwargs.get("reasoning_content") == "the thought"


# --- robustness: malformed chunks must not kill the stream -------------------


@pytest.mark.parametrize(
    "chunk",
    [
        {},
        {"choices": []},
        {"choices": [{"delta": None}]},
        {"choices": [{"delta": {}}]},  # empty delta, no keys
        {"weird": "shape"},  # no choices key at all
    ],
)
def test_malformed_chunks_dont_crash(chunk: dict) -> None:
    """A reasoning extractor must never raise — a malformed chunk (heartbeat,
    empty delta, weird shape) returns gracefully, matching base-class behavior."""
    gen = _model()._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})
    # Either a valid (empty) chunk or None is acceptable — the point is no raise
    # AND no spurious reasoning_content.
    if gen is not None:
        assert "reasoning_content" not in gen.message.additional_kwargs


# --- the registry IS the adapter (the user's explicit ask) -------------------


def test_registry_is_extensible() -> None:
    """Adding a new provider shape is ONE extractor function. A custom extractor
    that reads a hypothetical ``delta.thinking`` block fires through the same
    ``_extract_reasoning`` registry — this is the 'we do need adapters' seam."""
    def _ext_custom_thinking(chunk, _gen):
        delta = (chunk.get("choices") or [{}])[0].get("delta") if chunk.get("choices") else None
        if not isinstance(delta, dict):
            return None
        t = delta.get("thinking")
        return t if isinstance(t, str) and t else None

    rc = _extract_reasoning(
        {"choices": [{"delta": {"thinking": "a new provider shape"}}]},
        gen=None,
        extractors=(_ext_custom_thinking,),
    )
    assert rc == "a new provider shape"
