"""Provider reasoning-content adapter (model-side).

``langchain_openai==1.3.x`` deliberately drops non-standard streaming fields
(``reasoning_content``, ``reasoning``, …) — its module docstring says so. But
many providers (DeepSeek, MiMo, OpenRouter, OpenCode Zen Go, …) stream the
model's reasoning as a DISTINCT SSE delta field BEFORE the answer content.
Without this adapter that reasoning is silently discarded the instant it enters
LangChain, so no downstream surface (notably the ACP ``AgentThoughtChunk`` path
in :mod:`pux_harness.acp`) can ever see it.

``ReasoningChatOpenAI`` subclasses :class:`~langchain_openai.ChatOpenAI` and
overrides ``_convert_chunk_to_generation_chunk`` — which receives the RAW
provider chunk dict before the lossy ``_convert_delta_to_message_chunk`` runs —
to run each registered extractor and accumulate the result onto the one
canonical field ``message.additional_kwargs["reasoning_content"]``. For chunks
that carry no reasoning, behavior is byte-identical to ``ChatOpenAI``
(super() does all the standard work; we only ADD reasoning when an extractor
finds it) — so non-reasoning providers and non-streaming consumers are
unaffected (no-fallbacks: no behavior change where there's nothing to adapt).

Providers stream reasoning in DIFFERENT shapes, so the per-shape logic lives in
a registry of extractor functions (``_REASONING_EXTRACTORS``). Adding a new
provider shape = one extractor function + append to the tuple. Shipped now:

* ``_ext_delta_reasoning_content`` — ``delta.reasoning_content``
  (DeepSeek / MiMo / OpenRouter / **OpenCode Zen Go — the harness's actual
  endpoint**, proven live against ``mimo-v2.5``).
* ``_ext_delta_reasoning`` — ``delta.reasoning`` (some OpenAI-compat variants;
  trivial: same delta lookup, different key).

NOT shipped (documented extension points — no org routes through these today;
every role in ``models.yaml`` resolves to ``mimo-v2.5`` via OpenCode Go, so
shipping them would be speculative per no-fallbacks / no-aliases):

* Anthropic-native ``{"type":"thinking","thinking":…}`` content blocks.
* OpenAI-Responses ``{"type":"reasoning",…}`` content blocks.

When an org pins such a model, add an extractor that reads the block off
``gen.message.content`` (a list of block dicts) — same registry, one function.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_openai import ChatOpenAI

# A reasoning extractor: given the raw provider chunk dict + the LangChain
# GenerationChunk produced by super(), return the reasoning text fragment for
# this chunk (or ``None``/``""`` if this shape carries no reasoning here).
# Exceptions are swallowed by ``_extract_reasoning`` so a buggy extractor can
# never break the stream.
ReasoningExtractor = Callable[[dict[str, Any], Any], "str | None"]


def _delta(chunk: dict[str, Any]) -> dict[str, Any] | None:
    """Best-effort ``choices[0].delta`` lookup; ``None`` on any shape mismatch
    (non-dict chunk, missing ``choices``, empty list, …)."""
    try:
        choices = chunk["choices"]
    except (KeyError, TypeError):
        return None
    if not isinstance(choices, list) or not choices:
        return None
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    return delta if isinstance(delta, dict) else None


def _ext_delta_reasoning_content(chunk: dict[str, Any], _gen: Any) -> str | None:
    """DeepSeek / MiMo / OpenRouter / OpenCode Zen Go: ``delta.reasoning_content``."""
    delta = _delta(chunk)
    if delta is None:
        return None
    rc = delta.get("reasoning_content")
    return rc if isinstance(rc, str) and rc else None


def _ext_delta_reasoning(chunk: dict[str, Any], _gen: Any) -> str | None:
    """Some OpenAI-compat variants: ``delta.reasoning``."""
    delta = _delta(chunk)
    if delta is None:
        return None
    rc = delta.get("reasoning")
    return rc if isinstance(rc, str) and rc else None


# The registry. New provider shape → new extractor → append. Extractors are
# independent (each guards its own shape), so order only affects concatenation
# order when (unusually) two shapes fire on the same chunk.
_REASONING_EXTRACTORS: tuple[ReasoningExtractor, ...] = (
    _ext_delta_reasoning_content,
    _ext_delta_reasoning,
)


def _extract_reasoning(
    chunk: dict[str, Any],
    gen: Any,
    extractors: Sequence[ReasoningExtractor] = _REASONING_EXTRACTORS,
) -> str:
    """Run every registered extractor; concatenate the non-empty results."""
    parts: list[str] = []
    for ext in extractors:
        try:
            rc = ext(chunk, gen)
        except Exception:  # noqa: BLE001 — a buggy extractor must not kill the stream
            rc = None
        if rc:
            parts.append(rc)
    return "".join(parts)


class ReasoningChatOpenAI(ChatOpenAI):
    """:class:`~langchain_openai.ChatOpenAI` that preserves provider reasoning
    content the base class discards.

    Reasoning from N provider shapes is normalized into the one canonical field
    ``message.additional_kwargs["reasoning_content"]``. Non-reasoning chunks are
    byte-identical to ``ChatOpenAI``: ``super()`` performs all standard
    conversion (content, tool_calls, usage), and we only touch the message when
    an extractor actually finds reasoning. Drop-in for ``get_model`` — same
    constructor surface, no new required args.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: Any,
        base_generation_info: Any,
    ):
        gen = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if gen is None:
            return None
        rc = _extract_reasoning(chunk, gen)
        if rc:
            ak = gen.message.additional_kwargs
            ak["reasoning_content"] = (ak.get("reasoning_content") or "") + rc
        return gen
