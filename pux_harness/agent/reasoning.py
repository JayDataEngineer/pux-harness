"""Provider reasoning-content adapter (model-side).

``langchain_openai==1.3.x`` deliberately drops non-standard streaming fields
(``reasoning_content``, ``reasoning``, …) — its module docstring says so. But
many providers (DeepSeek, MiMo, OpenRouter, OpenRouter, …) stream the
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
  (DeepSeek / MiMo / OpenRouter / **OpenRouter — the harness's actual
  endpoint**, proven live against ``mimo-v2.5``).
* ``_ext_delta_reasoning`` — ``delta.reasoning`` (some OpenAI-compat variants;
  trivial: same delta lookup, different key).

NOT shipped (documented extension points — no org routes through these today;
every role in ``models.yaml`` resolves to ``mimo-v2.5`` via OpenRouter, so
shipping them would be speculative per no-fallbacks / no-aliases):

* Anthropic-native ``{"type":"thinking","thinking":…}`` content blocks.
* OpenAI-Responses ``{"type":"reasoning",…}`` content blocks.

When an org pins such a model, add an extractor that reads the block off
``gen.message.content`` (a list of block dicts) — same registry, one function.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from langchain_openai import ChatOpenAI

_log = logging.getLogger("pux.model.reasoning")


def _trace(model: str, outcome: str, t0: float, attempts: int,
           chunks: int, idle_s: float, dur_s: float) -> None:
    """Append a one-line per-stream trace to ``<project>/.pux/stream-trace.log``.

    Always-on, non-intrusive (a single append write). Captures the data a
    post-hoc stall diagnosis needs: which model, TTFT-vs-post-output, how many
    chunks got through, the idle seconds at death, retry count, total duration.
    Failures to write are swallowed (tracing must never break a stream)."""
    try:
        from datetime import datetime as _dt
        root = os.environ.get("PUX_PROJECT_ROOT") or "."
        path = os.path.join(root, ".pux", "stream-trace.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = (
            f"{_dt.now().isoformat(timespec='seconds')} "
            f"model={model} outcome={outcome} attempts={attempts} "
            f"chunks={chunks} idle_s={idle_s:.0f} dur_s={dur_s:.0f}\n"
        )
        with open(path, "a") as fh:
            fh.write(line)
    except Exception:  # noqa: BLE001 — tracing must never break a stream
        pass

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
    """DeepSeek / MiMo / OpenRouter / OpenRouter: ``delta.reasoning_content``."""
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

    # ------------------------------------------------------------------
    # Mid-stream stall recovery (the "goes fast, then freezes" fix).
    # ------------------------------------------------------------------
    # PROVEN root cause (reproduced at THIS layer with NO graph): on long
    # large-context generations, the provider intermittently goes silent
    # mid-stream — chunks flow at ≤9s gaps, then the TCP connection stays
    # open but ZERO SSE bytes arrive for the full 180s httpx ``timeout``.
    # langchain-openai's ``stream_chunk_timeout`` was DISABLED (its
    # ``StreamChunkTimeoutError`` is raised OUTSIDE the SDK retry loop, so it
    # escaped ``astream`` → the editor froze). That left nothing to catch the
    # dead stream → a 180s freeze the user had to kill manually.
    #
    # This is OUR code (the model client), not the provider: the SAME model +
    # provider driven directly (Claude Code) never freezes — it recovers. So
    # we add our OWN idle watchdog here with transparent pre-output retry:
    #
    #   * Stall BEFORE any chunk reached the caller (the reasoning/TTFT phase,
    #     where a long large-context think goes silent before the first answer
    #     token): retry the whole stream up to ``PUX_STREAM_IDLE_RETRIES``
    #     times. Transparent — the caller sees one healthy stream. This is the
    #     recovery path Claude Code gets.
    #   * Stall AFTER chunks were already yielded (mid-generation): raise
    #     ``_StreamIdle``. A retry here would DUPLICATE output already shown to
    #     the user, so we let it propagate to ``acp.prompt``'s except-clause →
    #     a clean ``end_turn`` + notice (partial reply stays; NO freeze, NO
    #     dup). Still a 3x faster failure than the 180s wall.
    #
    # The default 60s idle threshold is 6x the largest LEGITIMATE reasoning
    # gap we've MEASURED (≤9s on glm-5.2 @ reasoning_effort=max), so a live
    # thinker is never falsely killed. Set ``PUX_STREAM_IDLE_TIMEOUT_S=0`` to
    # disable (back to raw langchain behavior).
    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """astream with an idle watchdog + at most ONE pre-output retry.

        A reasoning provider (glm-5.2 @ reasoning_effort=max) intermittently
        goes silent mid-stream on long large-context generations — chunks flow
        at ≤9s gaps, then the TCP connection stays open but ZERO SSE bytes
        arrive. Without a watchdog the editor froze for the full 180s httpx
        ``timeout``. This wraps ``super().astream`` with an idle detector:

          * idle BEFORE any chunk reached the caller (TTFT/reasoning phase):
            retry ONCE (a provider that failed to start a stream usually starts
            on re-send — the same recovery every robust streaming client does).
          * idle AFTER chunks were already yielded: raise immediately (retrying
            would duplicate output already shown). Propagates to
            ``acp.prompt``'s except-clause → clean ``end_turn`` + notice.

        Defaults are deliberately TIGHT so a deterministic stall cannot loop:
        30s idle (3x the largest legitimate reasoning gap measured) + 1 retry =
        ~60s worst case to a clean notice, never a freeze, never a multi-loop.

        ``PUX_STREAM_IDLE_TIMEOUT_S=0`` disables the guard entirely (raw
        passthrough). All outcomes append a one-line trace to
        ``.pux/stream-trace.log`` so a real stall is diagnosable post-hoc."""
        idle_s = float(os.environ.get("PUX_STREAM_IDLE_TIMEOUT_S", "30") or 0)
        max_retries = int(os.environ.get("PUX_STREAM_IDLE_RETRIES", "1"))
        model_id = getattr(self, "model_name", None) or getattr(self, "model", "?")
        t_start = time.monotonic()
        if idle_s <= 0:
            async for chunk in super().astream(*args, **kwargs):
                yield chunk
            _trace(model_id, "ok(guard-off)", t_start, 0, 0, 0, 0)
            return
        yielded = False
        chunks = 0
        last_idle = 0.0
        for attempt in range(max_retries + 1):
            attempt_start = time.monotonic()
            try:
                async for chunk in self._astream_idle_guarded(
                    super().astream(*args, **kwargs), idle_s
                ):
                    yielded = True
                    chunks += 1
                    yield chunk
                _trace(model_id, "ok", t_start, attempt + 1, chunks, 0.0,
                       time.monotonic() - attempt_start)
                return  # clean completion
            except _StreamIdle as e:
                last_idle = e.idle_s
                if not yielded and attempt < max_retries:
                    _log.warning(
                        "astream idle %.0fs before any output (attempt %d/%d); "
                        "retrying once — intermittent mid-stream silence",
                        idle_s, attempt + 1, max_retries,
                    )
                    continue
                # post-output stall OR the one retry also stalled → do NOT loop;
                # propagate so acp.prompt ends the turn cleanly.
                _trace(model_id, "stall(post-output)" if yielded else "stall(ttfb)",
                       t_start, attempt + 1, chunks, last_idle,
                       time.monotonic() - t_start)
                raise

    @staticmethod
    async def _astream_idle_guarded(
        source: AsyncIterator[Any], idle_s: float
    ) -> AsyncIterator[Any]:
        """Yield from ``source``; raise ``_StreamIdle`` if no chunk arrives
        within ``idle_s`` seconds (a dead stream — connection open, zero bytes).

        On idle, the underlying iterator is closed (``aclose``) so the stale
        HTTP stream is released before the caller retries."""
        ait = source.__aiter__()
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(ait.__anext__(), timeout=idle_s)
                except StopAsyncIteration:
                    return
                except asyncio.TimeoutError:
                    raise _StreamIdle(idle_s)
                yield chunk
        finally:
            await ait.aclose()


class _StreamIdle(Exception):
    """Raised by ``_astream_idle_guarded`` when no SSE chunk arrives within the
    idle threshold — a dead stream (TCP open, zero bytes). Carries the elapsed
    idle seconds for logging."""

    def __init__(self, idle_s: float) -> None:
        super().__init__(f"stream idle {idle_s:.0f}s (no SSE chunk; dead connection)")
        self.idle_s = idle_s
