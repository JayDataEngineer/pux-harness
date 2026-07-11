"""Langfuse observability — the no-op-unless-configured contract.

Owns ``agent/observability.py``: the single graph-invoke-config builder wired
into both ``main.py`` (``pux direct``) and the Aegra runtime (prod serve). Two
guarantees pinned here:

1. **No-op unless configured** — when langfuse is NOT importable OR the
   ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` env vars are unset,
   ``build_invoke_config`` returns the exact plain dict the graph already used
   (``configurable.thread_id`` + ``recursion_limit``; NO ``callbacks`` / NO
   ``metadata``). This is why both invoke sites import it unconditionally — zero
   behavior change for an unconfigured deployment.
2. **Attached when configured** — with the import faked in (langfuse is an
   optional extra) AND both env vars set, a ``CallbackHandler`` lands in
   ``config["callbacks"]`` and the reserved ``langfuse_session_id`` /
   ``langfuse_tags`` keys land in ``config["metadata"]``.

**v4 API drift guard:** ``test_constructor_takes_no_session_id`` (skipped when
langfuse isn't installed) asserts the real handler REJECTS ``session_id=`` —
this is the exact regression that bit Phase C's first cut (a constructor kwarg
that raised ``TypeError`` live; the v4 handler takes only ``public_key`` /
``trace_context`` and reads session/tags from run metadata).

The env-gated LIVE proof (a real configured run emitting a trace) is a separate
step (Phase D); these unit tests pin the decision logic + config shape.
"""
from __future__ import annotations

import pytest

from pux_harness.agent import observability as obs


def test_handler_none_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert obs.langfuse_handler() is None


def test_handler_none_when_credentials_partial(monkeypatch) -> None:
    # Only PUBLIC_KEY set, no SECRET_KEY -> the AND guard keeps it a no-op
    # (a half-configured host must never start a half-working trace).
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert obs.langfuse_handler() is None


def test_config_is_plain_when_off(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    cfg = obs.build_invoke_config("t-3", 80, "general", "direct")
    # Identical to the prior inline dict — no callbacks, no metadata.
    assert cfg == {
        "configurable": {"thread_id": "t-3"},
        "recursion_limit": 80,
    }
    assert "callbacks" not in cfg
    assert "metadata" not in cfg


def test_import_absence_keeps_module_importable() -> None:
    # langfuse is an optional extra; in a base env it is absent, so the guarded
    # import must resolve _HAS_LANGFUSE=False without crashing THIS module's
    # import (both invoke sites import it unconditionally).
    assert obs._HAS_LANGFUSE in (True, False)  # never an ImportError
    if obs._HAS_LANGFUSE is False:
        assert obs.langfuse_handler() is None


def test_handler_constructed_when_importable_and_env_set(monkeypatch) -> None:
    class FakeHandler:
        # v4: the handler ctor takes NO kwargs — session/tags are config metadata.
        def __init__(self) -> None:
            pass

    monkeypatch.setattr(obs, "_HAS_LANGFUSE", True)
    monkeypatch.setattr(obs, "_LangfuseHandler", FakeHandler)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")

    handler = obs.langfuse_handler()
    assert isinstance(handler, FakeHandler)


def test_config_carries_callbacks_and_metadata_when_on(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(obs, "_HAS_LANGFUSE", True)
    monkeypatch.setattr(obs, "_LangfuseHandler", lambda **_kw: sentinel)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")

    cfg = obs.build_invoke_config("thread-abc", 200, "coder", "serve")
    assert cfg["configurable"] == {"thread_id": "thread-abc"}
    assert cfg["recursion_limit"] == 200
    assert cfg["callbacks"] == [sentinel]
    # v4 attribute-propagation keys (read by the handler from run metadata).
    assert cfg["metadata"]["langfuse_session_id"] == "thread-abc"
    assert cfg["metadata"]["langfuse_tags"] == ["org:coder", "transport:serve"]


@pytest.mark.skipif(not obs._HAS_LANGFUSE,
                    reason="needs the real langfuse extra to probe its API")
def test_constructor_takes_no_session_id() -> None:
    """v4 API drift guard: the real CallbackHandler MUST reject session_id=.

    Phase C's first cut passed session_id= in the constructor, which raised
    TypeError live (the unit test had faked the handler and missed it). The v4
    handler reads session/tags from run METADATA (langfuse_session_id /
    langfuse_tags), not the ctor. This test pins that contract so the drift
    can't silently recur on a langfuse upgrade.
    """
    with pytest.raises(TypeError):
        obs._LangfuseHandler(session_id="x")  # type: ignore[misc]
