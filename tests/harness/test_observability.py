"""Langfuse observability — the no-op-unless-configured contract.

Owns ``agent/observability.py``: the single graph-invoke-config builder wired
into both ``main.py`` (``pux direct``) and ``server.py`` (``pux serve``). Two
guarantees pinned here:

1. **No-op unless configured** — when langfuse is NOT importable OR the
   ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` env vars are unset,
   ``build_invoke_config`` returns the exact plain dict the graph already used
   (``configurable.thread_id`` + ``recursion_limit``; NO ``callbacks``). This is
   why both invoke sites import it unconditionally — zero behavior change for an
   unconfigured deployment.
2. **Attached when configured** — with the import faked in (langfuse is an
   optional extra, absent from the base install) AND both env vars set, a
   ``CallbackHandler`` is constructed with the thread_id as ``session_id`` +
   org/transport tags, and lands in ``config["callbacks"]``.

The env-gated LIVE proof (a real configured run emitting a trace) is a separate
step (Phase D); these unit tests pin the decision logic + config shape.
"""
from __future__ import annotations

from pux_harness.agent import observability as obs


def test_handler_none_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert obs.langfuse_handler("general", "direct", "t-1") is None


def test_handler_none_when_credentials_partial(monkeypatch) -> None:
    # Only PUBLIC_KEY set, no SECRET_KEY -> the AND guard keeps it a no-op
    # (a half-configured host must never start a half-working trace).
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert obs.langfuse_handler("general", "serve", "t-2") is None


def test_config_is_plain_when_off(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    cfg = obs.build_invoke_config("t-3", 80, "general", "direct")
    # Identical to the prior inline dict — no callbacks key at all.
    assert cfg == {
        "configurable": {"thread_id": "t-3"},
        "recursion_limit": 80,
    }
    assert "callbacks" not in cfg


def test_import_absence_keeps_module_importable() -> None:
    # langfuse is an optional extra; in this base env it is absent, so the
    # guarded import must resolve _HAS_LANGFUSE=False without crashing the
    # import of THIS module (both invoke sites import it unconditionally).
    assert obs._HAS_LANGFUSE in (True, False)  # never an ImportError
    # When genuinely absent, the handler is a no-op regardless of env.
    if obs._HAS_LANGFUSE is False:
        assert obs.langfuse_handler("general", "direct", "t-4") is None


def test_handler_constructed_when_importable_and_env_set(monkeypatch) -> None:
    captured: dict = {}

    class FakeHandler:
        def __init__(self, **kwargs):  # noqa: ANN204 - record what we were called with
            captured.update(kwargs)

    monkeypatch.setattr(obs, "_HAS_LANGFUSE", True)
    monkeypatch.setattr(obs, "_LangfuseHandler", FakeHandler)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")

    handler = obs.langfuse_handler("dev-bot", "serve", "thread-abc")
    assert isinstance(handler, FakeHandler)
    # thread_id -> session_id (one Langfuse session per resumed thread).
    assert captured["session_id"] == "thread-abc"
    # org + transport are tags (the UI filter axes) AND metadata.
    assert "org:dev-bot" in captured["tags"]
    assert "transport:serve" in captured["tags"]
    assert captured["metadata"]["org"] == "dev-bot"
    assert captured["metadata"]["transport"] == "serve"


def test_config_carries_callbacks_when_on(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(obs, "_HAS_LANGFUSE", True)
    monkeypatch.setattr(obs, "_LangfuseHandler", lambda **_kw: sentinel)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")

    cfg = obs.build_invoke_config("t-5", 200, "general", "direct")
    assert cfg["configurable"] == {"thread_id": "t-5"}
    assert cfg["recursion_limit"] == 200
    assert cfg["callbacks"] == [sentinel]
