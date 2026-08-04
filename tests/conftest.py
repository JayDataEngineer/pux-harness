"""Shared fixtures for the harness test suite.

Auto-seeds provider keys — see tests/conftest.py for the model-agnostic
rationale (reads api_key_env from models.yaml, no per-provider hardcoding)."""
import pytest


@pytest.fixture(autouse=True)
def _provider_test_keys(monkeypatch):
    from pux_harness.agent import model as _model  # local import: tests may patch it
    for prof in _model._providers().values():
        env = prof.get("api_key_env")
        if env:
            monkeypatch.setenv(env, "test-key")
