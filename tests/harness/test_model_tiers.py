"""Model tiers + per-id capability + dcode helpers.

Companion to ``test_models.py``: proves the TIER layer (``tiers:`` +
``default_tier`` + ``PUX_TIER``), the per-id ``multimodal`` capability registry,
the cross-check that a tier's ``multimodal_model`` must be multimodal-capable,
and the dcode ``provider:id`` forwarding refs the TUI uses. Token- and
Docker-free — every check is pure id resolution + lookup.
"""
from __future__ import annotations

import pytest

from pux_harness.agent import model


@pytest.fixture
def _spec_cleared():
    """``model._spec`` is ``lru_cache``'d for the process; clear it so a test
    that monkeypatches the yaml path re-reads."""
    model._spec.cache_clear()
    yield
    model._spec.cache_clear()


# --- active_tier -------------------------------------------------------------

def test_active_tier_defaults_to_default_tier(monkeypatch):
    """No ``PUX_TIER`` -> the shipped ``default_tier`` (``default``)."""
    monkeypatch.delenv("PUX_TIER", raising=False)
    assert model.active_tier() == "default"


def test_active_tier_reads_pux_tier(monkeypatch):
    """``PUX_TIER`` selects the tier (set by --tier/--fast)."""
    monkeypatch.setenv("PUX_TIER", "fast")
    assert model.active_tier() == "fast"


def test_active_tier_rejects_unknown(monkeypatch):
    """An unknown ``PUX_TIER`` fails LOUD (no silent fallback to default) — a
    typo dies at the CLI / first resolution, not mid-run on the wrong model."""
    monkeypatch.setenv("PUX_TIER", "bogus")
    with pytest.raises(ValueError, match=r"PUX_TIER='bogus'.*not a known tier"):
        model.active_tier()


# --- the shipped tiers resolve the intended role split -----------------------

def test_default_tier_is_sota_supervisor_plus_cheap_workers(monkeypatch):
    """The shipped DEFAULT tier pairs a SOTA text-reasoning supervisor
    (glm-5.2) with cheap multimodal workers — the smart-coordinator /
    cheap-doers split. glm-5.2 is NOT multimodal (reaches vision via the
    describe_image fallback)."""
    monkeypatch.delenv("PUX_TIER", raising=False)
    assert model.resolve_model_id(role="base") == "glm-5.2"
    assert model.resolve_model_id(role="worker") == "mimo-v2.5"
    assert model.resolve_model_id(role="multimodal") == "mimo-v2.5"
    assert model.resolve_model_id(role="grader") == "mimo-v2.5"


def test_fast_tier_cheapens_every_role(monkeypatch):
    """``--fast`` / ``PUX_TIER=fast`` flips the supervisor to the cheap model
    too (the rate-limit-fallback / trivial-task mode)."""
    monkeypatch.setenv("PUX_TIER", "fast")
    assert model.resolve_model_id(role="base") == "mimo-v2.5"


def test_org_override_wins_over_tier(tmp_path, monkeypatch, _spec_cleared):
    """Org ``models:`` is ABOVE the tier (priority 2 > 4) — an explicit org pin
    holds in EVERY tier. (The tier is the FLOOR, never a ceiling.)"""
    from pux_harness.agent import orgs, profile
    (tmp_path / "orgs").mkdir()
    monkeypatch.setattr(orgs, "_orgs_dir", lambda: tmp_path / "orgs")
    monkeypatch.setattr(profile._orgs_mod, "_orgs_dir", lambda: tmp_path / "orgs")
    (tmp_path / "orgs" / "o").mkdir(parents=True)
    (tmp_path / "orgs" / "o" / "profile.yaml").write_text(
        "models:\n  base_model: kimi-k2.7-code\n"
    )
    monkeypatch.setenv("PUX_TIER", "fast")  # would otherwise give mimo-v2.5
    assert model.resolve_model_id(role="base", org="o") == "kimi-k2.7-code"


# --- per-id capability registry ---------------------------------------------

def test_is_multimodal_reads_registry():
    """``multimodal`` is a property of the MODEL ID, read from the registry."""
    assert model.is_multimodal("mimo-v2.5") is True
    assert model.is_multimodal("mimo-v2.5-pro") is True
    assert model.is_multimodal("glm-5.2") is False
    assert model.is_multimodal("kimi-k2.7-code") is False


def test_is_multimodal_unknown_defaults_false():
    """An UNKNOWN id defaults to non-multimodal (fail-safe: a driver of unknown
    capability takes the describe_image fallback, never image blocks it can't
    read)."""
    assert model.is_multimodal("brand-new-model-id") is False
    assert model.is_multimodal("") is False


def test_driver_multimodal_flips_with_tier(monkeypatch):
    """``driver_multimodal`` resolves the role's id (per the active tier) then
    its capability — the seam BrowserVisionMiddleware uses. Default tier's
    supervisor is glm-5.2 (text-only); fast tier flips it to mimo (multimodal)."""
    monkeypatch.delenv("PUX_TIER", raising=False)
    assert model.driver_multimodal(role="base") is False  # glm-5.2
    monkeypatch.setenv("PUX_TIER", "fast")
    assert model.driver_multimodal(role="base") is True   # mimo-v2.5


def test_driver_multimodal_worker_is_multimodal_in_default_tier(monkeypatch):
    """The DEFAULT tier's WORKER is mimo (multimodal) even though the supervisor
    isn't — subagents read screenshots natively while the supervisor delegates."""
    monkeypatch.delenv("PUX_TIER", raising=False)
    assert model.driver_multimodal(role="worker") is True


# --- a tier missing a role key fails loud (the modern form of the old roles: check) ---

def test_validate_rejects_tier_missing_role_key(tmp_path, monkeypatch, _spec_cleared):
    """A TIER missing one of the four role keys fails loud (the contract checker
    calls validate_models_spec; this is the tier-era equivalent of the old
    ``roles:`` missing-key check)."""
    (tmp_path / "models.yaml").write_text(
        "providers:\n  p: {kind: openai, base_url: x, api_key_env: K}\n"
        "default_provider: p\n"
        "tiers:\n"
        "  default:\n"
        "    base_model: a\n"
        "    worker_model: a\n"
        "    multimodal_model: a\n"
        "    # grader_model missing\n"
        "default_tier: default\n"
    )
    monkeypatch.setattr(model, "_YAML", tmp_path / "models.yaml")
    with pytest.raises(RuntimeError, match=r"tier 'default'.*missing role key.*grader_model"):
        model.validate_models_spec()


def test_validate_rejects_unknown_default_tier(tmp_path, monkeypatch, _spec_cleared):
    """``default_tier`` must NAME a real tier — a typo fails loud."""
    (tmp_path / "models.yaml").write_text(
        "providers:\n  p: {kind: openai, base_url: x, api_key_env: K}\n"
        "default_provider: p\n"
        "tiers:\n"
        "  default: {base_model: a, worker_model: a, multimodal_model: a, grader_model: a}\n"
        "default_tier: defualt\n"   # typo
    )
    monkeypatch.setattr(model, "_YAML", tmp_path / "models.yaml")
    with pytest.raises(RuntimeError, match=r"default_tier.*must name a tier"):
        model.validate_models_spec()


# --- the cross-check: a tier's multimodal_model must be multimodal-capable ---

def test_validate_rejects_non_multimodal_vision_model(tmp_path, monkeypatch, _spec_cleared):
    """A tier whose ``multimodal_model`` is flagged non-multimodal in the
    registry fails the contract — vision (describe_image) would silently break
    at runtime, so fail it at --check-contract."""
    (tmp_path / "models.yaml").write_text(
        "providers:\n  p: {kind: openai, base_url: x, api_key_env: K}\n"
        "default_provider: p\n"
        "models:\n  glm-5.2: {multimodal: false}\n  a: {multimodal: true}\n"
        "tiers:\n"
        "  default:\n"
        "    base_model: a\n"
        "    worker_model: a\n"
        "    multimodal_model: glm-5.2\n"   # non-multimodal vision model!
        "    grader_model: a\n"
        "default_tier: default\n"
    )
    monkeypatch.setattr(model, "_YAML", tmp_path / "models.yaml")
    with pytest.raises(RuntimeError, match=r"multimodal_model.*non-multimodal"):
        model.validate_models_spec()


# --- dcode forwarding refs (the TUI uses these) ------------------------------

def test_dcode_model_ref_default(monkeypatch):
    """The TUI forwards the resolved base model as ``<provider>:<id>`` so dcode
    never falls back to its own default (deepseek-v4-flash drift)."""
    monkeypatch.delenv("PUX_TIER", raising=False)
    assert model.dcode_model_ref(role="base") == "opencode-go-openai:glm-5.2"
    monkeypatch.setenv("PUX_TIER", "fast")
    assert model.dcode_model_ref(role="base") == "opencode-go-openai:mimo-v2.5"


def test_dcode_model_ref_none_when_no_provider(tmp_path, monkeypatch, _spec_cleared):
    """With no ``dcode:`` block, the ref is None (the TUI then lets dcode pick —
    honest, never a fabricated ref)."""
    (tmp_path / "models.yaml").write_text(
        "providers:\n  p: {kind: openai, base_url: x, api_key_env: K}\n"
        "default_provider: p\n"
        "tiers:\n"
        "  default: {base_model: a, worker_model: a, multimodal_model: a, grader_model: a}\n"
        "default_tier: default\n"
    )
    monkeypatch.setattr(model, "_YAML", tmp_path / "models.yaml")
    assert model.dcode_provider() is None
    assert model.dcode_model_ref(role="base") is None


def test_dcode_model_ref_literal_override(monkeypatch):
    """An explicit ``model=`` literal (a bare id, the frontmatter-override path)
    flows through ``dcode_model_ref`` with the provider prepended."""
    monkeypatch.delenv("PUX_TIER", raising=False)
    assert model.dcode_model_ref(
        role="base", model="kimi-k2.7-code",
    ) == "opencode-go-openai:kimi-k2.7-code"
