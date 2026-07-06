"""Model-role spec (Phase 17.B.0).

Proves the resolution priority stack in ``model.resolve_model_id`` /
``model.get_model`` (literal > org profile > env > shipped default), the
``models.yaml`` shape contract (``validate_models_spec``), and that
``profile._validate_models_block`` rejects bad role keys at read time so a
typo fails ``--check-contract``.

Token- and Docker-free: ``resolve_model_id`` never builds a ``ChatOpenAI``.
``get_model`` IS exercised (it builds one) — tests set a throwaway
``OPENCODE_API_KEY``; no real chat happens.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_openai import ChatOpenAI

from pux_harness.agent import model, profile


@pytest.fixture
def _spec_cleared():
    """``model._spec`` is ``lru_cache``'d for the process; clear it so a test
    that monkeypatches the yaml path re-reads."""
    model._spec.cache_clear()
    yield
    model._spec.cache_clear()


# --- shipped spec + validate_models_spec -----------------------------------

def test_shipped_models_yaml_is_well_formed():
    """The shipped models.yaml passes its own validator (the contract checker
    calls validate_models_spec; this proves the green baseline)."""
    model.validate_models_spec()  # raises on any problem


def test_roles_and_keys_align():
    """ROLE_KEYS is exactly ``<role>_model`` for each role, in order — the
    profile loader validates an org's ``models:`` map against this set."""
    assert model.ROLES == ("base", "worker", "multimodal", "grader")
    assert model.ROLE_KEYS == (
        "base_model", "worker_model", "multimodal_model", "grader_model",
    )


def test_validate_models_spec_missing_file(tmp_path, _spec_cleared, monkeypatch):
    """A missing models.yaml fails loud (RuntimeError), not silently."""
    monkeypatch.setattr(model, "_YAML", tmp_path / "nope.yaml")
    with pytest.raises(RuntimeError, match="models.yaml not found"):
        model.validate_models_spec()


def test_validate_models_spec_missing_role(tmp_path, _spec_cleared, monkeypatch):
    """A roles: map missing one of the four role keys fails loud."""
    (tmp_path / "models.yaml").write_text(
        "provider:\n  base_url: x\n"
        "roles:\n"
        "  base_model: a\n"
        "  worker_model: a\n"
        "  multimodal_model: a\n"
        "  # grader_model missing\n"
    )
    monkeypatch.setattr(model, "_YAML", tmp_path / "models.yaml")
    with pytest.raises(RuntimeError, match=r"missing.*grader_model"):
        model.validate_models_spec()


# --- resolve_model_id priority stack ---------------------------------------

def test_resolve_default_is_shipped_value():
    """No overrides -> the shipped default for the role."""
    assert model.resolve_model_id(role="grader") == "mimo-v2.5"
    assert model.resolve_model_id(role="multimodal") == "mimo-v2.5"


def test_resolve_literal_wins_over_everything(monkeypatch):
    """A caller-supplied ``model=`` literal beats org, env, AND default."""
    monkeypatch.setenv("PUX_GRADER_MODEL", "env-glm")
    assert model.resolve_model_id(
        role="grader", org="dev-bot", model="literal-x",
    ) == "literal-x"


def test_resolve_env_wins_over_default(monkeypatch):
    """``PUX_<ROLE>_MODEL`` beats the shipped default."""
    monkeypatch.setenv("PUX_GRADER_MODEL", "env-glm")
    assert model.resolve_model_id(role="grader") == "env-glm"


def test_resolve_legacy_pux_model_for_base_only(monkeypatch):
    """The legacy ``PUX_MODEL`` env var is honored for the base role (back-compat)
    but NOT for the other roles."""
    monkeypatch.setenv("PUX_MODEL", "legacy-base")
    assert model.resolve_model_id(role="base") == "legacy-base"
    # Other roles ignore PUX_MODEL.
    assert model.resolve_model_id(role="worker") == "mimo-v2.5"


def test_resolve_rejects_unknown_role():
    """An unknown role fails loud (no silent fallback to base)."""
    with pytest.raises(ValueError, match="unknown model role"):
        model.resolve_model_id(role="intern")


# --- org profile `models:` override ----------------------------------------

@pytest.fixture
def fake_org_tree(tmp_path: Path, monkeypatch):
    """Scratch orgs/ tree so an org ``models:`` override can be written + read
    through the real ``profile._read_profile_yaml`` path resolver."""
    from pux_harness.agent import orgs
    (tmp_path / "orgs").mkdir()
    monkeypatch.setattr(orgs, "_orgs_dir", lambda: tmp_path / "orgs")
    monkeypatch.setattr(profile._orgs_mod, "_orgs_dir", lambda: tmp_path / "orgs")
    return tmp_path


def test_resolve_org_override_beats_env_and_default(fake_org_tree, monkeypatch):
    """An org ``models: {grader_model: X}`` beats env + the shipped default
    (priority 2 > 3 > 4)."""
    (fake_org_tree / "orgs" / "o").mkdir(parents=True)
    (fake_org_tree / "orgs" / "o" / "profile.yaml").write_text(
        "models:\n  grader_model: org-glm\n"
    )
    monkeypatch.setenv("PUX_GRADER_MODEL", "env-glm")
    assert model.resolve_model_id(role="grader", org="o") == "org-glm"


def test_resolve_org_override_absent_falls_through(fake_org_tree):
    """An org with no ``models:`` block falls through to the shipped default."""
    (fake_org_tree / "orgs" / "o").mkdir(parents=True)
    (fake_org_tree / "orgs" / "o" / "profile.yaml").write_text(
        "system_prompt_suffix: 'just a suffix'\n"
    )
    assert model.resolve_model_id(role="worker", org="o") == "mimo-v2.5"


# --- profile models-block validation ---------------------------------------

def test_models_block_rejects_unknown_role_key(fake_org_tree):
    """A typo'd role key (``grader_modle:``) fails loud at read time — not a
    silent fallback to the default."""
    (fake_org_tree / "orgs" / "o").mkdir(parents=True)
    (fake_org_tree / "orgs" / "o" / "profile.yaml").write_text(
        "models:\n  grader_modle: glm-5.2\n"
    )
    with pytest.raises(TypeError, match="unknown key"):
        profile.load_profile("o")


def test_models_block_rejects_non_string_value(fake_org_tree):
    """A non-string role value (e.g. a bare int) fails loud."""
    (fake_org_tree / "orgs" / "o").mkdir(parents=True)
    (fake_org_tree / "orgs" / "o" / "profile.yaml").write_text(
        "models:\n  grader_model: 5\n"
    )
    with pytest.raises(TypeError, match="non-empty string"):
        profile.load_profile("o")


def test_models_block_rejects_non_mapping(fake_org_tree):
    """A ``models:`` that isn't a mapping (e.g. a bare scalar) fails loud."""
    (fake_org_tree / "orgs" / "o").mkdir(parents=True)
    (fake_org_tree / "orgs" / "o" / "profile.yaml").write_text(
        "models: glm-5.2\n"
    )
    with pytest.raises(TypeError, match="models: must be a mapping"):
        profile.load_profile("o")


def test_models_block_peeled_from_harness_config(fake_org_tree):
    """A valid ``models:`` block is peeled out before
    ``HarnessProfileConfig.from_dict`` (which would otherwise reject it as an
    unknown key) — the org loads cleanly with its other profile fields."""
    (fake_org_tree / "orgs" / "o").mkdir(parents=True)
    (fake_org_tree / "orgs" / "o" / "profile.yaml").write_text(
        "system_prompt_suffix: 'hi'\n"
        "models:\n  grader_model: glm-5.2\n"
    )
    cfg = profile.load_profile("o")
    assert cfg is not None
    assert cfg.system_prompt_suffix == "hi"


# --- get_model builds a ChatOpenAI on the resolved id ----------------------

def test_get_model_builds_on_resolved_id(monkeypatch):
    """``get_model`` builds a ChatOpenAI whose ``model_name`` is the resolved
    role id (proves the provider-config wiring without a real chat)."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    m = model.get_model(role="grader")
    assert isinstance(m, ChatOpenAI)
    assert m.model_name == "mimo-v2.5"


def test_get_model_literal_override(monkeypatch):
    """``get_model(model=...)`` builds on the literal id (the subagent-
    frontmatter override path)."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    assert model.get_model(model="glm-5.2").model_name == "glm-5.2"
