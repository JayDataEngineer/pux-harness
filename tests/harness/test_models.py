"""Model-role spec.

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

import openai
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_openai import ChatOpenAI

from pux_harness.agent import model, profile


def _model_id(m) -> str:
    """Provider-agnostic model-id accessor — ``ChatOpenAI`` exposes
    ``model_name``, ``ChatAnthropic`` exposes ``model``. Tests assert the
    RESOLVED id is plumbed through regardless of which concrete class
    ``get_model`` built (the harness serves both kinds now)."""
    return getattr(m, "model_name", None) or getattr(m, "model", None)


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
    """A tier missing one of the four role keys fails loud."""
    (tmp_path / "models.yaml").write_text(
        "providers:\n  p: {kind: openai, base_url: x, api_key_env: K}\n"
        "default_provider: p\n"
        "tiers:\n  t:\n"
        "    base_model: a\n"
        "    worker_model: a\n"
        "    multimodal_model: a\n"
        "    # grader_model missing\n"
        "default_tier: t\n"
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
        role="grader", org="coder", model="literal-x",
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
    frontmatter override path). Covers BOTH provider kinds: glm-5.2 (anthropic)
    and mimo-v2.5 (openai) — each builds on its own profile's protocol."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
    assert _model_id(model.get_model(model="glm-5.2")) == "glm-5.2"
    assert _model_id(model.get_model(model="mimo-v2.5")) == "mimo-v2.5"


# --- per-role declared fallback chains -------------------------------------
#
# A tier may declare ``<role>_fallbacks``: an ordered list of model ids
# ``get_model`` wraps via LangChain ``with_fallbacks`` so a transient error
# (429/timeout/5xx) exhausting the primary's ``max_retries`` fails over. The
# chain applies ONLY to a role resolved from the tier — an explicit override
# (frontmatter / org / env) is a hard pin. Declared + logged, not a silent swap.


def test_fallback_keys_align_with_roles():
    """FALLBACK_KEYS is exactly ``<role>_fallbacks`` for each role, in order."""
    assert model.FALLBACK_KEYS == (
        "base_fallbacks", "worker_fallbacks",
        "multimodal_fallbacks", "grader_fallbacks",
    )


def test_shipped_default_base_has_fallback_chain():
    """The shipped default tier declares a base fallback chain (cross-provider
    escape: glm-5.2 zai-coding → mimo-v2.5-pro opencode-go)."""
    assert model.resolve_fallback_ids(role="base") == ["mimo-v2.5-pro"]


def test_worker_role_has_no_shipped_fallbacks():
    """Roles without a declared chain resolve to an empty list (no fallback)."""
    assert model.resolve_fallback_ids(role="worker") == []
    assert model.resolve_fallback_ids(role="grader") == []


def test_resolve_fallback_ids_rejects_unknown_role():
    """An unknown role fails loud (consistent with resolve_model_id)."""
    with pytest.raises(ValueError, match="unknown model role"):
        model.resolve_fallback_ids(role="intern")


def test_literal_override_disables_fallbacks():
    """A caller-supplied ``model=`` literal is a hard pin — no failover."""
    assert model.resolve_fallback_ids(role="base", model="glm-5.1") == []


def test_env_override_disables_fallbacks(monkeypatch):
    """``PUX_<ROLE>_MODEL`` pins the role — the tier's chain is NOT attached."""
    monkeypatch.setenv("PUX_BASE_MODEL", "pinned-base")
    assert model.resolve_fallback_ids(role="base") == []


def test_legacy_pux_model_disables_base_fallbacks(monkeypatch):
    """The legacy ``PUX_MODEL`` (base-only) is also a hard pin for the base role."""
    monkeypatch.setenv("PUX_MODEL", "legacy-base")
    assert model.resolve_fallback_ids(role="base") == []


def test_org_override_disables_fallbacks(fake_org_tree):
    """An org ``models: {base_model: X}`` pins the role above the tier — the
    chain is NOT attached for that org's base."""
    (fake_org_tree / "orgs" / "o").mkdir(parents=True)
    (fake_org_tree / "orgs" / "o" / "profile.yaml").write_text(
        "models:\n  base_model: org-base\n"
    )
    assert model.resolve_fallback_ids(role="base", org="o") == []
    # An org WITHOUT a base override still gets the tier chain.
    (fake_org_tree / "orgs" / "p").mkdir(parents=True)
    (fake_org_tree / "orgs" / "p" / "profile.yaml").write_text(
        "system_prompt_suffix: hi\n"
    )
    assert model.resolve_fallback_ids(role="base", org="p") == ["mimo-v2.5-pro"]


def test_get_model_attaches_fallback_chain(monkeypatch):
    """``get_model(role=base)`` returns a ``BaseChatModel`` (an anthropic-kind
    ``_FallbackChatAnthropic`` for the shipped glm-5.2 default) carrying the
    declared tier chain — NOT a bare ``RunnableWithFallbacks`` (which crashes
    ``deepagents.resolve_model``). The primary id is still the resolved role id;
    the chain ids are the declared fallbacks. Proves the rely-on-upstream wiring
    (LangChain ``with_fallbacks``) without a live call."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
    m = model.get_model(role="base")
    # MUST be a BaseChatModel — deepagents' resolve_model fast-path requires it.
    assert isinstance(m, BaseChatModel)
    # And NOT a RunnableWithFallbacks (the pre-fix shape that crashed serve startup).
    assert not isinstance(m, RunnableWithFallbacks)
    assert _model_id(m) == "glm-5.2"
    assert [_model_id(fb) for fb in m._fallback_models] == ["mimo-v2.5-pro"]


def test_fallback_model_passes_deepagents_resolve_model_seam(monkeypatch):
    """REGRESSION (verify-or-die): the fallback-bearing base model is accepted by
    ``deepagents.resolve_model``'s ``isinstance(model, BaseChatModel)`` fast-path
    and returned UNCHANGED. A bare ``RunnableWithFallbacks`` (the pre-fix shape)
    missed the fast-path → ``apply_provider_profile(model)`` →
    ``get_provider_profile(model)`` → ``spec.count(':")`` → ``AttributeError``
    ('ReasoningChatOpenAI' object has no attribute 'count') → Aegra
    startup crash. This is the EXACT gap that let the regression ship: bind_tools
    was proven but the model was never driven through the real deepagents seam."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
    from deepagents._models import resolve_model
    m = model.get_model(role="base")
    # Fast-path: the BaseChatModel is returned as-is (no re-init, no string parse).
    assert resolve_model(m) is m


def test_fallback_model_builds_in_real_deepagent_factory(monkeypatch):
    """REGRESSION (verify-or-die): the fallback-bearing base model is accepted by
    the REAL ``deepagents.create_deep_agent`` factory — the full seam that crashed
    Aegra startup (build_graph → create_deep_agent → resolve_model →
    AttributeError). Building binds tools internally; proving the factory accepts
    the model is the strongest guarantee that Aegra boots. No network —
    the model is never invoked, only assembled."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
    from deepagents import create_deep_agent
    m = model.get_model(role="base")
    graph = create_deep_agent(
        model=m,
        system_prompt="you are a test agent",
        tools=[],
        subagents=[],
    )
    assert graph is not None


def test_get_model_without_fallbacks_is_plain_chat(monkeypatch):
    """A role with no declared chain returns a plain ChatOpenAI (no wrapper) —
    zero behavior change for non-fallback roles."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    m = model.get_model(role="grader")
    assert not isinstance(m, RunnableWithFallbacks)
    assert isinstance(m, ChatOpenAI)


def test_get_model_override_skips_fallback_chain(monkeypatch):
    """An explicit ``model=`` literal returns a plain model on that id — the
    tier chain is NOT attached (a hard pin has no failover). glm-5.2 is served
    via the anthropic profile, so the plain model is a ``ChatAnthropic``."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
    m = model.get_model(role="base", model="glm-5.2")
    assert not isinstance(m, RunnableWithFallbacks)
    assert _model_id(m) == "glm-5.2"


def test_bind_tools_flows_through_fallback_chain(monkeypatch):
    """``with_fallbacks`` propagates ``.bind_tools`` to every model in the chain
    — the load-bearing precondition for deepagents, which binds tools AFTER
    ``get_model`` returns. If this breaks, the supervisor's tool surface vanishes."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    m = model.get_model(role="base")
    bound = m.bind_tools([{"name": "t", "description": "d", "input_schema": {}}])
    assert isinstance(bound, RunnableWithFallbacks)
    assert len(bound.fallbacks) == 1


def test_transient_exceptions_are_narrow(monkeypatch):
    """The failover set is EXACTLY the transient API errors — 401/400/404 (auth,
    bad-request, not-found) are deliberately ABSENT so a config/request bug dies
    loud instead of silently burning the chain (no-silent-fallbacks)."""
    excs = set(model._TRANSIENT_EXCEPTIONS)
    assert excs == {
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    }
    # The non-transient config/request errors are NOT in the set.
    for non_transient in (openai.AuthenticationError, openai.BadRequestError,
                          openai.NotFoundError):
        assert non_transient not in excs


def test_transient_set_triggers_failover_deterministically():
    """Behavioral proof (no live outage needed): with ``_TRANSIENT_EXCEPTIONS``,
    a transient error on the primary fails over to the next declared model —
    the runtime contract ``get_model`` relies on when it wraps the chain via
    LangChain ``with_fallbacks``. Uses ``RunnableLambda`` so the failover logic
    (a Runnable-level concern, identical for any Runnable incl. ChatOpenAI) is
    exercised deterministically."""
    import httpx
    from langchain_core.runnables import RunnableLambda

    def _rate_limit(_):
        resp = httpx.Response(
            status_code=429, request=httpx.Request("POST", "http://x/v1"))
        raise openai.RateLimitError("limited", response=resp, body=None)

    chain = RunnableLambda(_rate_limit).with_fallbacks(
        [RunnableLambda(lambda _: "FALLBACK-OK")],
        exceptions_to_handle=model._TRANSIENT_EXCEPTIONS,
    )
    # Transient 429 on the primary → the fallback's value is returned.
    assert chain.invoke("x") == "FALLBACK-OK"


def test_non_transient_error_does_not_failover():
    """The flip side of no-silent-fallbacks: a config/request bug (400
    BadRequest) is NOT in the transient set, so the chain does NOT mask it — the
    error raises loud instead of being silently swallowed by the next model."""
    import httpx
    from langchain_core.runnables import RunnableLambda

    def _bad_request(_):
        resp = httpx.Response(
            status_code=400, request=httpx.Request("POST", "http://x/v1"))
        raise openai.BadRequestError("bad", response=resp, body=None)

    chain = RunnableLambda(_bad_request).with_fallbacks(
        [RunnableLambda(lambda _: "SHOULD-NOT-REACH")],
        exceptions_to_handle=model._TRANSIENT_EXCEPTIONS,
    )
    with pytest.raises(openai.BadRequestError):
        chain.invoke("x")


def test_resolve_fallback_ids_returns_a_copy():
    """The returned list is a copy — mutating it must not corrupt the cached spec
    (a later call would otherwise see the mutation)."""
    chain = model.resolve_fallback_ids(role="base")
    chain.append("mutated")
    assert model.resolve_fallback_ids(role="base") == ["mimo-v2.5-pro"]


# --- validate_models_spec rejects malformed fallback chains ----------------


def _minimal_spec(tmp_path, monkeypatch, tier_body: str) -> None:
    """Write a minimal VALID spec (one provider profile + one tier with all four
    role keys, no models registry so the multimodal cross-check is skipped) +
    ``tier_body`` appended inside the tier, then point ``model._YAML`` at it.
    Used to assert specific ``<role>_fallbacks`` malformations fail loud."""
    (tmp_path / "models.yaml").write_text(
        "providers:\n  p: {kind: openai, base_url: x, api_key_env: K}\n"
        "default_provider: p\n"
        "tiers:\n  t:\n"
        + tier_body
        + "\ndefault_tier: t\n"
    )
    monkeypatch.setattr(model, "_YAML", tmp_path / "models.yaml")


def _minimal_role_lines() -> str:
    return (
        "    base_model: a\n"
        "    worker_model: a\n"
        "    multimodal_model: a\n"
        "    grader_model: a\n"
    )


def test_validate_accepts_well_formed_fallbacks(tmp_path, _spec_cleared, monkeypatch):
    """A non-empty list of unique string ids is accepted."""
    _minimal_spec(
        tmp_path, monkeypatch, _minimal_role_lines() + "    base_fallbacks: [b, c]\n"
    )
    model.validate_models_spec()  # raises on any problem


def test_validate_rejects_non_list_fallbacks(tmp_path, _spec_cleared, monkeypatch):
    """A scalar ``base_fallbacks`` (not a list) fails loud."""
    _minimal_spec(
        tmp_path, monkeypatch, _minimal_role_lines() + "    base_fallbacks: b\n"
    )
    with pytest.raises(RuntimeError, match="base_fallbacks must be a non-empty list"):
        model.validate_models_spec()


def test_validate_rejects_empty_fallback_list(tmp_path, _spec_cleared, monkeypatch):
    """An empty ``base_fallbacks: []`` fails loud (a declared-but-empty chain is
    almost certainly a config mistake — it declares failover intent with no ids)."""
    _minimal_spec(
        tmp_path, monkeypatch, _minimal_role_lines() + "    base_fallbacks: []\n"
    )
    with pytest.raises(RuntimeError, match="base_fallbacks must be a non-empty list"):
        model.validate_models_spec()


def test_validate_rejects_non_string_fallback_entry(tmp_path, _spec_cleared, monkeypatch):
    """A non-string entry (e.g. an int) fails loud."""
    _minimal_spec(
        tmp_path, monkeypatch, _minimal_role_lines() + "    base_fallbacks: [5]\n"
    )
    with pytest.raises(RuntimeError, match="base_fallbacks entries must be non-empty"):
        model.validate_models_spec()


def test_validate_rejects_duplicate_fallback_ids(tmp_path, _spec_cleared, monkeypatch):
    """Duplicate ids in the chain fail loud (a redundant failover target)."""
    _minimal_spec(
        tmp_path, monkeypatch, _minimal_role_lines() + "    base_fallbacks: [b, b]\n"
    )
    with pytest.raises(RuntimeError, match="base_fallbacks has duplicate"):
        model.validate_models_spec()
