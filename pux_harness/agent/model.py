"""Model spec — roles resolved from a shipped ``models.yaml``.

The single source of truth is ``models.yaml`` next to this module: it pins the
provider (``base_url`` + which env var holds the api key), a per-id capability
registry (``models:``, e.g. ``multimodal: true/false``), a set of named TIERS
(``tiers:`` — each a full role→id map), and the dcode provider name. No model id
is hardcoded anywhere else in the harness — every consumer asks
``get_model(role=..., org=...)`` and the resolution picks the concrete id.

Resolution priority for a role (highest wins):
  1. ``model=`` literal                — a caller-supplied id (subagent frontmatter
                                         ``model:`` — the agent-level override).
  2. org ``profile.yaml`` ``models:``  — org-level role override.
  3. ``PUX_<ROLE>_MODEL`` env var       — deployment-level (e.g. PUX_GRADER_MODEL);
                                         ``PUX_MODEL`` is honored for the base role
                                         as legacy back-compat.
  4. the ACTIVE TIER's role map         — selected by ``PUX_TIER`` (set by
                                         ``--tier``/``--fast``), defaulting to
                                         ``models.yaml``'s ``default_tier``.

Tier is the COARSE selector (a whole role map per "mode": the shipped ``default``
= SOTA supervisor + cheap workers; ``fast`` = everything cheap). Per-role
overrides (org, env, frontmatter) still win ABOVE the tier — the tier is the
floor, so an org's explicit ``models:`` pin is respected in every tier.

A tier may ALSO declare a ``<role>_fallbacks`` chain: an ordered list of model
ids ``get_model`` wraps via LangChain ``with_fallbacks`` so a transient error
(429 / timeout / 5xx) that exhausts the primary's ``max_retries`` fails over to
the next declared id. The chain applies ONLY to a role that resolved from the
tier — an explicit override (frontmatter / org / env) is a hard pin with no
failover. Declared + WARNING-logged, not a silent swap.

Capability (``is_multimodal``) is a property of the MODEL ID, read from the
``models:`` registry. Unknown ids default to non-multimodal (fail-safe: a driver
of unknown capability takes the describe_image fallback, never image blocks it
can't read). ``driver_multimodal`` resolves a role to its id then its capability
— the seam ``BrowserVisionMiddleware`` uses to decide image-block vs text-pointer.

Provider config comes from ``models.yaml``'s ``providers:`` map — named
profiles each pinning a wire protocol (``kind: openai`` | ``anthropic``), a
``base_url``, an ``api_key_env``, and generation defaults. A MODEL ID binds to a
profile in the ``models:`` registry (``provider: <name>``; ids without one use
``default_provider``). ``get_model`` builds ``ReasoningChatOpenAI`` for an
openai-kind id, ``ChatAnthropic`` for an anthropic-kind id — so one deployment
serves models from different vendors/protocols (e.g. glm-5.2 over ZAI's
Anthropic-compat endpoint, mimo-v2.5 over OpenCode Go's OpenAI-compat router).
``PUX_MAX_TOKENS`` / ``PUX_TEMPERATURE`` still override per-deployment; the
OpenCode-Go profile's ``base_url`` still honors the legacy ``OPENCODE_BASE_URL``
override for back-compat.

Historical notes:
  - Pre-17.B.0 a single ``DEFAULT_MODEL``/``get_model(model=None)`` was shared by
    every consumer; the rubric gate needs an independent grader model → roles.
  - Pre-tier the file had a flat ``roles:`` map (one role set, the default). That
    made the WEAK model the shipped supervisor — backwards for a "cheap vs SOTA"
    dial. ``tiers:`` + ``default_tier`` replaced it: the default is SOTA, ``fast``
    is the opt-in cheap mode.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import anthropic
import openai
import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from pydantic import PrivateAttr

from pux_harness.agent.reasoning import ReasoningChatOpenAI

_log = logging.getLogger(__name__)

_YAML = Path(__file__).resolve().parent / "models.yaml"

# The four shipped roles. The profile-loader validates an org's `models:` map
# against this set so a typo (e.g. `grader_modle:`) fails at contract time.
ROLES: tuple[str, ...] = ("base", "worker", "multimodal", "grader")
ROLE_KEYS: tuple[str, ...] = tuple(f"{r}_model" for r in ROLES)
# The per-role fallback-chain key inside a tier: ``<role>_fallbacks`` is an
# ordered list of model ids tried in turn when the primary ``<role>_model``
# raises a transient API error after its own ``max_retries``. A fallback chain
# applies ONLY when the role resolved from the tier — an explicit override
# (frontmatter ``model:``, org ``models:``, ``PUX_<ROLE>_MODEL``) is a HARD pin.
FALLBACK_KEYS: tuple[str, ...] = tuple(f"{r}_fallbacks" for r in ROLES)

# Transient provider errors that warrant failing over to the next declared
# model. Deliberately NARROW: 401/400/404 (auth / bad-request / not-found) are
# config or request bugs that would recur on every fallback, so they are NOT in
# the set — the call dies loud instead of silently burning the chain. The
# primary's ``max_retries=6`` already rides transient 429s via backoff; this set
# catches the case where retries EXHAUST or the model is persistently down.
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

# Transient exceptions for the Anthropic-compat provider (zai-anthropic). Same
# NARROW policy as the OpenAI set: 401/400/404 (auth / bad-request / not-found)
# are config/request bugs that would recur on every fallback, so they stay OUT —
# the call dies loud instead of silently burning a chain. Used by the
# Anthropic-kind fallback wrapper so a glm-5.2 429/timeout/5xx fails over to
# glm-5.1 (same provider kind — chains are protocol-homogeneous).
_ANTHROPIC_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)

# Legacy single-model-era env vars still honored (back-compat).
_LEGACY_DEFAULT_MODEL_ENV = "PUX_MODEL"        # base role only
_LEGACY_BASE_URL_ENV = "OPENCODE_BASE_URL"
# The runtime tier selector (set by --tier/--fast). Falls back to default_tier.
_TIER_ENV = "PUX_TIER"


@lru_cache(maxsize=1)
def _spec() -> dict:
    """The parsed ``models.yaml`` (cached for the process). Raises RuntimeError
    on a missing/malformed spec so the failure is loud at the first model build
    (and at ``--check-contract`` via ``validate_models_spec``)."""
    if not _YAML.is_file():
        raise RuntimeError(f"models.yaml not found at {_YAML}")
    data = yaml.safe_load(_YAML.read_text()) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"models.yaml: top level must be a mapping, got {type(data).__name__}")
    for key in ("providers", "default_provider", "tiers", "default_tier"):
        if key not in data:
            raise RuntimeError(
                f"models.yaml must define `providers:`, `default_provider:`, "
                f"`tiers:`, `default_tier:`; missing: {key}")
    providers = data["providers"]
    if not isinstance(providers, dict) or not providers:
        raise RuntimeError(
            f"models.yaml `providers:` must be a non-empty mapping of named "
            f"profiles, got {type(providers).__name__}")
    for pname, prof in providers.items():
        if not isinstance(prof, dict):
            raise RuntimeError(
                f"models.yaml provider {pname!r} must be a mapping, got {type(prof).__name__}")
        kind = prof.get("kind", "openai")
        if kind not in ("openai", "anthropic"):
            raise RuntimeError(
                f"models.yaml provider {pname!r}: kind {kind!r} must be 'openai' or 'anthropic'")
        for req in ("base_url", "api_key_env"):
            val = prof.get(req)
            if not isinstance(val, str) or not val:
                raise RuntimeError(
                    f"models.yaml provider {pname!r}: {req} must be a non-empty string")
    default_provider = data["default_provider"]
    if not isinstance(default_provider, str) or default_provider not in providers:
        raise RuntimeError(
            f"models.yaml `default_provider: {default_provider!r}` must name a "
            f"profile in `providers:` (known: {sorted(providers)})")
    tiers = data["tiers"]
    if not isinstance(tiers, dict) or not tiers:
        raise RuntimeError(f"models.yaml `tiers:` must be a non-empty mapping, got {type(tiers).__name__}")
    for tname, tmap in tiers.items():
        if not isinstance(tmap, dict):
            raise RuntimeError(f"models.yaml tier {tname!r} must be a mapping, got {type(tmap).__name__}")
        missing = [k for k in ROLE_KEYS if k not in tmap]
        if missing:
            raise RuntimeError(f"models.yaml tier {tname!r} missing role key(s): {missing}")
        # ``<role>_fallbacks`` (when declared) must be a non-empty list of
        # unique model-id strings — anything else fails loud (no silent skip of
        # a malformed chain that would leave a role with no real failover).
        for fkey in FALLBACK_KEYS:
            if fkey not in tmap:
                continue
            fbs = tmap[fkey]
            if not isinstance(fbs, list) or not fbs:
                raise RuntimeError(
                    f"models.yaml tier {tname!r}: {fkey} must be a non-empty list "
                    f"of model ids, got {type(fbs).__name__}")
            for fb in fbs:
                if not isinstance(fb, str) or not fb:
                    raise RuntimeError(
                        f"models.yaml tier {tname!r}: {fkey} entries must be "
                        f"non-empty strings, got {fb!r}")
            if len(set(fbs)) != len(fbs):
                raise RuntimeError(
                    f"models.yaml tier {tname!r}: {fkey} has duplicate ids: {fbs}")
    default_tier = data["default_tier"]
    if not isinstance(default_tier, str) or default_tier not in tiers:
        raise RuntimeError(
            f"models.yaml `default_tier: {default_tier!r}` must name a tier in `tiers:` "
            f"(known: {sorted(tiers)})")
    if "models" in data and not isinstance(data["models"], dict):
        raise RuntimeError(f"models.yaml `models:` must be a mapping, got {type(data['models']).__name__}")
    if "dcode" in data and not isinstance(data["dcode"], dict):
        raise RuntimeError(f"models.yaml `dcode:` must be a mapping, got {type(data['dcode']).__name__}")
    return data


def validate_models_spec() -> None:
    """Offline contract entry point — exercises ``_spec()`` so a malformed
    ``models.yaml`` fails ``--check-contract``, not the first live run. Raises
    RuntimeError on any problem; returns None when the spec is well-formed.

    Also cross-checks each tier's ``multimodal_model`` is flagged multimodal in
    the ``models:`` registry (when the id is KNOWN) — a non-multimodal vision
    model would silently break ``describe_image`` at runtime, so fail it here."""
    spec = _spec()
    registry = spec.get("models") or {}
    for tname, tmap in spec["tiers"].items():
        mm = tmap["multimodal_model"]
        caps = registry.get(mm)
        if isinstance(caps, dict) and caps.get("multimodal") is False:
            raise RuntimeError(
                f"models.yaml tier {tname!r}: multimodal_model {mm!r} is flagged "
                f"non-multimodal in the `models:` registry — vision (describe_image) "
                f"would break. Pick a multimodal-capable id or flag it true.")


def _providers() -> dict:
    return _spec()["providers"]


def _default_provider_name() -> str:
    return _spec()["default_provider"]


def _provider_profile(name: str) -> dict:
    """The named profile dict (validated by ``_spec()``). Raises RuntimeError
    (loud, at first model build) when ``name`` is not a declared profile — a
    typo in a model's ``provider:`` binding fails here, not as a silent default."""
    prof = _providers().get(name)
    if prof is None:
        raise RuntimeError(
            f"models.yaml: provider profile {name!r} is not declared in "
            f"`providers:` (known: {sorted(_providers())})")
    return prof


def _provider_for_model(model_id: str) -> dict:
    """The provider profile serving ``model_id``: the profile the model names in
    the ``models:`` registry, else ``default_provider``. Single resolution path
    so a model's protocol (openai vs anthropic) is unambiguous everywhere."""
    registry = _models_registry()
    caps = registry.get(model_id)
    if isinstance(caps, dict):
        named = caps.get("provider")
        if isinstance(named, str) and named:
            return _provider_profile(named)
    return _provider_profile(_default_provider_name())


def _provider_kind(model_id: str) -> str:
    """``"openai"`` or ``"anthropic"`` — the wire protocol for ``model_id``."""
    return _provider_for_model(model_id).get("kind", "openai")


def _transient_exceptions_for(model_id: str) -> tuple[type[BaseException], ...]:
    """The transient-error set for ``model_id``'s provider kind — OpenAI's or
    Anthropic's SDK exceptions. A fallback chain is protocol-homogeneous (all
    entries share the primary's kind), so the primary's kind picks the set."""
    return (
        _ANTHROPIC_TRANSIENT_EXCEPTIONS
        if _provider_kind(model_id) == "anthropic"
        else _TRANSIENT_EXCEPTIONS
    )


def _tiers() -> dict:
    return _spec()["tiers"]


def active_tier() -> str:
    """The tier roles resolve from: ``PUX_TIER`` (set by --tier/--fast) if it
    names a real tier, else ``default_tier``. Raises ValueError on an UNKNOWN
    ``PUX_TIER`` so a typo fails loud rather than silently falling back."""
    env_tier = os.environ.get(_TIER_ENV)
    tiers = _tiers()
    if env_tier:
        if env_tier not in tiers:
            raise ValueError(
                f"PUX_TIER={env_tier!r} is not a known tier; known: {sorted(tiers)}")
        return env_tier
    return _spec()["default_tier"]


def _tier_role(role: str, tier: str) -> str:
    roles = _tiers()[tier]
    key = f"{role}_model"
    # _spec() already guaranteed every tier carries every ROLE_KEY.
    return roles[key]


def _models_registry() -> dict:
    return _spec().get("models") or {}


def _dcode_block() -> dict:
    return _spec().get("dcode") or {}


def _org_role_override(org: str, role_key: str) -> str | None:
    """Read the ``models:`` map -> the id for ``role_key``. None when the org
    (and every ancestor) ships no profile, no ``models:`` block, or no entry for
    this role. Reuses ``profile._resolved_profile_yaml`` so the path resolution
    + non-mapping guard apply AND org inheritance is honored (a child
    inherits a parent's ``models:`` map and overrides the roles it restates; the
    single-source-of-truth discipline — the contract tests' monkeypatch
    of ``orgs._orgs_dir`` reaches here too)."""
    from pux_harness.agent import profile as _profile  # noqa: PLC0415 — avoid import cycle
    data = _profile._resolved_profile_yaml(org)
    if not data:
        return None
    models = data.get("models")
    if not isinstance(models, dict):
        return None
    val = models.get(role_key)
    return val if isinstance(val, str) and val else None


def _tier_fallbacks(role: str, tier: str) -> list[str]:
    """The declared ``<role>_fallbacks`` chain for ``tier`` — empty when the tier
    declares none for this role. Returns a copy so callers can't mutate the
    cached spec. ``_spec()`` already guaranteed any declared chain is a
    non-empty list of unique non-empty strings."""
    fbs = _tiers()[tier].get(f"{role}_fallbacks") or []
    return [str(fb) for fb in fbs]


def _role_overridden(role: str, org: str | None, model: str | None) -> bool:
    """True when the role's id is an EXPLICIT override (frontmatter / org / env)
    rather than the active tier's default. A fallback chain applies ONLY to the
    tier default — an explicit pin is the operator saying "use exactly this
    model", so no failover (mirrors the priority stack in ``resolve_model_id``
    and honors no-silent-fallbacks: an override is never quietly swapped)."""
    if model:
        return True
    role_key = f"{role}_model"
    if org is not None and _org_role_override(org, role_key):
        return True
    if os.environ.get(f"PUX_{role.upper()}_MODEL"):
        return True
    if role == "base" and os.environ.get(_LEGACY_DEFAULT_MODEL_ENV):
        return True
    return False


def resolve_fallback_ids(
    *, role: str = "base", org: str | None = None, model: str | None = None,
) -> list[str]:
    """The ordered fallback model ids for ``role`` against ``org`` — empty when
    the role is an explicit override (frontmatter / org / env) OR the active
    tier declares no ``<role>_fallbacks``. Public so tests + the contract
    checker can inspect the chain WITHOUT building ``ChatOpenAI`` instances."""
    if role not in ROLES:
        raise ValueError(f"unknown model role {role!r}; known roles: {ROLES}")
    if _role_overridden(role, org, model):
        return []
    return _tier_fallbacks(role, active_tier())


def resolve_model_id(
    *, role: str = "base", org: str | None = None, model: str | None = None,
) -> str:
    """Resolve a role to a concrete model id via the priority stack. Public so
    tests + the contract checker can assert the id WITHOUT building a
    ``ChatOpenAI`` (no key, no network)."""
    if role not in ROLES:
        raise ValueError(f"unknown model role {role!r}; known roles: {ROLES}")
    # 1. caller-supplied literal (subagent frontmatter `model:`).
    if model:
        return model
    role_key = f"{role}_model"
    # 2. org profile.yaml `models:` override.
    if org is not None:
        org_val = _org_role_override(org, role_key)
        if org_val:
            return org_val
    # 3. env PUX_<ROLE>_MODEL (and the legacy PUX_MODEL for the base role).
    env_val = os.environ.get(f"PUX_{role.upper()}_MODEL")
    if env_val:
        return env_val
    if role == "base":
        legacy = os.environ.get(_LEGACY_DEFAULT_MODEL_ENV)
        if legacy:
            return legacy
    # 4. the active tier's role map.
    return _tier_role(role, active_tier())


class _FallbackReasoningChatOpenAI(ReasoningChatOpenAI):
    """A ``ReasoningChatOpenAI`` carrying a LangChain ``with_fallbacks`` chain.

    WHY THIS CLASS EXISTS (the deepagents seam)
      ``get_model`` previously returned ``primary.with_fallbacks(chain)`` — a
      ``RunnableWithFallbacks``. But ``deepagents.create_deep_agent`` calls
      ``resolve_model(model)`` (``_models.py``), whose fast-path is
      ``isinstance(model, BaseChatModel)``. ``RunnableWithFallbacks`` is a
      ``Runnable``, NOT a ``BaseChatModel`` → the fast-path misses → deepagents
      falls through to ``init_chat_model(model, ...)`` →
      ``apply_provider_profile(model)`` → ``get_provider_profile(model)`` →
      ``spec.count(":")`` → ``AttributeError`` (caught live at ``pux serve``
      startup: "'ReasoningChatOpenAI' object has no attribute 'count'"). LangChain
      ships NO ``ChatModelWithFallbacks`` adapter — only ``RunnableWithFallbacks``
      — so the fallback-bearing model MUST be a ``BaseChatModel`` subclass we own.

    This subclass IS a ``BaseChatModel`` (via ``ReasoningChatOpenAI``) so
    ``resolve_model``'s fast-path returns it unchanged, AND it carries the chain.
    The failover itself stays upstream (LangChain ``with_fallbacks``): the hot
    path is ``bind_tools`` — deepagents always binds the supervisor's tools, and
    a bound model is a ``Runnable`` the supervisor drives — so
    ``bind_tools`` returns ``super().bind_tools(...).with_fallbacks(
    [fb.bind_tools(...) ...], exceptions_to_handle=...)``. ``with_structured_output``
    mirrors that. The defensive ``_generate`` (off the agent hot path — the model
    is reached bound, not bare) re-runs the same failover inline so the LangChain
    callback chain is preserved on each attempt with the SAME
    ``exceptions_to_handle`` semantics as ``with_fallbacks``.

    Profile matching (``_harness_profile_for_model`` → ``get_model_identifier`` /
    ``get_model_provider``) reads ``model_name`` / ``_get_ls_params()``, inherited
    unchanged from ``ChatOpenAI`` — so it reports the PRIMARY id (e.g.
    ``glm-5.2``), not the chain.
    """

    _fallback_models: list = PrivateAttr(default_factory=list)
    _fallback_exceptions: tuple = PrivateAttr(default_factory=tuple)

    def _chain(self, runnable):
        """Wrap a bound/structured ``Runnable`` with the fallback chain (upstream
        ``with_fallbacks``). No chain → the runnable unchanged."""
        if not self._fallback_models:
            return runnable
        return runnable.with_fallbacks(
            self._fallback_models, exceptions_to_handle=self._fallback_exceptions)

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self._chain(super().bind_tools(tools, **kwargs))

    def with_structured_output(self, schema, **kwargs):  # type: ignore[override]
        chain = [fb.with_structured_output(schema, **kwargs) for fb in self._fallback_models]
        bound = super().with_structured_output(schema, **kwargs)
        if not chain:
            return bound
        return bound.with_fallbacks(chain, exceptions_to_handle=self._fallback_exceptions)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if not self._fallback_models:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except self._fallback_exceptions:
            pass
        last_exc: BaseException | None = None
        for fb in self._fallback_models:
            try:
                return fb._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except self._fallback_exceptions as exc:
                last_exc = exc
        assert last_exc is not None  # _fallback_models non-empty → loop ran ≥1
        raise last_exc


class _FallbackChatAnthropic(ChatAnthropic):
    """Anthropic-protocol twin of ``_FallbackReasoningChatOpenAI`` — a
    ``ChatAnthropic`` carrying a LangChain ``with_fallbacks`` chain.

    WHY a SECOND class (not a generic wrapper)
      Same deepagents seam as the OpenAI variant: ``create_deep_agent``'s
      ``resolve_model`` fast-path needs ``isinstance(model, BaseChatModel)``, and
      ``RunnableWithFallbacks`` is a ``Runnable`` not a ``BaseChatModel``. LangChain
      ships no ``ChatModelWithFallbacks`` adapter, so a fallback-bearing model MUST
      be a ``BaseChatModel`` subclass we own. This one subclasses ``ChatAnthropic``
      (so the Anthropic-protocol glm-5.2 primary is driven with native Anthropic
      behavior) and carries the chain via the same upstream ``with_fallbacks``.

      A chain is protocol-homogeneous (all Anthropic — ``base_fallbacks`` on an
      anthropic primary must name anthropic ids), so every chain model is also a
      ``ChatAnthropic`` and the ``bind_tools`` / ``with_structured_output`` /
      ``_generate`` overrides below mirror ``_FallbackReasoningChatOpenAI`` exactly.

      Profile matching (``_harness_profile_for_model`` → ``get_model_identifier``
      / ``get_model_provider``) reads ``model`` / ``_get_ls_params()``, inherited
      unchanged from ``ChatAnthropic`` — so it reports the PRIMARY id (e.g.
      ``glm-5.2``), not the chain.
    """

    _fallback_models: list = PrivateAttr(default_factory=list)
    _fallback_exceptions: tuple = PrivateAttr(default_factory=tuple)

    def _chain(self, runnable):
        """Wrap a bound/structured ``Runnable`` with the fallback chain (upstream
        ``with_fallbacks``). No chain → the runnable unchanged."""
        if not self._fallback_models:
            return runnable
        return runnable.with_fallbacks(
            self._fallback_models, exceptions_to_handle=self._fallback_exceptions)

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self._chain(super().bind_tools(tools, **kwargs))

    def with_structured_output(self, schema, **kwargs):  # type: ignore[override]
        chain = [fb.with_structured_output(schema, **kwargs) for fb in self._fallback_models]
        bound = super().with_structured_output(schema, **kwargs)
        if not chain:
            return bound
        return bound.with_fallbacks(chain, exceptions_to_handle=self._fallback_exceptions)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if not self._fallback_models:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except self._fallback_exceptions:
            pass
        last_exc: BaseException | None = None
        for fb in self._fallback_models:
            try:
                return fb._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except self._fallback_exceptions as exc:
                last_exc = exc
        assert last_exc is not None  # _fallback_models non-empty → loop ran ≥1
        raise last_exc


def _instantiate(
    model_id: str,
    *,
    fallback_models: list[BaseChatModel] | None = None,
    fallback_exceptions: tuple[type[BaseException], ...] = (),
) -> BaseChatModel:
    """Build the chat model for ``model_id`` against ITS provider profile.

    The provider kind (openai vs anthropic — resolved from the model's profile)
    picks the concrete class: ``ReasoningChatOpenAI`` for OpenAI-compatible
    endpoints, ``ChatAnthropic`` for Anthropic-compatible ones (e.g. ZAI's
    glm-5.2, served where its billing balance lives). ``max_tokens`` /
    ``temperature`` come from the profile (with PUX_MAX_TOKENS / PUX_TEMPERATURE
    env overrides); the OpenCode-Go profile's ``base_url`` still honors the
    legacy ``OPENCODE_BASE_URL`` override (matched by api_key_env) for back-compat.

    When ``fallback_models`` is given, returns the kind's fallback wrapper — a
    ``BaseChatModel`` subclass (so ``deepagents.resolve_model``'s fast-path
    accepts it, not a ``RunnableWithFallbacks`` which crashes it) carrying the
    chain via LangChain ``with_fallbacks``. A chain is protocol-homogeneous
    (every entry shares the primary's kind), so the wrapper class matches the
    primary's.

    ``max_retries=6`` lets the client ride transient limits (429s on the free
    OpenCode Zen Go router; ZAI throttles) with built-in exponential backoff
    rather than dying on the first 429. Standard client resilience, not a
    behavior fallback — and the precondition for ``get_model``'s declared
    fallback chain (each model is independently resilient; ``with_fallbacks``
    only fires when retries exhaust)."""
    prof = _provider_for_model(model_id)
    kind = prof.get("kind", "openai")
    base_url = prof["base_url"]
    api_key_env = prof["api_key_env"]  # _spec() validated non-empty
    # Legacy back-compat: OPENCODE_BASE_URL overrides the OpenCode-Go profile's
    # base_url (matched by its api_key_env) so existing dev workflows keep working.
    if api_key_env == "OPENCODE_API_KEY":
        legacy = os.environ.get(_LEGACY_BASE_URL_ENV)
        if legacy:
            base_url = legacy
    api_key = os.environ[api_key_env]
    max_tokens = int(os.environ.get("PUX_MAX_TOKENS", prof.get("max_tokens", 8192)))
    temperature = float(os.environ.get("PUX_TEMPERATURE", prof.get("temperature", 0.2)))

    if kind == "anthropic":
        kwargs = dict(
            model=model_id, base_url=base_url, api_key=api_key,
            max_tokens=max_tokens, temperature=temperature, timeout=180, max_retries=6,
        )
        if fallback_models:
            m = _FallbackChatAnthropic(**kwargs)
            m._fallback_models = list(fallback_models)
            m._fallback_exceptions = tuple(fallback_exceptions)
            return m
        return ChatAnthropic(**kwargs)

    # openai-compatible
    kwargs = dict(
        model=model_id, base_url=base_url, api_key=api_key, timeout=180, max_retries=6,
        max_tokens=max_tokens, temperature=temperature,
    )
    if fallback_models:
        m = _FallbackReasoningChatOpenAI(**kwargs)
        m._fallback_models = list(fallback_models)
        m._fallback_exceptions = tuple(fallback_exceptions)
        return m
    return ReasoningChatOpenAI(**kwargs)


def get_model(
    *, role: str = "base", org: str | None = None, model: str | None = None,
) -> BaseChatModel:
    """Build the chat model for ``role`` resolved against ``org``, or for the
    literal ``model`` id when one is supplied (the subagent-frontmatter override
    path). The model's provider profile (from ``models.yaml``) picks the concrete
    class — ``ReasoningChatOpenAI`` (OpenAI-compatible) or ``ChatAnthropic``
    (Anthropic-compatible, e.g. ZAI glm-5.2). Legacy env overrides
    (``OPENCODE_BASE_URL``, ``PUX_MAX_TOKENS``, ``PUX_TEMPERATURE``) still win.

    When the role resolves from the active tier AND that tier declares a
    ``<role>_fallbacks`` chain, the returned model is a
    ``_FallbackReasoningChatOpenAI`` (a ``BaseChatModel`` subclass — NOT a bare
    ``RunnableWithFallbacks``, which crashes ``deepagents.resolve_model``) whose
    ``bind_tools`` / ``with_structured_output`` / ``_generate`` wrap the chain
    via LangChain's ``with_fallbacks``: a transient error (429/timeout/5xx — see
    ``_TRANSIENT_EXCEPTIONS``) that exhausts the primary's ``max_retries`` fails
    over to the next declared id. This is a DECLARED, per-tier chain (auditable
    in ``models.yaml`` + WARNING-logged when attached) — not a silent swap; and
    an explicit override (frontmatter / org / env) disables it (a hard pin).
    The failover is LangChain's own ``with_fallbacks`` (rely-on-upstream): the
    primary IS a ``ReasoningChatOpenAI`` so ``bind_tools`` etc. inherit upstream
    behavior unchanged, only adapted to carry the chain."""
    model_id = resolve_model_id(role=role, org=org, model=model)
    fb_ids = resolve_fallback_ids(role=role, org=org, model=model)
    if not fb_ids:
        return _instantiate(model_id)
    chain = [_instantiate(fb) for fb in fb_ids]
    _log.warning(
        "model role %r: primary %s with declared fallback chain %s "
        "(transient-error failover; declared in models.yaml tier %r)",
        role, model_id, fb_ids, active_tier(),
    )
    return _instantiate(
        model_id, fallback_models=chain,
        fallback_exceptions=_transient_exceptions_for(model_id),
    )


# --- capability + dcode seams ----------------------------------------------


def is_multimodal(model_id: str) -> bool:
    """True iff ``model_id`` is flagged ``multimodal: true`` in the ``models:``
    registry. Unknown ids default to False (fail-safe: a driver of unknown
    capability takes the describe_image fallback, never image blocks it can't
    read)."""
    caps = _models_registry().get(model_id)
    if not isinstance(caps, dict):
        return False
    return bool(caps.get("multimodal", False))


def driver_multimodal(*, role: str = "base", org: str | None = None) -> bool:
    """True iff the model resolved for ``role`` (against ``org`` + the active
    tier) is multimodal-capable. The seam ``BrowserVisionMiddleware`` uses to
    decide image-block (multimodal driver) vs text-pointer (non-multimodal →
    call ``describe_image``). No key, no network — pure id resolution + lookup."""
    return is_multimodal(resolve_model_id(role=role, org=org))


def dcode_provider() -> str | None:
    """dcode's provider name for our endpoint (``models.yaml`` ``dcode.provider``),
    or None when unset. ``pux tui`` forwards ``<provider>:<id>`` to dcode so it
    never falls back to its own default model."""
    provider = _dcode_block().get("provider")
    return provider if isinstance(provider, str) and provider else None


def dcode_model_ref(
    *, role: str = "base", org: str | None = None, model: str | None = None,
) -> str | None:
    """The dcode ``provider:model`` string for ``role`` (e.g.
    ``opencode-go-openai:glm-5.2``), or None when no dcode provider is
    configured. Resolves the id through the same priority stack as
    ``resolve_model_id`` so the TUI forwards exactly what the harness would
    drive."""
    provider = dcode_provider()
    if not provider:
        return None
    return f"{provider}:{resolve_model_id(role=role, org=org, model=model)}"
