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

Capability (``is_multimodal``) is a property of the MODEL ID, read from the
``models:`` registry. Unknown ids default to non-multimodal (fail-safe: a driver
of unknown capability takes the describe_image fallback, never image blocks it
can't read). ``driver_multimodal`` resolves a role to its id then its capability
— the seam ``BrowserVisionMiddleware`` uses to decide image-block vs text-pointer.

Provider config (``base_url``, ``api_key_env``, ``max_tokens``, ``temperature``)
comes from ``models.yaml``'s ``provider:`` block, with the legacy env overrides
(``OPENCODE_BASE_URL``, ``PUX_MAX_TOKENS``, ``PUX_TEMPERATURE``) still winning
for back-compat.

Historical notes:
  - Pre-17.B.0 a single ``DEFAULT_MODEL``/``get_model(model=None)`` was shared by
    every consumer; the rubric gate needs an independent grader model → roles.
  - Pre-tier the file had a flat ``roles:`` map (one role set, the default). That
    made the WEAK model the shipped supervisor — backwards for a "cheap vs SOTA"
    dial. ``tiers:`` + ``default_tier`` replaced it: the default is SOTA, ``fast``
    is the opt-in cheap mode.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_openai import ChatOpenAI

from pux_harness.agent.reasoning import ReasoningChatOpenAI

_YAML = Path(__file__).resolve().parent / "models.yaml"

# The four shipped roles. The profile-loader validates an org's `models:` map
# against this set so a typo (e.g. `grader_modle:`) fails at contract time.
ROLES: tuple[str, ...] = ("base", "worker", "multimodal", "grader")
ROLE_KEYS: tuple[str, ...] = tuple(f"{r}_model" for r in ROLES)

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
    for key in ("provider", "tiers", "default_tier"):
        if key not in data:
            raise RuntimeError(
                f"models.yaml must define `provider:`, `tiers:`, `default_tier:`; "
                f"missing: {key}")
    tiers = data["tiers"]
    if not isinstance(tiers, dict) or not tiers:
        raise RuntimeError(f"models.yaml `tiers:` must be a non-empty mapping, got {type(tiers).__name__}")
    for tname, tmap in tiers.items():
        if not isinstance(tmap, dict):
            raise RuntimeError(f"models.yaml tier {tname!r} must be a mapping, got {type(tmap).__name__}")
        missing = [k for k in ROLE_KEYS if k not in tmap]
        if missing:
            raise RuntimeError(f"models.yaml tier {tname!r} missing role key(s): {missing}")
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


def _provider() -> dict:
    return _spec()["provider"]


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


def get_model(
    *, role: str = "base", org: str | None = None, model: str | None = None,
) -> ChatOpenAI:
    """Build a ``ChatOpenAI`` for ``role`` resolved against ``org``, or for the
    literal ``model`` id when one is supplied (the subagent-frontmatter override
    path). Provider config comes from ``models.yaml``; legacy env overrides
    (``OPENCODE_BASE_URL``, ``PUX_MAX_TOKENS``, ``PUX_TEMPERATURE``) still win."""
    provider = _provider()
    base_url = os.environ.get(_LEGACY_BASE_URL_ENV, provider["base_url"])
    api_key_env = provider.get("api_key_env", "OPENCODE_API_KEY")
    model_id = resolve_model_id(role=role, org=org, model=model)
    return ReasoningChatOpenAI(
        model=model_id,
        base_url=base_url,
        api_key=os.environ[api_key_env],
        timeout=180,
        # OpenCode Zen Go is a free router with a tight per-account rate limit
        # (429 "provider_rate_limit_exceeded"). Let the OpenAI client ride
        # transient limits with built-in exponential backoff (~30-60s across
        # 6 retries) rather than dying on the first 429. Standard client
        # resilience, not a behavior fallback.
        max_retries=6,
        max_tokens=int(os.environ.get("PUX_MAX_TOKENS", provider.get("max_tokens", 8192))),
        temperature=float(os.environ.get("PUX_TEMPERATURE", provider.get("temperature", 0.2))),
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
