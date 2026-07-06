"""Model spec — roles resolved from a shipped ``models.yaml``.

The single source of truth is ``models.yaml`` next to this module: it pins the
provider (``base_url`` + which env var holds the api key) and the role→model-id
map (``base`` / ``worker`` / ``multimodal`` / ``grader``). No model id is
hardcoded anywhere else in the harness — every consumer asks
``get_model(role=..., org=...)`` and the resolution picks the concrete id.

Resolution priority for a role (highest wins):
  1. ``model=`` literal                — a caller-supplied id (subagent frontmatter
                                         ``model:`` — the agent-level override).
  2. org ``profile.yaml`` ``models:``  — org-level role override.
  3. ``PUX_<ROLE>_MODEL`` env var       — deployment-level (e.g. PUX_GRADER_MODEL);
                                         ``PUX_MODEL`` is honored for the base role
                                         as legacy back-compat.
  4. ``models.yaml`` ``roles:``         — the shipped default.

Provider config (``base_url``, ``api_key_env``, ``max_tokens``, ``temperature``)
comes from ``models.yaml``'s ``provider:`` block, with the legacy env overrides
(``OPENCODE_BASE_URL``, ``PUX_MAX_TOKENS``, ``PUX_TEMPERATURE``) still winning
for back-compat.

A cloner repoints at their own provider by editing ``models.yaml`` — one file,
no code changes, no scattered hardcodes.

Historical note: pre-17.B.0 this module had a single ``DEFAULT_MODEL``/``get_model(model=None)``
that every consumer shared; the rubric gate needs an independent
grader model, which would have meant a second hardcode — the spec exists to make
that a role resolution instead.
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
    if "provider" not in data or "roles" not in data:
        raise RuntimeError(
            "models.yaml must define `provider:` and `roles:`; "
            f"got top-level keys: {sorted(data)}"
        )
    roles = data["roles"]
    if not isinstance(roles, dict):
        raise RuntimeError(f"models.yaml `roles:` must be a mapping, got {type(roles).__name__}")
    missing = [k for k in ROLE_KEYS if k not in roles]
    if missing:
        raise RuntimeError(f"models.yaml `roles:` missing: {missing}")
    return data


def validate_models_spec() -> None:
    """Offline contract entry point — exercises ``_spec()`` so a malformed
    ``models.yaml`` fails ``--check-contract``, not the first live run. Raises
    RuntimeError on any problem; returns None when the spec is well-formed."""
    _spec()


def _provider() -> dict:
    return _spec()["provider"]


def _role_default(role: str) -> str:
    roles = _spec()["roles"]
    key = f"{role}_model"
    if key not in roles:
        raise ValueError(f"unknown model role {role!r}; known roles: {ROLES}")
    return roles[key]


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
    # 4. shipped default.
    return _role_default(role)


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
