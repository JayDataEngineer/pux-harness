"""Per-org harness profile loader (OPTIONAL ``orgs/<org>/profile.yaml``).

A profile lets an org apply small overrides to the deepagents stack the harness
compiles for it — a system-prompt suffix (org-wide), per-tool description
rewrites, and tool exclusions. The user's stated use case: when the shared
browser agent (``orgs/_shared/agents/browser.md``) is rostered into orgs like
deep-research / twitter, an org can nudge the *whole stack* (CTO + every
specialist) without forking the agent itself.

**Why a per-org YAML and NOT the global ``_HARNESS_PROFILES`` registry.** The
registry is a flat ``dict[str, HarnessProfile]`` keyed by *model spec* (e.g.
``openai:gpt-4o``), and ``_get_harness_profile`` rejects any key with more than
one colon — it resolves only ``provider:model`` -> bare ``provider``. There is
no per-org namespace. Two orgs sharing a model (twitter + deep-research both on
``mimo-v2.5``) would merge-collide, and the long-lived ``server.py``/ACP path
builds graphs for multiple orgs in one process -> a cross-org leak. So we reuse
deepagents' own ``HarnessProfileConfig`` SCHEMA (faithful field set) but apply
its three fields directly at the ``build_graph(org)`` call site — collision-free
and server-safe. See ``graph.build_graph`` for the application, and the plan's
Phase 16.3b for the full rationale.

Path resolution calls ``orgs._orgs_dir()`` at runtime (via the module, not an
import-time binding) — single source of truth, so the contract tests'
monkeypatch of ``orgs._orgs_dir`` covers this module too.

**Phase 17.B — RubricMiddleware verify-gate.** An org may add a ``rubric:``
block to its ``profile.yaml`` to opt into a post-agent grader loop (deepagents'
beta ``RubricMiddleware``). The block is peeled out of the YAML BEFORE
``HarnessProfileConfig.from_dict`` (which rejects unknown keys), so the
deepagents schema stays untouched, and surfaced via ``load_rubric_gate`` +
``default_rubric``. ``load_profile``'s signature is unchanged on purpose: zero
ripple to existing callers/tests (the gate is wired separately in
``graph.build_graph``). The grader IS the tester + reviewer — one non-skippable
gate rather than a subagent the CTO might skip.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from deepagents import HarnessProfileConfig
from deepagents._tools import _apply_tool_description_overrides
from langchain_core.tools import BaseTool

from pux_harness.agent import orgs as _orgs_mod
from pux_harness.agent.model import ROLE_KEYS

__all__ = [
    "RubricGate",
    "MiddlewareOverrides",
    "load_profile",
    "load_rubric_gate",
    "load_middleware_overrides",
    "default_rubric",
    "validate_profile",
    "apply_profile_to_tools",
]


def _profile_path(org: str) -> Path:
    # Resolve via the orgs module at CALL time (not an import-time binding) so
    # the contract tests' monkeypatch of ``orgs._orgs_dir`` reaches this module
    # too — same single-source-of-truth discipline contract.py relies on.
    # Specialists-aware (orgs/<org> then orgs/specialists/<org>) but NON-raising:
    # callers (``_read_profile_yaml``) handle a missing file as ``None``, so an
    # unknown org yields a non-existent path rather than ``FileNotFoundError``.
    base = _orgs_mod._orgs_dir()
    top = base / org
    if top.is_dir():
        return top / "profile.yaml"
    return base / "specialists" / org / "profile.yaml"


@dataclass(frozen=True)
class RubricGate:
    """Per-org ``RubricMiddleware`` verify-gate config (Phase 17.B).

    An org opts into a post-agent grader loop by adding a ``rubric:`` block to
    its ``profile.yaml``. deepagents' ``RubricMiddleware`` (beta) runs the
    ``default`` rubric — a ship-gate checklist ("tests pass", "lint clean",
    "no out-of-scope changes") — using sandbox grader tools after the main agent
    finishes, returns a verdict (``satisfied`` / ``needs_revision`` /
    ``max_iterations_reached`` / ``failed`` / ``grader_error``), and the agent
    revises until ``satisfied`` or ``max_iterations`` is hit. The grader IS the
    tester + reviewer (folded into one non-skippable gate rather than a
    subagent the CTO might skip — the Phase 17 design decision).

    The block is peeled out of the YAML BEFORE ``HarnessProfileConfig.from_dict``
    (which rejects unknown keys), so the deepagents schema stays untouched. See
    ``graph.build_graph`` for the middleware wiring (it resolves the grader
    model via ``get_model(role="grader", org=org)`` — the model is NOT a field
    here; override it per-org under the top-level ``models:`` map, like any
    other role), ``tools.build_grader_tools`` for the grader's sandbox tools,
    and ``server._execute`` / ``main._run`` for the default-rubric injection.

    Beta mitigation: the gate is per-org opt-in (only orgs that add the block)
    and behind ``enabled: true``; orgs without a block are byte-identical to
    today. A future deepagents API break hits only opted-in orgs and is killed
    by flipping ``enabled: false``.
    """

    enabled: bool = True
    max_iterations: int = 3
    default: str | None = None  # the ship-gate rubric text


@dataclass(frozen=True)
class MiddlewareOverrides:
    """Per-org ``middleware:`` override block (Phase 21 — the stack factory).

    The harness factory (``agent.stack``) resolves the middleware stack from
    DEFAULTS (in ``stack.py``) + these per-org DELTAS — the org system as an
    override layer. An org may ADD or REMOVE a registered middleware, scoped to
    the supervisor or to every subagent::

        middleware:
          supervisor:
            add: []           # names from stack.MIDDLEWARE_REGISTRY
            remove: []        # drop a default (e.g. ``routing``)
          subagent:
            add: []           # e.g. put ``session_guide`` on subagents too
            remove: []

    The block is peeled out of the YAML BEFORE ``HarnessProfileConfig.from_dict``
    (which would reject it as an unknown key), the same way ``rubric:`` and
    ``models:`` are peeled. Name + scope VALIDITY (is ``routing`` real? is it
    allowed on subagents?) is checked by ``stack.validate_overrides`` (the
    registry lives in ``stack.py`` — this module stays registry-agnostic to
    avoid a cycle); here we only SHAPE-validate. The deepagents
    ``excluded_middleware`` field is ALSO honored (as an unscoped supervisor
    remove) inside the factory — both forms work, the scoped block is primary.
    """

    supervisor_add: frozenset[str] = frozenset()
    supervisor_remove: frozenset[str] = frozenset()
    subagent_add: frozenset[str] = frozenset()
    subagent_remove: frozenset[str] = frozenset()


def _validate_models_block(org: str, data: dict) -> None:
    """Validate the top-level ``models:`` map (Phase 17.B.0).

    Keys must be ⊆ ``model.ROLE_KEYS`` (``base_model`` / ``worker_model`` /
    ``multimodal_model`` / ``grader_model``); values must be non-empty strings.
    Raises ``TypeError`` on a bad shape so a typo (``grader_modle:``) fails at
    load / contract time — otherwise the org would silently fall back to the
    shipped default and wonder why its override isn't taking. No silent skip."""
    models = data.get("models")
    if models is None:
        return
    if not isinstance(models, dict):
        msg = (
            f"{org}/profile.yaml: models: must be a mapping, "
            f"got {type(models).__name__}"
        )
        raise TypeError(msg)
    known = set(ROLE_KEYS)
    unknown = set(models) - known
    if unknown:
        msg = (
            f"{org}/profile.yaml: models: unknown key(s) {sorted(unknown)}; "
            f"valid keys: {sorted(known)}"
        )
        raise TypeError(msg)
    for key, val in models.items():
        if not isinstance(val, str) or not val:
            msg = (
                f"{org}/profile.yaml: models.{key} must be a non-empty "
                f"string, got {val!r}"
            )
            raise TypeError(msg)


def _read_profile_yaml(org: str) -> dict | None:
    """Read + parse THIS org's OWN ``profile.yaml`` -> mapping; ``None`` if absent.

    The RAW single-hop reader — no inheritance. Shared by the contract's raw
    per-file checks (``_no_legacy_subagents_block``) and as the building block
    for ``_resolved_profile_yaml`` (the inheritance-aware reader the runtime
    loaders use). Validates the top-level ``models:`` map
    (``_validate_models_block``) so a bad role key fails every reader, not just
    the model one. A non-mapping top level (e.g. a bare list) raises
    ``TypeError`` — no silent skip; a malformed profile is a real bug."""
    path = _profile_path(org)
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        msg = (
            f"{org}/profile.yaml: top level must be a mapping, "
            f"got {type(data).__name__}"
        )
        raise TypeError(msg)
    _validate_models_block(org, data)
    return data


def _deep_merge_profile(base: Any, delta: Any) -> Any:
    """Deep-merge two parsed ``profile.yaml`` dicts root→child (``delta`` = child
    wins). ONE universal rule that happens to be correct for EVERY field — the
    native ``HarnessProfileConfig`` merge semantics + the pux blocks:

    * dicts merge per-key (recurse) — so ``tool_description_overrides``,
      ``general_purpose_subagent``, ``models``, and ``middleware``'s scope
      sub-blocks all compose key-by-key with the child winning each key it sets;
    * lists UNION (dedup, base order preserved) — so ``excluded_tools``,
      ``excluded_middleware``, and ``middleware.{scope}.{add,remove}`` accumulate
      down the chain;
    * scalars: ``delta`` wins — so ``base_system_prompt`` / ``system_prompt_suffix``
      (child replaces) and every leaf in ``rubric`` / ``models`` (child wins).

    Type-mismatch (dict vs list vs scalar) falls back to ``delta`` wins — the
    child explicitly restated it, which is the honest resolution. Pure +
    recursive; no knowledge of which key is which (the rule is the same
    everywhere, which is what makes it universal)."""
    if isinstance(base, dict) and isinstance(delta, dict):
        out = dict(base)
        for key, dval in delta.items():
            out[key] = _deep_merge_profile(out[key], dval) if key in out else dval
        return out
    if isinstance(base, list) and isinstance(delta, list):
        merged = list(base)
        for item in delta:
            if item not in merged:
                merged.append(item)
        return merged
    return delta


def _resolved_profile_yaml(org: str) -> dict | None:
    """Read + merge this org's ``profile.yaml`` WITH its ``extends:`` chain
    (root→child, deepest-ancestor first). ``None`` when the org AND every
    ancestor ships no profile. Each ancestor's OWN block is read +
    ``models``-validated by ``_read_profile_yaml``; the merged dict is composed
    by ``_deep_merge_profile``.

    This is the inheritance-aware reader the RUNTIME loaders
    (``load_profile`` / ``load_rubric_gate`` / ``load_middleware_overrides`` /
    ``model._org_role_override``) route through, so a child inherits a parent's
    ``system_prompt_suffix`` / ``models`` / ``middleware`` deltas while overriding
    the keys it restates. Cycle-safe: a broken chain falls back to ``[org]``
    (``_read_profile_yaml(org)`` alone) — the contract's ``org-extends-*`` rules
    report the real fault offline. For a non-extending org the chain is ``[org]``
    → byte-identical to the raw own read."""
    try:
        chain = _orgs_mod.org_extends_chain(org)  # root→child
    except (ValueError, FileNotFoundError):
        chain = [org]
    merged: dict | None = None
    for ancestor in chain:
        own = _read_profile_yaml(ancestor)
        if own is None:
            continue
        merged = own if merged is None else _deep_merge_profile(merged, own)
    return merged


def _rubric_gate_from_block(org: str, block: object) -> RubricGate:
    """Build a ``RubricGate`` from the parsed ``rubric:`` block.

    Validates shape (``enabled`` bool, ``max_iterations`` a positive int,
    ``default`` a string) so a typo fails loud at load / contract time, not at
    the first invoke. Unknown keys are rejected — in particular the legacy
    ``rubric.grader_model`` (the grader model moved to the top-level ``models:``
    map in Phase 17.B.0; surface it there as ``grader_model: <id>``). No silent
    skip — a stale form is a real bug."""
    if not isinstance(block, dict):
        msg = (
            f"{org}/profile.yaml: rubric: must be a mapping, "
            f"got {type(block).__name__}"
        )
        raise TypeError(msg)
    known = {"enabled", "max_iterations", "default"}
    unknown = set(block) - known
    if unknown:
        msg = (
            f"{org}/profile.yaml: rubric: unknown key(s) {sorted(unknown)}; "
            f"valid keys: {sorted(known)}. (grader_model moved to the top-level "
            f"`models:` map in Phase 17.B.0.)"
        )
        raise TypeError(msg)
    enabled = block.get("enabled", True)
    if not isinstance(enabled, bool):
        msg = (
            f"{org}/profile.yaml: rubric.enabled must be a bool, "
            f"got {type(enabled).__name__}"
        )
        raise TypeError(msg)
    max_iterations = block.get("max_iterations", 3)
    # bool is a subclass of int — reject it explicitly so `true` (parsed as
    # bool) isn't silently accepted as 1.
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        msg = (
            f"{org}/profile.yaml: rubric.max_iterations must be an int, "
            f"got {type(max_iterations).__name__}"
        )
        raise TypeError(msg)
    if max_iterations < 1:
        msg = (
            f"{org}/profile.yaml: rubric.max_iterations must be >= 1, "
            f"got {max_iterations}"
        )
        raise ValueError(msg)
    default = block.get("default")
    if default is not None and not isinstance(default, str):
        msg = (
            f"{org}/profile.yaml: rubric.default must be a string, "
            f"got {type(default).__name__}"
        )
        raise TypeError(msg)
    return RubricGate(enabled=enabled, max_iterations=max_iterations, default=default)


def _parse_scope_block(org: str, scope: str, block: object) -> tuple[frozenset[str], frozenset[str]]:
    """Parse one ``supervisor:``/``subagent:`` sub-block -> ``(add, remove)``.

    Shape-validated only (name + scope validity is ``stack.validate_overrides``'s
    job — this module is registry-agnostic to avoid a cycle). Each sub-block is a
    mapping with keys ⊆ ``{add, remove}``; each value is a list of non-empty
    strings. A bad shape raises ``TypeError`` so a typo fails at load /
    contract time, not at the first build. No silent skip."""
    if not isinstance(block, dict):
        msg = (
            f"{org}/profile.yaml: middleware.{scope}: must be a mapping, "
            f"got {type(block).__name__}"
        )
        raise TypeError(msg)
    known = {"add", "remove"}
    unknown = set(block) - known
    if unknown:
        msg = (
            f"{org}/profile.yaml: middleware.{scope}: unknown key(s) "
            f"{sorted(unknown)}; valid keys: {sorted(known)}"
        )
        raise TypeError(msg)
    out: dict[str, frozenset[str]] = {}
    for key in ("add", "remove"):
        raw = block.get(key, [])
        if not isinstance(raw, list):
            msg = (
                f"{org}/profile.yaml: middleware.{scope}.{key}: must be a list, "
                f"got {type(raw).__name__}"
            )
            raise TypeError(msg)
        names: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                msg = (
                    f"{org}/profile.yaml: middleware.{scope}.{key}: each entry "
                    f"must be a non-empty string, got {item!r}"
                )
                raise TypeError(msg)
            names.append(item.strip())
        out[key] = frozenset(names)
    return out["add"], out["remove"]


def _middleware_overrides_from_block(org: str, block: object) -> MiddlewareOverrides:
    """Build ``MiddlewareOverrides`` from the parsed ``middleware:`` block.

    Top-level must be a mapping with keys ⊆ ``{supervisor, subagent}``. An empty
    block (``middleware: {}`` or all-empty lists) is valid and yields empty
    overrides — byte-identical to no block at all."""
    if not isinstance(block, dict):
        msg = (
            f"{org}/profile.yaml: middleware: must be a mapping, "
            f"got {type(block).__name__}"
        )
        raise TypeError(msg)
    known = {"supervisor", "subagent"}
    unknown = set(block) - known
    if unknown:
        msg = (
            f"{org}/profile.yaml: middleware: unknown key(s) {sorted(unknown)}; "
            f"valid keys: {sorted(known)}"
        )
        raise TypeError(msg)
    sup_add, sup_remove = _parse_scope_block(org, "supervisor", block.get("supervisor", {}))
    sub_add, sub_remove = _parse_scope_block(org, "subagent", block.get("subagent", {}))
    return MiddlewareOverrides(
        supervisor_add=sup_add,
        supervisor_remove=sup_remove,
        subagent_add=sub_add,
        subagent_remove=sub_remove,
    )


def load_middleware_overrides(org: str) -> MiddlewareOverrides:
    """Read the ``middleware:`` block from ``orgs/<org>/profile.yaml``.

    Returns EMPTY overrides (the common case — no block, byte-identical stack)
    when the org ships no ``profile.yaml`` or no ``middleware:`` block. When
    present, the block is shape-validated (``_middleware_overrides_from_block``
    raises on a malformed entry). Read independently of ``load_profile`` so the
    factory (``stack.build_stack``) can resolve the middleware stack without
    disturbing the ``HarnessProfileConfig`` path. Name + scope validity is
    checked downstream by ``stack.validate_overrides``.

    Phase 5 — inheritance-aware: reads the ``extends:``-chain-merged block
    (``_resolved_profile_yaml``), so a child inherits a parent's middleware
    deltas and overrides the keys it restates. For a non-extending org,
    byte-identical to the raw own read."""
    data = _resolved_profile_yaml(org)
    if data is None or "middleware" not in data:
        return MiddlewareOverrides()
    return _middleware_overrides_from_block(org, data["middleware"])


def load_profile(org: str) -> HarnessProfileConfig | None:
    """Read ``orgs/<org>/profile.yaml`` -> ``HarnessProfileConfig``, or ``None``.

    ``None`` (no file) is the COMMON case — most orgs ship no profile and the
    ``build_graph`` path is byte-identical to today (the regression guarantee).
    If present, the ``rubric:`` block (Phase 17.B), the ``models:`` map
    (Phase 17.B.0), and the ``middleware:`` block (Phase 21 — the stack factory)
    are PEELED out before ``HarnessProfileConfig.from_dict`` (which would
    otherwise reject them as unknown keys) — the rubric block is surfaced
    separately by ``load_rubric_gate``, the models map is read by the model-role
    resolver (``model._org_role_override``), and the middleware block is read by
    ``load_middleware_overrides``. ``from_dict`` validates the rest of the
    schema: unknown keys + bad shapes raise ``TypeError``; bad
    ``excluded_middleware`` grammar raises ``ValueError``. A non-mapping top
    level raises ``TypeError`` here. No silent skip — a malformed profile is a
    real bug.

    Phase 5 — inheritance-aware: reads the ``extends:``-chain-merged block
    (``_resolved_profile_yaml``), so a child inherits a parent's
    ``system_prompt_suffix`` / ``tool_description_overrides`` / ``excluded_*`` /
    ``general_purpose_subagent`` (native fields deep-merged root→child) and
    overrides the keys it restates. For a non-extending org, byte-identical to
    the raw own read.
    """
    data = _resolved_profile_yaml(org)
    if data is None:
        return None
    # ``rubric`` / ``models`` / ``middleware`` are VALID harness blocks peeled
    # out + read by their own loaders. ``subagents`` is NOT peeled on purpose
    # (Phase 2 fold): the legacy ``profile.yaml`` ``subagents:`` block was
    # replaced by per-agent ``extends:`` + delta frontmatter fields, so leaving
    # it in lets ``HarnessProfileConfig.from_dict`` reject it as an unknown key
    # (and the ``no-legacy-subagents-block`` contract tripwire points the operator
    # at the replacement). No silent skip — a stale block is a real bug.
    peeled = {k: v for k, v in data.items() if k not in ("rubric", "models", "middleware")}
    return HarnessProfileConfig.from_dict(peeled)


def load_rubric_gate(org: str) -> RubricGate | None:
    """Read the ``rubric:`` block from ``orgs/<org>/profile.yaml`` -> ``RubricGate``.

    ``None`` when the org ships no ``profile.yaml`` OR no ``rubric:`` block —
    the common case (no gate, byte-identical to today). When present, the block
    is shape-validated (``_rubric_gate_from_block`` raises on a malformed
    entry). Read independently of ``load_profile`` so the gate can be wired in
    ``build_graph`` without disturbing the ``HarnessProfileConfig`` path
    (load_profile's signature stays stable → zero ripple to existing callers).

    Phase 5 — inheritance-aware: reads the ``extends:``-chain-merged block
    (``_resolved_profile_yaml``); a child inherits a parent's rubric gate and
    overrides the fields it restates. For a non-extending org, byte-identical to
    the raw own read.
    """
    data = _resolved_profile_yaml(org)
    if data is None or "rubric" not in data:
        return None
    return _rubric_gate_from_block(org, data["rubric"])


def default_rubric(org: str) -> str | None:
    """The rubric text to inject at invoke time when the operator supplies none.

    Returns ``RubricGate.default`` ONLY when the gate is present + enabled + has
    a default. ``None`` otherwise (no gate, gate disabled, or no default text).
    ``None`` means ``server._execute`` / ``main._run`` skip injection, so
    ``RubricMiddleware`` does not run (its contract: "When no rubric is supplied
    on input state, the middleware does not run")."""
    gate = load_rubric_gate(org)
    if gate is None or not gate.enabled or not gate.default:
        return None
    return gate.default


def validate_profile(org: str) -> HarnessProfileConfig | None:
    """Offline contract check (no Docker, no model). Exercises EVERY loader so
    a malformed ``rubric:`` block, a bad ``models:`` role key, OR a bad
    ``middleware:`` shape fails ``--check-contract`` too — not just the
    ``HarnessProfileConfig`` schema (all readers route through
    ``_read_profile_yaml`` → ``_validate_models_block``). Raises on malformed;
    ``None`` when the org ships no ``profile.yaml`` (the contract checker treats
    absence as 'skipped', not a violation). Called from ``--check-contract`` for
    every discovered org. (Middleware NAME/scope validity — is ``routing`` real,
    allowed on subagents — is checked by ``stack.validate_overrides``; this only
    shape-checks.)
    """
    cfg = load_profile(org)              # raises on a malformed schema (incl. a
                                        # legacy ``subagents:`` block — Phase 2
                                        # fold: from_dict now rejects that key)
    load_rubric_gate(org)                # raises on a malformed rubric: block
    load_middleware_overrides(org)       # raises on a malformed middleware: block
    return cfg


def apply_profile_to_tools(
    tools: list[BaseTool], cfg: HarnessProfileConfig
) -> list[BaseTool]:
    """Apply ``tool_description_overrides`` + ``excluded_tools`` to a tool list.

    Used at both application sites — the MAIN agent stack (in ``build_graph``)
    and EACH subagent's resolved whitelist (in ``load_subagents``) — so an
    org-wide override reaches the browser subagent, not just the CTO.
    ``_apply_tool_description_overrides`` copies + rewrites (it never mutates
    caller-owned tools), so this is safe to call per-subagent. Filtering by
    ``tool.name`` (the prefixed ``pux_sandbox_*`` identifier the profile keys
    on)."""
    out: list[BaseTool] = tools
    if cfg.tool_description_overrides:
        out = _apply_tool_description_overrides(out, cfg.tool_description_overrides)
    if cfg.excluded_tools:
        out = [t for t in out if t.name not in cfg.excluded_tools]
    return out
