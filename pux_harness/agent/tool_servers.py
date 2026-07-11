"""Tool-server dependency resolver — foreign MCP servers declared per-org.

Given an org's ``org.yaml`` ``capabilities:`` (``kind: mcp``) entries (bare
strings → catalog refs, or mappings → inline or catalog-ref-with-override),
``resolve_tool_servers`` produces a list of resolved ``ToolServerSpec`` — the
single shape the async bridge (``mcp_client.py``) consumes. (CU-4: this is the
ONE declaration site — the legacy ``policy.yaml`` ``tool_servers:`` block was
removed and is now a permanent contract failure.)

The catalog lives at ``orgs/_shared/tool_servers.yaml`` and mirrors the
Claude-Code/dcode ``.mcp.json`` schema (``McpServerSpec``). Resolution is
pure-data (sync, no network): no MCP handshake, no live probing.

Allowlist semantics: ``tools:`` absent on a resolved spec → take ALL tools the
server exposes (INFO log). ``tools:`` present → take only those names; a name
the server doesn't expose fails LOUD at load (no silent skip). A catalog ref's
own ``tools:`` overrides (narrows) the catalog entry's allowlist.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pux_harness.agent.orgs import _org_path, _orgs_dir
from pux_harness.kit.capabilities_decl import org_mcp_items_from_dict

_log = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_KNOWN_TRANSPORTS = frozenset({"stdio", "sse", "http"})

_VALID_KINDS = frozenset({"mcp"})

_TRANSPORT_REQUIRED: dict[str, tuple[str, ...]] = {
    "stdio": ("command",),
    "sse": ("url",),
    "http": ("url",),
}


@dataclass
class ToolServerSpec:
    """A fully-resolved foreign tool-server spec — the shape ``mcp_client.py``
    consumes.

    ``kind`` is always ``"mcp"`` in v1 (``http``/``sidecar`` reserved, fail-loud
    if used). ``tools`` is the allowlist — ``None`` means "take everything".

    ``github`` (stdio only) is the optional release-bootstrap source — present
    when the binary is fetched on-demand from a GitHub release (see
    ``mcp_bootstrap.ensure_server``). ``None`` means no bootstrap; the
    ``command`` must already resolve on PATH (or the server fails loud at load).
    """

    name: str = ""
    kind: str = "mcp"
    transport: str = ""
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    tools: list[str] | None = None
    credentials: list[str] = field(default_factory=list)
    github: dict[str, Any] | None = None


def _expand_env(value: str, env: dict[str, str]) -> str | None:
    """Replace ``${VAR}`` placeholders. Returns ``None`` if any var is unset."""

    def _repl(m: re.Match) -> str:
        var = m.group(1)
        v = env.get(var, "")
        if not v:
            raise _MissingPlaceholder(var)
        return v

    try:
        return _PLACEHOLDER_RE.sub(_repl, value)
    except _MissingPlaceholder:
        return None


class _MissingPlaceholder(Exception):
    def __init__(self, var: str) -> None:
        self.var = var
        super().__init__(f"unresolved placeholder ${{{var}}}")


def _substitute_spec(
    spec: ToolServerSpec,
    env: dict[str, str] | None = None,
    *,
    permissive: bool = False,
) -> ToolServerSpec:
    """Return a copy of ``spec`` with ``${VAR}`` placeholders resolved in
    ``url``, ``headers`` values, and ``env`` values.

    Raises ``ValueError`` on any unresolved placeholder or missing credential.
    When ``permissive=True`` (the offline-contract path), unresolved
    placeholders are LEFT AS-IS instead of raising — the field is structurally
    present (non-empty); its VALUE is a runtime/env concern the contract cannot
    check without the operator's secrets. This is what lets a catalog ship a
    git-safe ``url: ${PUX_MCP_WEB_RESEARCH_URL}`` that passes
    ``--check-contract`` offline while still failing loud at load time if the
    operator forgot to set the var. Credential checks still raise in both modes
    (a declared credential absent from env is reported as its own contract
    error, not swallowed)."""
    e = os.environ if env is None else dict(env)
    missing_creds = [c for c in spec.credentials if not e.get(c, "")]
    if missing_creds:
        raise ValueError(
            f"tool server {spec.name!r}: missing required credential(s): "
            f"{missing_creds}"
        )

    def _resolve(value: str, field: str) -> str:
        if not value:
            return value
        resolved = _expand_env(value, e)
        if resolved is not None:
            return resolved
        if permissive:
            return value  # leave ${VAR} as-is (offline structural check)
        raise ValueError(
            f"tool server {spec.name!r}: unresolved ${_missing_var(value, e)} "
            f"in {field} {value!r}"
        )

    url = _resolve(spec.url, "url")
    headers = {k: _resolve(v, f"header {k}") for k, v in spec.headers.items()}
    env_resolved = {k: _resolve(v, f"env {k}") for k, v in spec.env.items()}
    return ToolServerSpec(
        name=spec.name,
        kind=spec.kind,
        transport=spec.transport,
        url=url,
        headers=headers,
        command=spec.command,
        args=list(spec.args),
        env=env_resolved,
        tools=list(spec.tools) if spec.tools is not None else None,
        credentials=list(spec.credentials),
        github=dict(spec.github) if spec.github is not None else None,
    )


def _missing_var(value: str, env: dict[str, str]) -> str:
    for m in _PLACEHOLDER_RE.finditer(value):
        var = m.group(1)
        if not env.get(var, ""):
            return var
    return "?"


# --- catalog ----------------------------------------------------------------


def _catalog_path() -> Path:
    return _orgs_dir() / "_shared" / "tool_servers.yaml"


_catalog_cache: dict[str, ToolServerSpec] | None = None


def _parse_spec(name: str, d: dict[str, Any]) -> ToolServerSpec:
    """Parse one entry dict into a ``ToolServerSpec``. Raises ``ValueError`` on
    invalid shape."""
    kind = str(d.get("kind", "mcp") or "mcp")
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"tool server {name!r}: unknown kind {kind!r}; "
            f"valid: {sorted(_VALID_KINDS)}"
        )
    transport = str(d.get("transport", "") or "")
    if not transport:
        raise ValueError(f"tool server {name!r}: missing 'transport'")
    if transport not in _KNOWN_TRANSPORTS:
        raise ValueError(
            f"tool server {name!r}: unknown transport {transport!r}; "
            f"valid: {sorted(_KNOWN_TRANSPORTS)}"
        )
    required = _TRANSPORT_REQUIRED.get(transport, ())
    for field_name in required:
        if not d.get(field_name):
            raise ValueError(
                f"tool server {name!r}: transport {transport!r} requires "
                f"'{field_name}'"
            )
    tools_raw = d.get("tools")
    tools: list[str] | None = None
    if tools_raw is not None:
        if not isinstance(tools_raw, list):
            raise ValueError(f"tool server {name!r}: 'tools' must be a list")
        tools = [str(t) for t in tools_raw]
        if not tools:
            raise ValueError(f"tool server {name!r}: 'tools' list is empty")
    creds = d.get("credentials")
    if creds is not None and not isinstance(creds, list):
        raise ValueError(f"tool server {name!r}: 'credentials' must be a list")
    github = _parse_github_block(name, d.get("github"), transport)
    return ToolServerSpec(
        name=name,
        kind=kind,
        transport=transport,
        url=str(d.get("url", "") or ""),
        headers={str(k): str(v) for k, v in (d.get("headers") or {}).items()},
        command=str(d.get("command", "") or ""),
        args=[str(a) for a in (d.get("args") or [])],
        env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
        tools=tools,
        credentials=[str(c) for c in (creds or [])],
        github=github,
    )


# Required keys in a ``github:`` release-bootstrap block + their token contract.
# The ``asset`` glob MUST carry ``{os}`` and ``{arch}`` tokens so the per-platform
# asset can be selected deterministically (a token-less glob can't pick the right
# binary across platforms — fail it at contract, not at download time).
_GITHUB_REQUIRED: tuple[str, ...] = ("repo", "asset", "binary", "version")
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _parse_github_block(
    name: str, raw: Any, transport: str,
) -> dict[str, str] | None:
    """Parse the optional ``github:`` release-bootstrap block. ``None`` when
    absent. Raises ``ValueError`` on a malformed block. Only meaningful for
    stdio servers (the only transport with a local binary to fetch)."""
    if raw is None:
        return None
    if transport != "stdio":
        raise ValueError(
            f"tool server {name!r}: 'github' bootstrap is only meaningful for "
            f"stdio servers (transport={transport!r})"
        )
    if not isinstance(raw, dict):
        raise ValueError(f"tool server {name!r}: 'github' must be a mapping")
    block: dict[str, str] = {}
    for key in _GITHUB_REQUIRED:
        val = raw.get(key)
        if val is None or str(val) == "":
            raise ValueError(
                f"tool server {name!r}: 'github.{key}' is required and non-empty"
            )
        block[key] = str(val)
    if not _GITHUB_REPO_RE.match(block["repo"]):
        raise ValueError(
            f"tool server {name!r}: 'github.repo' must be 'owner/name', "
            f"got {block['repo']!r}"
        )
    for tok in ("{os}", "{arch}"):
        if tok not in block["asset"]:
            raise ValueError(
                f"tool server {name!r}: 'github.asset' must contain {tok} "
                f"(per-platform select); got {block['asset']!r}"
            )
    return block


def load_catalog() -> dict[str, ToolServerSpec]:
    """Read the shared catalog (``orgs/_shared/tool_servers.yaml``). Returns
    ``{}`` when absent. Result is cached after first read."""
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache
    path = _catalog_path()
    if not path.is_file():
        _catalog_cache = {}
        return _catalog_cache
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"tool_servers.yaml: not valid YAML: {e}") from e
    if data is None:
        _catalog_cache = {}
        return _catalog_cache
    if not isinstance(data, dict):
        raise ValueError(
            f"tool_servers.yaml: top-level must be a mapping, "
            f"got {type(data).__name__}"
        )
    catalog: dict[str, ToolServerSpec] = {}
    for name, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"tool_servers.yaml: entry {name!r} must be a mapping"
            )
        catalog[name] = _parse_spec(name, entry)
    _catalog_cache = catalog
    return catalog


# --- resolution --------------------------------------------------------------


def _org_yaml_mcp_items(org: str) -> list:
    """CU-3 sugar: the ``mcp`` entries from this org's OWN ``org.yaml``
    ``capabilities:`` block, in the same shape ``resolve_tool_servers`` consumes
    (a bare catalog-ref string, or a ``{ref, tools}`` mapping). Reads the org's
    OWN ``org.yaml`` only — NO inheritance, because mcp is security-scoped (the
    same reason ``policy.yaml`` is never inherited). Returns ``[]`` when the org
    ships no ``org.yaml`` or no ``capabilities:`` block.

    A malformed block (wrong kind for the home, bad shape) raises
    ``CapabilitiesSugarError`` — surfaced by ``validate_tool_servers`` (which
    calls ``resolve_tool_servers`` with ``permissive=True`` and catches
    ``ValueError``) as a ``tool-servers`` contract violation. Never silently
    skipped."""
    # ``_org_path`` raises ``FileNotFoundError`` for an org that isn't on disk
    # (a scratch test org, an exported-runner stub) — that means there's no
    # ``org.yaml`` mcp sugar to merge, so return ``[]`` (not propagate): mcp is
    # optional, and the only declaration site for such an org is policy.yaml.
    try:
        path = _org_path(org) / "org.yaml"
    except FileNotFoundError:
        return []
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return []
    if not isinstance(data, dict):
        return []
    return org_mcp_items_from_dict(data, org)


def resolve_tool_servers(
    org: str,
    env: dict[str, str] | None = None,
    *,
    permissive: bool = False,
) -> list[ToolServerSpec]:
    """Resolve the org's ``org.yaml`` ``capabilities:`` (``kind: mcp``) entries
    into fully-resolved ``ToolServerSpec`` objects.

    Each item is one of:
      - a bare string → catalog ref (copy)
      - ``{ref: name, tools?: [...]}`` → catalog ref + allowlist override
      - ``{name, kind, transport, ...}`` → inline spec

    Raises ``ValueError`` on unknown catalog ref, unknown kind, missing
    transport-required fields, duplicate resolved name, or (unless
    ``permissive=True``) unresolved ``${VAR}`` placeholders. Returns ``[]``
    when the org declares no mcp capabilities or the list is empty.

    ``permissive=True`` is the offline-contract path: ``${VAR}`` placeholders
    are left unresolved instead of raising, so structural validation (the field
    is declared, the transport is known, the catalog ref resolves) can run
    without the operator's env/secret values. The runtime path
    (``permissive=False``, the default) fails loud on any unresolved var."""
    # CU-4 (strict one-way): the ONE declaration site for foreign MCP servers is
    # org.yaml ``capabilities:`` (kind: mcp). The pre-unification policy.yaml
    # ``tool_servers:`` read path was REMOVED — that block is now a permanent
    # contract failure (``no-legacy-tool-servers``) and can no longer influence
    # resolution. ``_org_yaml_mcp_items`` is the sole source.
    items = _org_yaml_mcp_items(org)
    if not items:
        return []

    catalog = load_catalog()
    seen: set[str] = set()
    resolved: list[ToolServerSpec] = []

    for item in items:
        spec: ToolServerSpec | None = None
        name = ""

        if isinstance(item, str):
            name = item
            if name not in catalog:
                raise ValueError(
                    f"{org}: tool_servers: unknown catalog ref {name!r}; "
                    f"catalog has: {sorted(catalog)}"
                )
            spec = ToolServerSpec(**{**catalog[name].__dict__})
        elif isinstance(item, dict):
            item_d = dict(item)
            if "name" in item_d and "kind" in item_d:
                name = str(item_d.pop("name"))
                spec = _parse_spec(name, item_d)
            elif "ref" in item_d:
                name = str(item_d["ref"])
                if name not in catalog:
                    raise ValueError(
                        f"{org}: tool_servers: unknown catalog ref {name!r}; "
                        f"catalog has: {sorted(catalog)}"
                    )
                ref = catalog[name]
                spec = ToolServerSpec(**{**ref.__dict__})
                tools_override = item_d.get("tools")
                if tools_override is not None:
                    if not isinstance(tools_override, list):
                        raise ValueError(
                            f"{org}: tool_servers: ref {name!r} tools "
                            f"override must be a list"
                        )
                    spec.tools = [str(t) for t in tools_override]
            else:
                raise ValueError(
                    f"{org}: tool_servers: mapping entry must have "
                    f"'name'+'kind' (inline) or 'ref' (catalog override), "
                    f"got keys: {sorted(item_d)}"
                )
        else:
            raise ValueError(
                f"{org}: tool_servers: entry must be a string (catalog ref) "
                f"or a mapping, got {type(item).__name__}"
            )

        if not name:
            raise ValueError(f"{org}: tool_servers: entry has no resolved name")
        if name in seen:
            raise ValueError(
                f"{org}: tool_servers: duplicate resolved name {name!r}"
            )
        seen.add(name)

        try:
            spec = _substitute_spec(spec, env, permissive=permissive)
        except ValueError as exc:
            if permissive:
                raise  # offline contract — surface every error
            # RUNTIME per-server isolation: one spec with an unset ${VAR} or
            # missing credential is SKIPPED (logged), not allowed to kill the
            # whole org. The yaml catalog promises "one unset var can't brick
            # the org" — this catch makes that promise TRUE at resolution time
            # (McpSessionManager.open already isolates per-server AFTER
            # resolution; this extends it to the resolution step itself).
            _log.error(
                "org %s: tool server %r skipped at resolution — %s",
                org, name, exc,
            )
            continue
        resolved.append(spec)

    return resolved


# --- contract validation -----------------------------------------------------


def validate_tool_servers(org: str) -> list[str]:
    """Offline contract surface — validates the org's ``tool_servers:``
    declaration WITHOUT live credential resolution. Returns a list of error
    strings (empty = valid). Called from ``contract.check_org`` so a broken
    declaration fails ``--check-contract``.

    A ``github:`` release-bootstrap block is structurally validated at parse
    time (``_parse_github_block`` — required keys, ``owner/repo`` shape,
    ``{os}``/``{arch}`` tokens). A malformed block raises during
    ``load_catalog`` and surfaces here via the same error path as any other
    parse failure (the org need not even reference the entry — ``load_catalog``
    parses the whole catalog for any org with a tool_servers list)."""
    errors: list[str] = []
    try:
        # permissive=True: validate STRUCTURE without the operator's env values.
        # A catalog entry may ship a git-safe ``url: ${VAR}`` whose value is
        # injected at runtime; the contract must pass that offline (the field is
        # declared) while the runtime path still fails loud if the var is unset.
        specs = resolve_tool_servers(org, env={}, permissive=True)
    except ValueError as e:
        return [str(e)]
    for spec in specs:
        if spec.kind not in _VALID_KINDS:
            errors.append(
                f"{org}: tool server {spec.name!r}: invalid kind "
                f"{spec.kind!r}"
            )
        if spec.transport not in _KNOWN_TRANSPORTS:
            errors.append(
                f"{org}: tool server {spec.name!r}: invalid transport "
                f"{spec.transport!r}"
            )
        required = _TRANSPORT_REQUIRED.get(spec.transport, ())
        for field_name in required:
            val = getattr(spec, field_name, "")
            if not val:
                errors.append(
                    f"{org}: tool server {spec.name!r}: transport "
                    f"{spec.transport!r} requires {field_name!r}"
                )
        if spec.credentials:
            errors.append(
                f"{org}: tool server {spec.name!r}: declares credential(s) "
                f"{spec.credentials} (not checked offline; will fail at load "
                f"if absent from env)"
            )
    return errors
