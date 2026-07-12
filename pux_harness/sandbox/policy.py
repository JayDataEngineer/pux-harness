"""Declarative sandbox policy engine (port of
``backend/internal/policy``).

Pure logic — no Docker, no container state. Loads ``orgs/<name>/policy.yaml``,
validates credentials against the live env, resolves ``${VAR}`` mount
placeholders, and renders the iptables egress allowlist. The harness owns
policy *resolution*; container-side *enforcement* (binds/env/caps/egress.conf
staging) transfers once the harness owns container creation.

Faithful 1:1 port of the Go package (``policy.go`` / ``validate.go`` /
``egress.go``) so behavior + the parity test gate (``tests/test_policy.py``,
mirroring the 22 Go tests) match. The shipped-policy gate runs against the
same real ``orgs/*/policy.yaml`` files the Go side enforces.

Design notes that survive the port:

- ``NoPolicy`` is a *sentinel* (raised when the org has no policy.yaml) — callers
  MUST branch on it as the "not opted in" path, distinct from ``PolicyError``.
- ``validate_env`` / ``env_vars`` / ``resolve_mounts`` read the operator's live
  ``os.environ`` by default (matching Go's ``os.Getenv``); an explicit ``env``
  param is accepted for tests. Empty-string values are treated as absent, like
  Go's ``os.Getenv``.
- ``egress_rules`` resolves DNS *now* (sandbox-create time), not in-container at
  boot — by the time the firewall runs, DNS may be blocked. Container-resolved
  names (``host.docker.internal``) pass through verbatim: they don't resolve on
  the host but do inside the container via ``/etc/hosts``; the boot script
  resolves them via ``getent`` (no network).
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml

# ${VAR} placeholder, mirroring validate.go:13.
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Docker-internal /etc/hosts entries that do NOT resolve on the host (where
# EgressRules runs) but DO inside the container. Passed through verbatim;
# apply-egress-policy.sh resolves them at boot via getent (no DNS, works under
# deny-by-default). This is how bridge-networked orgs reach host-side services
# (e.g. a shared SurrealDB on the operator's machine) through the firewall.
_CONTAINER_RESOLVED = frozenset({"host.docker.internal"})


# --- exceptions ---------------------------------------------------------------


class NoPolicy(Exception):
    """Sentinel: the org has no ``policy.yaml``. Callers MUST treat this as
    "feature not opted in" (today's behavior), NOT an error."""


class PolicyError(Exception):
    """A real failure: malformed YAML, bad path, unresolvable mount, bad port…"""


class MissingCreds(PolicyError):
    """One or more required credentials absent from the operator env. ``missing``
    lists every absent name so the operator sees the full repair list in one
    shot, not one round-trip per credential."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = list(missing)
        super().__init__("missing required credentials: " + ", ".join(self.missing))


class UnresolvedMount(PolicyError):
    """A ``${VAR}`` placeholder in a mount's ``Host`` has no matching env var.
    Failing loud beats silently mounting the wrong directory."""

    def __init__(self, container: str, unresolved: str, missing_var: str) -> None:
        self.container = container
        self.unresolved = unresolved
        self.missing_var = missing_var
        super().__init__(
            f'mount {container}: host "{unresolved}" references unset env var {missing_var}'
        )


# --- schema (mirrors policy.go structs) ---------------------------------------


@dataclass
class Mount:
    host: str = ""
    container: str = ""
    mode: str = ""  # "rw" (default) or "ro"


@dataclass
class Workspace:
    mounts: list[Mount] = field(default_factory=list)
    run_as_host_user: bool = False


@dataclass
class Rule:
    host: str = ""
    port: int = 0
    ports: list[int] = field(default_factory=list)
    protocol: str = ""  # default tcp; only tcp supported today


@dataclass
class Egress:
    allow: list[Rule] = field(default_factory=list)


@dataclass
class Credentials:
    required: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)


@dataclass
class BuildSpec:
    """``sandbox.build`` — build the org's custom sandbox image before create
    if absent. Paths are project-relative; resolved + existence-checked by the
    runner (and the contract's offline validator). Host-side Docker build,
    no compose. Both fields blank == no build requested (``build_spec`` → None).
    """

    dockerfile: str = ""
    context: str = ""


@dataclass
class DepsSpec:
    """``sandbox.deps`` — extra apt + pip packages installed into the org's
    sandbox container at ``create()`` time (after ``start()``, before
    persisted-state restore). Installed in-container via ``apt-get`` / ``pip``;
    warn-and-continue on failure (a blocked mirror or bad package never breaks
    the run — mirror run_jobs / host_setup best-effort semantics). Egress for
    ``pypi.org`` / ``files.pythonhosted.org`` (pip) and the Debian mirrors (apt)
    MUST be on the org's egress allowlist — install is NOT auto-allowed
    (explicit-egress principle). The image's ``/root/.cache`` pip-cache mount
    makes repeat installs fast.
    """

    apt: list[str] = field(default_factory=list)
    pip: list[str] = field(default_factory=list)


@dataclass
class DisplaySpec:
    """``sandbox.display`` — publish the watchable desktop so a human can open
    the sandbox's live screen in a browser. The image already runs the full
    noVNC/KasmVNC stack under supervisord (Xvfb :99 → x11vnc → websockify);
    this just maps its VNC-web port to a host port so it's reachable.

    ``watch: true`` publishes the port bound to ``127.0.0.1`` ONLY with an
    ephemeral host port — x11vnc runs passwordless (``-nopw``), so it is NEVER
    exposed to the LAN. Default off: an explicit opt-in carries the security
    property. ``backend`` selects the image variant's VNC-web port:
    ``standard`` (noVNC, 6080) or ``kasm`` (KasmVNC H.264/WebRTC, 8444)."""

    watch: bool = False
    backend: str = "standard"


@dataclass
class SandboxSpec:
    image: str = ""
    tier: str = ""  # "isolated" or "bridged"
    build: BuildSpec = field(default_factory=BuildSpec)
    deps: DepsSpec = field(default_factory=DepsSpec)
    display: DisplaySpec = field(default_factory=DisplaySpec)
    # ``sandbox.dynamic_tools`` — opt the org INTO level (c): agent-authored
    # Python functions under ``orgs/<org>/lib/functions/`` that persist between
    # runs and are callable in-container (the tool rung that *compounds* — see
    # ``docs/dynamic-tools-and-packaging.md`` Part 1). Default off: an explicit
    # opt-in, mirroring ``display.watch`` / ``deps``.
    dynamic_tools: bool = False
    # ``sandbox.memory_mb`` / ``sandbox.cpu_cores`` — per-org resource budget
    # override. ``0`` / ``0.0`` means "unset" (defer to the env-var/default
    # chain in container.py). The browser orgs need this: Chrome + SeleniumBase
    # cold-start blows through the 2 GiB default and gets OOM-reaped (SIGKILL,
    # exit 137) before sb_server ever binds. A browser org declares
    # ``memory_mb: 4096``; a coding org leaves it unset and inherits the lean
    # default. See ``resolve_resources`` for the precedence chain.
    memory_mb: int = 0
    cpu_cores: float = 0.0


@dataclass
class BrowserSpec:
    cookies_env: str = ""
    proxy: str = ""


@dataclass
class ToolSurfaceSpec:
    """``tool_surface`` — scope the SUPERVISOR's specialist tool surface by
    capability group. ``groups`` is an ALLOWLIST of group names (``code`` /
    ``skills`` / ``media`` / ``browser`` / ``desktop``) and/or bare specialist
    slugs (``python``); anything not listed is dropped from the supervisor's
    own tool list (NOT from subagents — they resolve their own ``tools:``
    allowlist against the FULL surface). Absent/empty -> the supervisor carries
    EVERY specialist (byte-identical to today). See ``registry.resolve_tool_allowlist``
    for the resolution + validation semantics."""

    groups: list[str] = field(default_factory=list)


@dataclass
class HostSetupHook:
    """One host-side prep hook. Run BEFORE ``create()`` (so before
    ``validate_env``): each captures ``helper_script``'s stdout into the env
    vars named in ``exports`` (value ``stdout`` → captured stdout), which then
    flow through the existing ``credentials.required`` / ``browser.cookies_env``
    / ``env_vars`` / ``seed-cookies.sh`` chain UNCHANGED — one mechanism, not
    two. ``python_deps`` install into a cached per-hook uv venv at
    ``<project>/.pux/venvs/<name>/`` (runner-owned)."""

    name: str = ""
    helper_script: str = ""
    python_deps: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    exports: dict[str, str] = field(default_factory=dict)


@dataclass
class JobSpec:
    """One in-sandbox prep step declared in ``policy.yaml`` ``jobs:``. Runs
    AFTER ``create()`` (inside the container), producing artifacts (not env
    exports). Scripts reach external services over the existing egress
    allowlist — no new network surface. Idempotency delegated to the scripts
    (file caches + SurrealDB UPSERTs make repeat runs cheap)."""

    name: str = ""
    script: str = ""
    args: list[str] = field(default_factory=list)
    timeout: int = 0  # seconds; 0 = no limit
    description: str = ""
    when: str = ""  # shell predicate; exit 0 -> run, non-zero -> skip (status "skipped")


@dataclass
class Policy:
    workspace: Workspace = field(default_factory=Workspace)
    egress: Egress = field(default_factory=Egress)
    credentials: Credentials = field(default_factory=Credentials)
    sandbox: SandboxSpec = field(default_factory=SandboxSpec)
    browser: BrowserSpec = field(default_factory=BrowserSpec)
    tool_surface: ToolSurfaceSpec = field(default_factory=ToolSurfaceSpec)
    host_setup: list[HostSetupHook] = field(default_factory=list)
    jobs: list[JobSpec] = field(default_factory=list)
    # NOTE: foreign MCP servers are declared in org.yaml ``capabilities:``
    # (kind: mcp) — the ONE canonical site (CU-4). policy.yaml ``tool_servers:``
    # is a permanent contract failure (``no-legacy-tool-servers``); there is no
    # ``Policy.tool_servers`` field and no parse path for it.
    protocols: list = field(default_factory=list)


# Agent-facing protocol surfaces an org may declare in policy.yaml.
# ``acp`` = stdio Agent Client Protocol (``pux acp --org <name>`` — always
# available, per-spawn, nothing to mount); ``agui`` = CopilotKit AG-UI SSE.
# (The retired ``server.py`` used to mount ``/agui/{name}`` off this value;
# that mount went with server.py in Aegra phase D — the field stays as a
# validated self-declaration.) Absent/empty ``protocols:`` resolves to
# DEFAULT_PROTOCOLS (both), so the declaration is opt-in NARROWING + explicit
# self-description — never a regression: an org without it behaves as before
# (all surfaces). KNOWN_PROTOCOLS is the single source contract.py validates
# against.
DEFAULT_PROTOCOLS: list[str] = ["acp", "agui"]
KNOWN_PROTOCOLS: frozenset[str] = frozenset(DEFAULT_PROTOCOLS)


@dataclass
class ResolvedMount:
    host: str
    container: str
    mode: str  # "rw" or "ro"


# --- YAML -> dataclasses ------------------------------------------------------


def _mount(d: Mapping) -> Mount:
    return Mount(
        host=str(d.get("host", "") or ""),
        container=str(d.get("container", "") or ""),
        mode=str(d.get("mode", "") or ""),
    )


def _rule(d: Mapping) -> Rule:
    ports = d.get("ports") or []
    return Rule(
        host=str(d.get("host", "") or ""),
        port=int(d.get("port", 0) or 0),
        ports=[int(p) for p in ports],
        protocol=str(d.get("protocol", "") or ""),
    )


def _policy_from_dict(d: Mapping) -> Policy:
    """Lenient mapping, like Go's yaml.v3 into structs: unknown keys ignored,
    missing fields default. Raises PolicyError only on a non-mapping section."""
    pol = Policy()

    def _section(name: str) -> Mapping:
        sec = d.get(name)
        if sec is None:
            return {}
        if not isinstance(sec, Mapping):
            raise PolicyError(f"policy: section {name!r} must be a mapping")
        return sec

    ws = _section("workspace")
    pol.workspace = Workspace(
        mounts=[_mount(m) for m in (ws.get("mounts") or []) if isinstance(m, Mapping)],
        run_as_host_user=bool(ws.get("run_as_host_user", False)),
    )
    eg = _section("egress")
    pol.egress = Egress(allow=[_rule(r) for r in (eg.get("allow") or []) if isinstance(r, Mapping)])
    cr = _section("credentials")
    pol.credentials = Credentials(
        required=[str(x) for x in (cr.get("required") or [])],
        optional=[str(x) for x in (cr.get("optional") or [])],
    )
    sb = _section("sandbox")
    build_map = sb.get("build")
    if build_map is not None:
        if not isinstance(build_map, Mapping):
            raise PolicyError("policy: section 'sandbox.build' must be a mapping")
        build = BuildSpec(
            dockerfile=str(build_map.get("dockerfile", "") or ""),
            context=str(build_map.get("context", "") or ""),
        )
    else:
        build = BuildSpec()
    deps_map = sb.get("deps")
    if deps_map is not None:
        if not isinstance(deps_map, Mapping):
            raise PolicyError("policy: section 'sandbox.deps' must be a mapping")
        deps = DepsSpec(
            apt=[str(x) for x in (deps_map.get("apt") or [])],
            pip=[str(x) for x in (deps_map.get("pip") or [])],
        )
    else:
        deps = DepsSpec()
    display_map = sb.get("display")
    if display_map is not None:
        if not isinstance(display_map, Mapping):
            raise PolicyError("policy: section 'sandbox.display' must be a mapping")
        backend = str(display_map.get("backend", "standard") or "standard")
        if backend not in ("standard", "kasm"):
            raise PolicyError(
                f"policy: sandbox.display.backend {backend!r} must be 'standard' or 'kasm'"
            )
        display = DisplaySpec(
            watch=bool(display_map.get("watch", False)),
            backend=backend,
        )
    else:
        display = DisplaySpec()
    dynamic_tools_raw = sb.get("dynamic_tools")
    if dynamic_tools_raw is None:
        dynamic_tools = False
    elif not isinstance(dynamic_tools_raw, bool):
        raise PolicyError(
            f"policy: sandbox.dynamic_tools must be a bool, got {type(dynamic_tools_raw).__name__}"
        )
    else:
        dynamic_tools = dynamic_tools_raw
    pol.sandbox = SandboxSpec(
        image=str(sb.get("image", "") or ""),
        tier=str(sb.get("tier", "") or ""),
        build=build,
        deps=deps,
        display=display,
        dynamic_tools=dynamic_tools,
        memory_mb=int(sb.get("memory_mb", 0) or 0),
        cpu_cores=float(sb.get("cpu_cores", 0.0) or 0.0),
    )
    br = _section("browser")
    pol.browser = BrowserSpec(
        cookies_env=str(br.get("cookies_env", "") or ""),
        proxy=str(br.get("proxy", "") or ""),
    )
    # tool_surface: scope the supervisor's specialist tools by capability group.
    # Absent/empty -> ToolSurfaceSpec() (all groups) -> byte-identical to today.
    # Entries are validated by registry.resolve_tool_allowlist (raises on an
    # unknown group/slug) so a typo fails loud at load time, not silently.
    ts = _section("tool_surface")
    groups_raw = ts.get("groups") or []
    if not isinstance(groups_raw, list):
        raise PolicyError("policy: section 'tool_surface.groups' must be a list")
    pol.tool_surface = ToolSurfaceSpec(groups=[str(x) for x in groups_raw])
    # host_setup: a list of host-side prep hooks (run before create(), produce
    # env exports that flow through credentials/cookies unchanged). Absent or
    # empty -> no hooks (today's behavior).
    hooks_raw = d.get("host_setup") or []
    if not isinstance(hooks_raw, list):
        raise PolicyError("policy: section 'host_setup' must be a list")
    hooks: list[HostSetupHook] = []
    for h in hooks_raw:
        if not isinstance(h, Mapping):
            raise PolicyError("policy: each host_setup entry must be a mapping")
        exports_raw = h.get("exports") or {}
        if not isinstance(exports_raw, Mapping):
            raise PolicyError(
                f"policy: host_setup entry {h.get('name')!r} exports must be a mapping"
            )
        hooks.append(
            HostSetupHook(
                name=str(h.get("name", "") or ""),
                helper_script=str(h.get("helper_script", "") or ""),
                python_deps=[str(x) for x in (h.get("python_deps") or [])],
                args=[str(x) for x in (h.get("args") or [])],
                exports={str(k): str(v) for k, v in exports_raw.items()},
            )
        )
    pol.host_setup = hooks
    # jobs: a list of in-sandbox prep steps (run after create(), produce
    # artifacts, not env exports). Absent or empty -> no jobs.
    jobs_raw = d.get("jobs") or []
    if not isinstance(jobs_raw, list):
        raise PolicyError("policy: section 'jobs' must be a list")
    jobs: list[JobSpec] = []
    for j in jobs_raw:
        if not isinstance(j, Mapping):
            raise PolicyError("policy: each jobs entry must be a mapping")
        jobs.append(
            JobSpec(
                name=str(j.get("name", "") or ""),
                script=str(j.get("script", "") or ""),
                args=[str(x) for x in (j.get("args") or [])],
                timeout=int(j.get("timeout", 0) or 0),
                description=str(j.get("description", "") or ""),
                when=str(j.get("when", "") or ""),
            )
        )
    pol.jobs = jobs
    # protocols: which agent-facing surfaces this org declares (see
    # DEFAULT_PROTOCOLS). Absent or empty -> resolve_protocols() yields the
    # DEFAULT, preserving today's "all surfaces for all orgs".
    proto_raw = d.get("protocols") or []
    if not isinstance(proto_raw, list):
        raise PolicyError("policy: section 'protocols' must be a list")
    pol.protocols = [str(x) for x in proto_raw]
    return pol


# --- load ---------------------------------------------------------------------


def load(org_name: str, project_root: str | Path) -> Policy:
    """Read ``orgs/<org_name>/policy.yaml`` under ``project_root``. Raises
    ``NoPolicy`` (sentinel) if absent; ``PolicyError`` on a malformed file.

    Checks both ``orgs/<name>/policy.yaml`` and
    ``orgs/specialists/<name>/policy.yaml`` (orgs that were moved to the
    ``specialists/`` subfolder)."""
    if not org_name:
        raise NoPolicy()
    if not project_root:
        raise PolicyError("policy.load: project_root required")
    root = Path(project_root)
    path = root / "orgs" / org_name / "policy.yaml"
    if not path.is_file():
        path = root / "orgs" / "specialists" / org_name / "policy.yaml"
    try:
        data = path.read_text()
    except FileNotFoundError:
        raise NoPolicy()
    except OSError as e:
        raise PolicyError(f"policy.load {path}: {e}") from e
    try:
        parsed = yaml.safe_load(data)
    except yaml.YAMLError as e:
        raise PolicyError(f"policy.load {path}: {e}") from e
    if parsed is None:
        return Policy()
    if not isinstance(parsed, Mapping):
        raise PolicyError(f"policy.load {path}: top-level must be a mapping")
    pol = _policy_from_dict(parsed)
    # Default Protocol to "tcp" on every rule with an empty value (policy.go:141).
    for rule in pol.egress.allow:
        if not rule.protocol:
            rule.protocol = "tcp"
    return pol


# --- protocols ----------------------------------------------------------------


def resolve_protocols(pol: Policy) -> list[str]:
    """The surfaces an org actually exposes. Declared non-empty -> as-is (a
    NARROWED set, e.g. ``[acp]`` for a coding org that should not mount AG-UI);
    absent/empty -> :data:`DEFAULT_PROTOCOLS` (both surfaces)."""
    return list(pol.protocols) if pol.protocols else list(DEFAULT_PROTOCOLS)


# --- credentials --------------------------------------------------------------


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def validate_env(p: Policy | None, env: Mapping[str, str] | None = None) -> None:
    """Raise ``MissingCreds`` if any required credential is absent from env.
    Optional creds are not checked (absence is silent). No-op if ``p`` is None.
    Empty-string values count as absent, like Go's ``os.Getenv``."""
    if p is None:
        return
    e = _env(env)
    missing = [n for n in p.credentials.required if not e.get(n, "")]
    if missing:
        raise MissingCreds(missing)


def env_vars(p: Policy | None, env: Mapping[str, str] | None = None) -> list[str]:
    """``KEY=VALUE`` strings to inject into the container env. Required creds
    always (ValidateEnv proved them set); optional creds only when present;
    browser cookies value + ``SEED_COOKIES_ENV=<name>`` pointer when the
    cookies var is set."""
    if p is None:
        return []
    e = _env(env)
    out: list[str] = []
    for name in p.credentials.required:
        out.append(f"{name}={e.get(name, '')}")
    for name in p.credentials.optional:
        v = e.get(name, "")
        if v:
            out.append(f"{name}={v}")
    if p.browser.cookies_env:
        v = e.get(p.browser.cookies_env, "")
        if v:
            # Avoid duplicate: if cookies_env is already in credentials.required,
            # it was injected above — only add the SEED_COOKIES_ENV pointer here.
            if p.browser.cookies_env not in p.credentials.required:
                out.append(f"{p.browser.cookies_env}={v}")
            out.append(f"SEED_COOKIES_ENV={p.browser.cookies_env}")
    if p.browser.proxy:
        out.append(f"SB_SERVER_PROXY={p.browser.proxy}")
    return out


# --- mounts -------------------------------------------------------------------


def _expand_placeholders(
    value: str, container_path: str, env: Mapping[str, str]
) -> tuple[str, UnresolvedMount | None]:
    """Replace ``${VAR}`` with the env value. Returns the expanded string plus
    the first unresolved placeholder (or None). Mirrors validate.go:147 —
    replaces all set vars, tracks the first unset, raises on it after."""
    first: UnresolvedMount | None = None

    def _repl(m: re.Match) -> str:
        nonlocal first
        var = m.group(1)
        v = env.get(var, "")
        if v:
            return v
        if first is None:
            first = UnresolvedMount(container_path, m.group(0), var)
        return m.group(0)

    expanded = _PLACEHOLDER_RE.sub(_repl, value)
    return expanded, first


def resolve_mounts(p: Policy | None, env: Mapping[str, str] | None = None) -> list[ResolvedMount]:
    """Walk ``p.Workspace.Mounts``: expand ``${VAR}``, require absolute
    container paths, normalize mode. Raises ``UnresolvedMount`` on the first
    unset var (fail-fast) and ``PolicyError`` on a bad path/mode."""
    if p is None or not p.workspace.mounts:
        return []
    e = _env(env)
    out: list[ResolvedMount] = []
    for m in p.workspace.mounts:
        host, unresolved = _expand_placeholders(m.host, m.container, e)
        if unresolved is not None:
            raise unresolved
        if not os.path.isabs(m.container):
            raise PolicyError(
                f"mount {m.container}: container path must be absolute, got {m.container!r}"
            )
        mode = m.mode or "rw"
        if mode not in ("rw", "ro"):
            raise PolicyError(f"mount {m.container}: mode must be 'rw' or 'ro', got {mode!r}")
        out.append(ResolvedMount(host=host, container=m.container, mode=mode))
    return out


def host_user() -> str:
    """``UID:GID`` for the host user, suitable for Docker's ``User`` field.
    Only meaningful when ``workspace.run_as_host_user`` is true."""
    return f"{os.getuid()}:{os.getgid()}"


# --- egress -------------------------------------------------------------------


def _is_container_resolved(host: str) -> bool:
    return host.lower() in _CONTAINER_RESOLVED


def _is_literal_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _resolve_host(host: str) -> list[str]:
    """One or more IPs for a hostname, or validates a literal IP. Literal
    IPv4/IPv6 short-circuits (no DNS). DNS resolves via getaddrinfo; all IPs
    are included (multi-A-record fan-out), deduped."""
    if not host:
        raise PolicyError("empty host")
    if _is_literal_ip(host):
        return [host]
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise PolicyError(str(e)) from e
    seen: dict[str, None] = {}
    out: list[str] = []
    for _fam, _stype, _proto, _canon, sockaddr in infos:
        ip = sockaddr[0]
        if ip not in seen:
            seen[ip] = None
            out.append(ip)
    if not out:
        raise PolicyError(f"no IPs for {host}")
    return out


def egress_rules(p: Policy | None) -> str:
    """Render the iptables allow lines — one ``<ip> <port>`` per line, hostname
    pre-resolved to IP(s). DNS-resolved hosts get a ``# host: <name>`` comment
    first (for the periodic DNS refresh script); literal IPs + container-
    resolved names get none. Empty/None policy → ``""`` (no conf staged).

    Raises ``PolicyError`` on DNS failure, a rule with no port, or an
    out-of-range port."""
    if p is None or not p.egress.allow:
        return ""
    lines: list[str] = []
    for rule in p.egress.allow:
        if _is_container_resolved(rule.host):
            ips = [rule.host]
        else:
            ips = _resolve_host(rule.host)
        ports = list(rule.ports)
        if rule.port:
            ports = [rule.port, *ports]
        if not ports:
            raise PolicyError(f"egress: rule for {rule.host} has no port(s)")
        if not _is_literal_ip(rule.host) and not _is_container_resolved(rule.host):
            lines.append(f"# host: {rule.host}")
        for ip in ips:
            for port in ports:
                if port < 1 or port > 65535:
                    raise PolicyError(f"egress: port {port} for {rule.host} out of range")
                lines.append(f"{ip} {port}")
    return "\n".join(lines) + "\n"


# --- tier ---------------------------------------------------------------------


def resolve_tier(p: Policy | None, fallback: str) -> str:
    """The policy's ``sandbox.tier`` override, or ``fallback`` when unset/None.
    Single source of truth for the effective tier — callers consult this rather
    than reading ``p.sandbox.tier`` directly so the empty-vs-unset distinction
    is handled consistently."""
    if p is None or not p.sandbox.tier:
        return fallback
    return p.sandbox.tier


# --- resources ----------------------------------------------------------------


def resolve_resources(
    p: Policy | None,
    *,
    fallback_memory_mb: int,
    fallback_cpu_cores: float,
) -> tuple[int, float]:
    """The effective ``(memory_mb, cpu_cores)`` budget for the sandbox container.

    Precedence: **policy > fallback**. The policy's ``sandbox.memory_mb`` /
    ``sandbox.cpu_cores`` win when set (> 0); otherwise the caller's fallback
    (which is itself the env-var-then-default chain in container.py) is used.

    The env-var escape hatch (``PUX_SANDBOX_MEMORY_MB`` /
    ``PUX_SANDBOX_CPU_CORES``) is applied by the CALLER to compute the
    fallback — so an operator-set env var still wins for orgs that DON'T
    declare a budget, but an org that DECLARES its need (e.g. browser-agent:
    ``memory_mb: 4096``) gets what it asked for. This mirrors how
    ``sandbox.image`` is resolved (policy > env > default).
    """
    mem = fallback_memory_mb
    cpu = fallback_cpu_cores
    if p is not None:
        if p.sandbox.memory_mb > 0:
            mem = p.sandbox.memory_mb
        if p.sandbox.cpu_cores > 0.0:
            cpu = p.sandbox.cpu_cores
    return mem, cpu


# --- host setup + image build -------------------------------------------------


def host_setup_hooks(p: Policy | None) -> list[HostSetupHook]:
    """The policy's host-side prep hooks (run before ``create()``, produce env
    exports). Empty/None policy -> no hooks. Single accessor so callers never
    read ``p.host_setup`` directly and the empty-vs-unset distinction lives in
    one place."""
    if p is None:
        return []
    return list(p.host_setup)


def build_spec(p: Policy | None) -> BuildSpec | None:
    """The policy's image-build spec, or ``None`` when no build is requested.
    ``None`` => ``_ensure_image`` takes the existing pull path. A build is
    "requested" only when ``dockerfile`` is non-empty; ``context`` defaults to
    the Dockerfile's directory at run time when left blank."""
    if p is None:
        return None
    b = p.sandbox.build
    if not b.dockerfile:
        return None
    return b


# --- jobs (post-create prep steps) ------------------------------------------


def job_specs(p: Policy | None) -> list[JobSpec]:
    """The policy's in-sandbox prep jobs (run after ``create()``, before the
    agent loop). Empty/None policy -> no jobs. Single accessor so callers never
    read ``p.jobs`` directly and the empty-vs-unset distinction lives in one
    place."""
    if p is None:
        return []
    return list(p.jobs)


# --- tool surface (supervisor specialist scoping) --------------------------


def resolve_tool_allowlist(p: Policy | None) -> frozenset[str] | None:
    """The specialist slugs the org's SUPERVISOR may carry, or ``None`` for
    "all specialists" (byte-identical to today). Driven by
    ``p.tool_surface.groups``; delegates the real resolution + validation to
    ``registry.resolve_tool_allowlist`` (a typo'd group/slug raises
    ``ValueError`` there, re-raised as ``PolicyError`` so the contract checker
    reports it cleanly). ``None`` policy or empty ``groups`` -> ``None`` (all)."""
    if p is None or not p.tool_surface.groups:
        return None
    from pux_harness.sandbox.tools.registry import resolve_tool_allowlist as _resolve

    try:
        return _resolve(p.tool_surface.groups)
    except ValueError as exc:
        raise PolicyError(f"policy: {exc}") from exc
