"""The harness owns the Docker sandbox lifecycle (create/stop;
host_setup + image build).

Create extended ``create()`` with two pre-Docker steps run BEFORE the live
container is created: ``host_setup`` hooks run on the host first
(``sandbox/host_setup.py`` — cached uv venv at ``<project>/.pux/venvs/<name>/``,
captured stdout → env exports, injected into ``os.environ`` BEFORE
``validate_env`` so a hook can satisfy a required cred), then ``_ensure_image``
builds the image via ``client.images.build`` when ``policy.sandbox.build`` is set
+ the image is absent (else the existing pull-if-absent). No operator
``bootstrap.sh`` / ``docker-compose.yml`` — the harness is the ONE container
path, so policy is enforced by construction.

``create()`` reproduces the live container byte-for-byte (verified against the
inspect of ``orchestrator-sandbox-mcp-default`` 2026-07-03: runc, shared-infra,
2GB/2cpu/512pids, the 5 binds, the 3 ``openshell.*`` labels):

  image        ``OPENSHELL_IMAGE`` env | ``pux-sandbox:latest`` (policy
               ``sandbox.image`` override wins); built via ``policy.sandbox.build``
               if set + absent, else pulled if absent.
  labels       ``openshell.policy`` / ``openshell.sandbox-id`` /
               ``openshell.project-path`` — the label the exec path filters on.
  binds        ``<project>:/sandbox/workspace``,
               ``<policies>:/etc/openshell/policies:ro``, ``/tmp:/sandbox/tmp``,
               ``sandbox-<id>-persist`` (named vol) → ``/sandbox/persist``,
               ``pux-cache-<sha16>`` (per-project, sha256 of the abs path) →
               ``/root/.cache`` (``PUX_CACHE_VOLUME=off`` disables).
  env          ``SANDBOX_POLICY``/``DOCKER_HOST``/``HOST_GATEWAY`` + policy creds.
  resources    mem 2048MB / cpu 2.0 cores / pids 2048 (``PUX_SANDBOX_MEMORY_MB``/
               ``PUX_SANDBOX_CPU_CORES``/``PUX_SANDBOX_PIDS`` env knobs).
  runtime      ``runsc`` when TierIsolated + installed; ``PUX_SANDBOX_RUNTIME``
               overrides (``none`` opts out). Bridged tier never overrides.
  network      ``shared-infra`` (``OPENSHELL_NETWORK``); host net for Bridged.
  extra_hosts  ``host.docker.internal:host-gateway``.
  security_opt ``no-new-privileges:true`` — blocks setuid/setcap escalation
               inside the container (zero behavioral impact on normal work).
  caps         ``NET_ADMIN`` only when policy stages an egress allowlist.

Policy enforcement — the part that was blocked while the Go binary owned the
container (the *resolver* was ported; the *enforcer* stayed Go because it
mutates ``container.HostConfig`` at create) — runs here now. It mirrors
``backend/internal/sandbox/policy_hook.go::applyOrgPolicy`` step for step:
load → validate required creds → resolve ``${VAR}`` mounts → inject creds/cookies
env → ``run_as_host_user`` → ``UID:GID`` → stage ``<project>/.pux/egress.conf``
+ grant ``NET_ADMIN`` (skipped for effective TierBridged — host net makes
iptables-in-container meaningless).

Egress is enforced deny-by-default: ``apply-egress-policy.sh`` (supervised at
boot) reads ``<project>/.pux/egress.conf`` and installs iptables DROP on OUTPUT
except for the listed allow rules. The conf is only staged when
``policy.yaml`` declares an ``egress.allow`` block — so an org WITHOUT the block
has UNRESTRICTED egress (the default; opt in to restrict). When the block IS
declared, ``NET_ADMIN`` is granted so the boot script can install the rules;
``ensure()`` then fail-closes: a reused container missing ``NET_ADMIN`` while
the policy has an allowlist is REJECTED (not silently reused with open egress).

The legacy ``NETWORK_ALLOW`` / ``FS_READONLY`` / ``FS_READWRITE`` env vars were
removed — they were vestigial from the Go entrypoint that never survived the
port. Nothing in the image or harness consumed them (verified by grep of the
live container + repo 2026-07-11). Filesystem read-only enforcement (Docker
``--read-only`` + tmpfs for writable paths) is a separate hardening pass that
needs image-level testing; the dead env vars are gone so operators aren't
misled into thinking ``/etc`` is protected when it isn't.

``ensure()`` is the single-tenant gate: discover a running container by the
project label and reuse it; create+start only when none is running (removing a
stopped stale namesake first). ``destroy()`` saves persisted state, stops
(10s grace), removes (force).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import docker
from docker.errors import APIError, BuildError, ImageNotFound, NotFound

from pux_harness.kit._paths import project_root
from pux_harness.sandbox import host_setup, policy

log = logging.getLogger("pux.container")

# Container-side VNC-web port per ``sandbox.display.backend``. The standard
# image runs x11vnc (-nopw) + websockify/noVNC on 6080; the kasm target runs
# KasmVNC (H.264/WebRTC) on 8444. See ``sandbox/supervisord{,-kasm}.conf``.
_DISPLAY_PORT: dict[str, int] = {"standard": 6080, "kasm": 8444}


def display_port(backend: str) -> int:
    """Container-side VNC-web port for a ``sandbox.display.backend`` value."""
    return _DISPLAY_PORT.get(backend, 6080)


# --- defaults (mirror backend/internal/sandbox/{manager,defaults,cache}.go) ----

# The host project root (for the Docker bind-mount + the ``openshell.project-
# path`` label) comes from the kit's location-independent resolver — not from
# ``__file__`` (which would point into ``pux_harness``'s install dir, not the
# app being sandboxed). ``resolve_project_path()`` still honors ``PUX_PROJECT_PATH``
# as the live per-deploy override. Resolved LIVE at use-site (no import-time
# snapshot) via ``project_root()`` — see ``resolve_project_path`` below.
PROJECT_LABEL = "openshell.project-path"
SANDBOX_LABEL = "openshell.sandbox-id"
POLICY_LABEL = "openshell.policy"

DEFAULT_IMAGE = "pux-sandbox:latest"
DEFAULT_NETWORK = "shared-infra"
DEFAULT_POLICIES_DIR = "/etc/openshell/policies"
DEFAULT_POLICY = "developer"
# Legacy fixed default. Kept as the fallback ONLY when no project path can be
# hashed (should never happen post-``resolve_project_path``). Normal path:
# ``_derive_sandbox_id(project_path)`` makes every project auto-isolate.
DEFAULT_SANDBOX_ID = "mcp-default"


def _derive_sandbox_id(project_path: str) -> str:
    """Deterministic per-project sandbox id: ``p`` + first 8 hex of sha256(abspath).

    The container name (``orchestrator-sandbox-<id>``) and the persist volume
    (``sandbox-<id>-persist``) both key on ``sandbox_id``. With the old fixed
    default (``mcp-default``), launching pux from a SECOND project reused the
    same name → ``create()`` force-removed the first project's RUNNING
    container (silent session murder). Deriving the id from the project path
    means: different projects → different ids → different containers + volumes
    (zero collision, zero guessing); resuming the same project → same hash →
    reuses its container (the desired single-tenant-per-project behavior).

    8 hex chars = 32 bits → ~4 billion ids; collision probability is negligible
    for any realistic project count, and a genuine collision merely falls back
    to the refuse-to-kill-running guard in ``create()`` (no data loss)."""
    h = hashlib.sha256(os.path.abspath(project_path).encode()).hexdigest()[:8]
    return f"p{h}"


def _is_running_name_conflict(exc: Exception) -> bool:
    """True iff ``exc`` is the ``ContainerError`` raised by :meth:`create`'s
    collision handler when the conflicting container is RUNNING (the
    refuse-to-kill path) — the one case :meth:`ensure` can auto-recover from
    by re-checking for a now-registered container (the race where a parallel
    ``ensure()`` in the same project just won ``create()``).

    Narrow on purpose: a STOPPED-stale conflict is already reaped + retried
    inside ``create()`` itself (never reaches here), and any other
    ``ContainerError`` (start failure, egress-cap mismatch, unrelated docker
    error) must propagate unchanged so we don't mask real failures."""
    msg = str(exc)
    return "RUNNING" in msg and ("held by" in msg or "already in use" in msg)


DEFAULT_MEMORY_MB = 2048
DEFAULT_CPU_CORES = 2.0
# Headroom for a full desktop sandbox: Chrome alone spawns 15-30 procs
# (--no-sandbox disables the zygote so every tab/site is a fork), a Vite dev
# server with HMR + esbuild + node workers is another 5-10, and each dispatched
# subagent / Python helper is more. 512 (the old default) EAGAIN'd on fork
# ("Resource temporarily unavailable" / gVisor "procReady not received") once a
# browser org also ran a dev server + subagents — forcing the agent to restart
# the sandbox mid-task. 2048 is 4x headroom while still bounded; operators who
# need tighter/looser bounds override via PUX_SANDBOX_PIDS.
DEFAULT_PIDS = 2048

CACHE_MOUNT_TARGET = "/root/.cache"
CACHE_DISABLED_ENV = "PUX_CACHE_VOLUME"
CACHE_DISABLED_VALUE = "off"


class ContainerError(Exception):
    """Container lifecycle failure (create/start/destroy)."""


# --- env helpers ---------------------------------------------------------------


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key, "")
    return v if v else default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "")
    if not v:
        return default
    try:
        n = int(v)
    except ValueError:
        return default
    return n if n > 0 else default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, "")
    if not v:
        return default
    try:
        f = float(v)
    except ValueError:
        return default
    return f if f > 0 else default


def resolve_project_path() -> str:
    """Absolute project path. ``PUX_PROJECT_PATH`` wins, else the repo root.

    Mirrors Go's ``resolveProjectPath`` — Docker bind mounts require an
    absolute host path, and the label stores that absolute path (the exec path
    filters on it). URL schemes are rejected: their colons corrupt Docker's
    ``host:container[:mode]`` parsing (``ssh://host/path`` lands the container
    path in the mode slot → "invalid mode").

    **Foot-gun warning:** the fallback to ``project_root()`` (the harness repo)
    is what binds the sandbox workspace to the ORCHESTRATOR's own files when a
    launcher forgets to set ``PUX_PROJECT_PATH``. That is the cross-project
    isolation failure — the agent edits the harness repo instead of the user's
    project. When the fallback is taken we emit a loud one-time stderr line so
    the wrong bind is VISIBLE, not silent. ``run_acp`` (the editor-serving
    chokepoint) goes further and derives the path from CWD rather than falling
    back; this guard covers ``direct`` / ``serve`` / tests too.
    """
    p = os.environ.get("PUX_PROJECT_PATH")
    if not p:
        _harness = str(project_root())
        sys.stderr.write(
            "pux sandbox: WARNING — PUX_PROJECT_PATH unset; binding /sandbox/workspace "
            f"to the harness repo fallback ({_harness}). If you spawned this agent "
            "against a DIFFERENT project, it will now edit the orchestrator's own "
            "files. Export PUX_PROJECT_PATH=<your project> to fix.\n"
        )
        p = _harness
    if "://" in p:  # any URL scheme — ssh://, file://, http://, ...
        raise ContainerError(f"sandboxes require a local filesystem path; received a URL: {p!r}")
    return os.path.abspath(p)


# --- runtime (gVisor) — port of runtime.go -----------------------------------


def _is_runsc_available(client: docker.DockerClient) -> bool:
    """True if the daemon has the ``runsc`` runtime registered. Fails closed
    (returns False) on any error — runc is safer than refusing to create."""
    try:
        info = client.info()
    except APIError:
        return False
    return "runsc" in (info.get("Runtimes") or {})


def resolve_runtime(tier: str, runsc_available: bool) -> str | None:
    """Pure decision: which Docker runtime to use (port of resolveRuntime).

    Bridged → never override (runsc + NET_HOST + Xvfb is untested). ``none`` →
    explicit opt-out. TierIsolated + runsc present + env unset → ``runsc``
    (default-on kernel-level isolation). Otherwise None (Docker default runc).
    """
    env_value = os.environ.get("PUX_SANDBOX_RUNTIME", "")
    if tier == "bridged":
        return None
    if env_value == "none":
        return None
    if env_value:
        return env_value
    if tier == "isolated" and runsc_available:
        return "runsc"
    return None


# --- cache volume — port of cache.go -----------------------------------------


def cache_volume_name(project_path: str) -> str:
    """``pux-cache-<sha256(abs)[:16]>`` — deterministic per project so the same
    project reuses its wheels/weights cache across sessions."""
    return "pux-cache-" + hashlib.sha256(os.path.abspath(project_path).encode()).hexdigest()[:16]


def cache_enabled() -> bool:
    return os.environ.get(CACHE_DISABLED_ENV) != CACHE_DISABLED_VALUE


def _ensure_volume(client: docker.DockerClient, name: str) -> None:
    """Create a named local volume if absent. Idempotent."""
    try:
        client.volumes.get(name)
    except NotFound:
        client.volumes.create(name=name, driver="local")
        log.info("created volume %s", name)


# --- the manager --------------------------------------------------------------


class SandboxContainer:
    """Owns create/start/stop/remove for the one CLI-mode sandbox container.

    Stateless apart from the (long-lived) Docker client + the cached container
    name once discovered/created. ``ensure()`` is the entry point used by the
    exec path; ``destroy()`` tears down.
    """

    def __init__(
        self,
        *,
        project_path: str | None = None,
        sandbox_id: str | None = None,
        org: str | None = None,
        client: docker.DockerClient | None = None,
    ):
        self.project_path = project_path or resolve_project_path()
        # sandbox_id precedence: explicit kwarg > $PUX_SANDBOX_ID env >
        # _derive_sandbox_id(project_path). The derive is the default so that
        # two DIFFERENT projects never collide on the fixed ``mcp-default``
        # name (which previously caused ``create()`` to force-remove a running
        # session). Override via env/flag only when running two sessions against
        # the SAME project (an explicit, deliberate collision).
        if sandbox_id is not None:
            self.sandbox_id = sandbox_id
        elif os.environ.get("PUX_SANDBOX_ID"):
            self.sandbox_id = os.environ["PUX_SANDBOX_ID"]
        else:
            self.sandbox_id = _derive_sandbox_id(self.project_path)
        self.org = org if org is not None else os.environ.get("PUX_ORG", "")
        self._client = client
        self._name: str | None = None
        self._watch_url: str | None = None  # cached during create(); see watch_url

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env(timeout=300)
        return self._client

    @property
    def name(self) -> str:
        return f"orchestrator-sandbox-{self.sandbox_id}"

    # -- discovery --------------------------------------------------------------

    def _running_for_project(self) -> str | None:
        """The running container name for this project (label filter), or None.

        Single-tenant invariant: >1 match is an anomaly we raise on rather than
        silently exec the wrong one.
        """
        try:
            cs = self.client.containers.list(
                filters={"label": f"{PROJECT_LABEL}={self.project_path}", "status": "running"},
            )
        except APIError as exc:
            raise ContainerError(f"docker list failed: {exc}") from exc
        if not cs:
            return None
        if len(cs) > 1:
            names = sorted(c.name for c in cs)
            raise ContainerError(
                f"single-tenant invariant violated: {len(cs)} running containers for "
                f"project {self.project_path!r}: {names}"
            )
        return cs[0].name

    def _safe_remove(
        self,
        container: docker.models.containers.Container,
        *,
        name: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Force-remove ``container``, tolerating a concurrent in-flight removal.

        Docker returns ``409 Conflict ("removal of container X is already in
        progress")`` when two callers force-remove the same container at once —
        the exact race two ``pux acp`` sessions sharing the deterministic name
        ``orchestrator-sandbox-<id>`` hit when one tears down while another
        recreates. The removal the OTHER caller initiated achieves what WE
        wanted (the container gone), so we poll ``containers.get(name)`` until
        it raises :class:`NotFound` and return cleanly instead of surfacing the
        409 as a hard :class:`ContainerError` (the "manual yegsting" failure
        the user hit). Any other :class:`APIError` (auth, daemon down) re-raises
        unchanged so real failures still surface.
        """
        label = name or self.name
        try:
            container.remove(force=True)
            return
        except NotFound:
            return  # already gone — exactly what we wanted
        except APIError as exc:
            if "already in progress" not in str(exc):
                raise  # not the race — let real docker errors surface
        # 409 race: a concurrent removal is doing our job. Wait for it to land
        # (container → NotFound) instead of erroring in the user's face.
        log.info(
            "removal of %s already in progress (concurrent teardown) — waiting it out",
            label,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.5)
            try:
                self.client.containers.get(label)
            except NotFound:
                return  # the concurrent removal finished; container is gone
        log.warning(
            "concurrent removal of %s did not finish in %.1fs; one more attempt",
            label, timeout,
        )
        try:
            self.client.containers.get(label).remove(force=True)
        except NotFound:
            return
        raise ContainerError(
            f"remove {label}: concurrent removal stalled past {timeout:.1f}s"
        )

    def _remove_stale(self) -> None:
        """Remove a STOPPED container with our deterministic name if one lingers
        from a prior crash. Mirrors Go's "already in use" → force-remove retry,
        but scoped to stopped containers so we never clobber a live sandbox."""
        try:
            c = self.client.containers.get(self.name)
        except NotFound:
            return
        if c.status == "running":
            return  # not ours to touch — discovery should have found it
        log.info("removing stale stopped container %s", self.name)
        self._safe_remove(c, name=self.name)

    # -- create -----------------------------------------------------------------

    def _resolve_policy(self) -> tuple[policy.Policy | None, str]:
        """Load the org policy (or None). Returns (policy, effective_tier)."""
        fallback_tier = "isolated"  # CLI mode default — Go normalizes empty→isolated
        if not self.org:
            return None, fallback_tier
        try:
            pol = policy.load(self.org, self.project_path)
        except policy.NoPolicy:
            return None, fallback_tier
        return pol, policy.resolve_tier(pol, fallback_tier)

    def _build_env(self, pol: policy.Policy | None) -> list[str]:
        # Go defaults opts.Policy to "developer"; no knob is exposed, so the
        # SANDBOX_POLICY env + the openshell.policy label stay "developer".
        # The legacy NETWORK_ALLOW / FS_READONLY / FS_READWRITE env vars were
        # removed — nothing in the image or harness reads them (grep-verified
        # against the live container 2026-07-11). Egress is enforced via the
        # egress.conf → apply-egress-policy.sh → iptables path, NOT an env var.
        env = [
            f"SANDBOX_POLICY={DEFAULT_POLICY}",
            "DOCKER_HOST=unix:///var/run/docker.sock",
            "HOST_GATEWAY=host.docker.internal",
            # Where the Agent Protocol server (Aegra) lives, so in-container prep
            # jobs (e.g. warmup_webhook) + agent code can reach it. Aegra binds
            # the Tailscale IP, so the container uses this host (host.docker.internal
            # would NOT connect — Aegra isn't on the docker-gateway iface).
            f"PUX_API_HOST={_env_str('PUX_API_HOST', '127.0.0.1')}",
            f"PUX_API_PORT={_env_str('PUX_API_PORT', '9988')}",
        ]
        # Policy creds/cookies (required validated before this point). Appended
        # last → last-wins, matching Docker semantics.
        env.extend(policy.env_vars(pol))
        return env

    def _build_binds(self, pol: policy.Policy | None) -> list[str]:
        policies_dir = _env_str("OPENSHELL_POLICIES_DIR", DEFAULT_POLICIES_DIR)
        persist = f"sandbox-{self.sandbox_id}-persist"
        binds = [
            f"{self.project_path}:/sandbox/workspace",
            f"{policies_dir}:/etc/openshell/policies:ro",
            "/tmp:/sandbox/tmp",
            f"{persist}:/sandbox/persist",
        ]
        _ensure_volume(self.client, persist)
        # Per-project cache volume (disabled via PUX_CACHE_VOLUME=off).
        if cache_enabled():
            cache_vol = cache_volume_name(self.project_path)
            _ensure_volume(self.client, cache_vol)
            binds.append(f"{cache_vol}:{CACHE_MOUNT_TARGET}")
        # Org-declared workspace mounts (policy.workspace.mounts, ${VAR} expanded).
        for m in policy.resolve_mounts(pol):
            binds.append(f"{m.host}:{m.container}:{m.mode}")
            log.info("policy mount %s <- %s (%s)", m.container, m.host, m.mode)
        return binds

    def create(self) -> str:
        """Create + start the sandbox container. Returns its name. Assumes no
        running container for this project (call ``ensure`` for the gated path)."""
        pol, effective_tier = self._resolve_policy()

        if pol is not None:
            # host_setup runs FIRST (before validate_env): its exports populate
            # the process env so the existing credentials.required /
            # browser.cookies_env / env_vars / seed-cookies.sh chain consumes
            # them UNCHANGED — one mechanism, not two. No-op with no hooks.
            exports = host_setup.run_host_setup(pol, self.project_path)
            if exports:
                os.environ.update(exports)
                log.info("host_setup exported %d env var(s): %s", len(exports), sorted(exports))

            # Validate required creds BEFORE touching Docker (cheapest, fails
            # fast, no container leak). Resolve mounts too (unset ${VAR} fails
            # loud). Runs AFTER host_setup so creds the hooks produced are seen.
            policy.validate_env(pol)
            policy.resolve_mounts(pol)  # raises UnresolvedMount on first unset

        # Image: policy sandbox.image override wins; else env/default. Build when
        # sandbox.build is set + image absent; else pull if absent.
        image = _env_str("OPENSHELL_IMAGE", DEFAULT_IMAGE)
        if pol is not None and pol.sandbox.image:
            image = pol.sandbox.image
        self._ensure_image(image, policy.build_spec(pol))

        env = self._build_env(pol)
        binds = self._build_binds(pol)

        # Resource limits. Precedence: policy (org declares what it needs) >
        # env (operator escape hatch) > default. Browser orgs declare
        # ``sandbox.memory_mb`` because Chrome + SeleniumBase cold-start OOMs
        # under the lean 2 GiB default; coding orgs leave it unset and inherit
        # the default. See ``policy.resolve_resources``.
        fb_mem = _env_int("PUX_SANDBOX_MEMORY_MB", DEFAULT_MEMORY_MB)
        fb_cpu = _env_float("PUX_SANDBOX_CPU_CORES", DEFAULT_CPU_CORES)
        mem_mb, cpu_cores = policy.resolve_resources(
            pol, fallback_memory_mb=fb_mem, fallback_cpu_cores=fb_cpu
        )
        mem_bytes = mem_mb * 1024 * 1024
        nano_cpus = int(cpu_cores * 1e9)
        pids = _env_int("PUX_SANDBOX_PIDS", DEFAULT_PIDS)
        if mem_mb != DEFAULT_MEMORY_MB or cpu_cores != DEFAULT_CPU_CORES:
            log.info(
                "sandbox resources: %d MiB / %.1f CPU (default %d MiB / %.1f)",
                mem_mb, cpu_cores, DEFAULT_MEMORY_MB, DEFAULT_CPU_CORES,
            )

        runtime = resolve_runtime(effective_tier, _is_runsc_available(self.client))

        # Container User override (run_as_host_user → host UID:GID).
        user = ""
        if pol is not None and pol.workspace.run_as_host_user:
            user = policy.host_user()
            log.info("container will run as host user %s", user)

        cap_add: list[str] = []
        # Stage egress allowlist + grant NET_ADMIN. Skipped for effective Bridged
        # (host networking makes iptables-in-container meaningless) + when empty.
        if effective_tier != "bridged" and pol is not None and pol.egress.allow:
            rules = policy.egress_rules(pol)  # resolves DNS now — loud on failure
            egress_dir = os.path.join(self.project_path, ".pux")
            os.makedirs(egress_dir, exist_ok=True)
            egress_path = os.path.join(egress_dir, "egress.conf")
            with open(egress_path, "w", encoding="utf-8") as fh:
                fh.write(rules)
            os.chmod(egress_path, 0o600)
            cap_add.append("NET_ADMIN")
            log.info(
                "staged egress allowlist (%d rules) + NET_ADMIN → %s",
                len(pol.egress.allow),
                egress_path,
            )

        create_kwargs: dict[str, Any] = dict(
            image=image,
            name=self.name,
            environment=env,
            volumes=binds,
            labels={
                POLICY_LABEL: DEFAULT_POLICY,
                SANDBOX_LABEL: self.sandbox_id,
                PROJECT_LABEL: self.project_path,
            },
            mem_limit=mem_bytes,
            nano_cpus=nano_cpus,
            pids_limit=pids,
            extra_hosts={"host.docker.internal": "host-gateway"},
            # Block setuid/setcap privilege escalation inside the container.
            # Zero impact on normal work (the agent runs as root already; this
            # prevents a compromised process from gaining NEW caps via suid
            # binaries like passwd, sudo, ping).
            security_opt=["no-new-privileges:true"],
        )
        if user:
            create_kwargs["user"] = user
        if runtime:
            create_kwargs["runtime"] = runtime
        if cap_add:
            create_kwargs["cap_add"] = cap_add

        # Display: when ``sandbox.display.watch`` is on, publish the VNC-web port
        # so the live desktop is browser-reachable. Isolated tier gets an explicit
        # port binding (host net already exposes the port directly). Bound to
        # 127.0.0.1 ONLY with an ephemeral host port — x11vnc runs -nopw, so it is
        # never LAN-exposed; the explicit opt-in carries that security property.
        watch_on = pol is not None and pol.sandbox.display.watch
        disp = pol.sandbox.display if watch_on else None

        # Network: shared-infra for isolated; host net for bridged (skips ACLs).
        # The sandbox image runs its OWN Xvfb on :99 (Dockerfile ENV DISPLAY=:99,
        # supervisord %(ENV_DISPLAY)s) — it never shares the host's X server. We
        # do NOT passthrough the host DISPLAY or mount /tmp/.X11-unix: doing so
        # overwrites :99 with the host's display (e.g. :0), and Xvfb fails with
        # "Cannot establish any listening sockets" because that display is already
        # owned by the host. The container's own DISPLAY env wins by not overriding.
        if effective_tier == "bridged":
            create_kwargs["network_mode"] = "host"
        else:
            create_kwargs["network"] = _env_str("OPENSHELL_NETWORK", DEFAULT_NETWORK)
            if disp is not None:
                # Ephemeral host port (None) on localhost; read back after start.
                create_kwargs["ports"] = {f"{display_port(disp.backend)}/tcp": ("127.0.0.1", None)}

        self._remove_stale()
        log.info(
            "creating container %s (image=%s tier=%s runtime=%s)",
            self.name,
            image,
            effective_tier,
            runtime or "default",
        )
        try:
            container = self.client.containers.create(**create_kwargs)
        except APIError as exc:
            # Name conflict. Distinguish a STOPPED stale container (safe to
            # reap — the original intent of this branch: a stopped leftover
            # that ``_remove_stale`` raced on) from a RUNNING one. A running
            # container under this name is a LIVE session — force-removing it
            # would silently murder another agent's browser mid-task. Refuse
            # loud so the operator picks: reuse it, isolate via a different
            # sandbox_id, or take it over explicitly with ``docker rm -f``.
            msg = str(exc)
            if "already in use" in msg or "Conflict" in msg:
                try:
                    existing = self.client.containers.get(self.name)
                except NotFound:
                    existing = None
                if existing is not None and existing.status == "running":
                    raise ContainerError(
                        f"sandbox name {self.name!r} is held by a RUNNING container "
                        f"(id {existing.id[:12]}). pux refuses to force-remove a live "
                        f"session. Options:\n"
                        f"  • resume it: it is already up — re-run from the same "
                        f"project dir.\n"
                        f"  • isolate: launch a second concurrent session with a "
                        f"different id — `pux acp --sandbox-id <name>` or export "
                        f"PUX_SANDBOX_ID=<name>.\n"
                        f"  • take over explicitly: `docker rm -f {self.name}` then "
                        f"re-run (the previous session's workspace is safe — it is a "
                        f"host bind-mount, not container-internal storage)."
                    ) from exc
                if existing is not None:
                    log.info(
                        "name conflict on %s — removing STOPPED stale + retry",
                        self.name,
                    )
                    self._safe_remove(existing, name=self.name)
                container = self.client.containers.create(**create_kwargs)
            else:
                raise ContainerError(f"create {self.name}: {exc}") from exc

        try:
            container.start()
        except APIError as exc:
            self._safe_remove(container, name=self.name)
            raise ContainerError(f"start {self.name}: {exc}") from exc

        # Resolve the watch URL now that the container is up (isolated: read the
        # ephemeral host port docker assigned; bridged: host port == container
        # port). Cached so the reuse path doesn't need to re-inspect.
        self._watch_url = self._resolve_watch_url(container, disp, effective_tier)
        if self._watch_url:
            log.info("watchable desktop for %s: %s", self.name, self._watch_url)

        # Restore persisted state (Chrome profile, dotfiles, packages). Fire-and-
        # forget — failures don't block the sandbox (Go treats them the same way).
        self._restore_persisted(container)
        # Install the org's declared sandbox.deps (apt + pip) — best-effort, after
        # start + restore, before workspace scaffold. A blocked mirror or bad
        # package logs a warning and never breaks the run (mirror run_jobs /
        # host_setup). Egress for pypi + Debian mirrors must be on the org's
        # allowlist — install is NOT auto-allowed (explicit-egress).
        if pol is not None:
            self._install_deps(container, pol)
        # Scaffold writable workspace dirs + chown to the host project owner so
        # host-side tools can read artifacts the agent writes.
        self._scaffold_workspace(container)

        log.info("sandbox ready: %s (container %s)", self.name, container.id[:12])
        self._name = self.name
        return self.name

    def _resolve_watch_url(
        self,
        container: docker.models.containers.Container,
        disp: policy.DisplaySpec | None,
        effective_tier: str,
    ) -> str | None:
        """Browser URL for the live desktop, given a running container.

        ``disp`` is the resolved ``sandbox.display`` (None when watch is off).
        Bridged (host net): the container's port IS the host's own. Isolated:
        read the ephemeral host port docker assigned to the published binding.
        KasmVNC serves its own TLS web client on the port; noVNC (standard) is
        websockify's plain-HTTP client at ``/vnc.html``."""
        if disp is None:
            return None
        dport = display_port(disp.backend)
        if effective_tier == "bridged":
            host_port = dport
        else:
            container.reload()
            binds = (container.ports or {}).get(f"{dport}/tcp")
            if not binds:
                return None
            host_port = int(binds[0]["HostPort"])
        if disp.backend == "kasm":
            return f"https://127.0.0.1:{host_port}/"
        return f"http://127.0.0.1:{host_port}/vnc.html"

    @property
    def watch_url(self) -> str | None:
        """Browser URL for the live desktop when ``sandbox.display.watch`` is on.

        Works on both the create path (cached in ``self._watch_url``) and the
        reuse path (inspects the running container's port bindings). Returns
        None when watch is off or no container is up — so callers can print it
        unconditionally and stay quiet when the org hasn't opted in."""
        if self._watch_url:
            return self._watch_url
        pol, tier = self._resolve_policy()
        if pol is None or not pol.sandbox.display.watch:
            return None
        name = self._name or self._running_for_project()
        if not name:
            return None
        try:
            c = self.client.containers.get(name)
        except NotFound:
            return None
        return self._resolve_watch_url(c, pol.sandbox.display, tier)

    def _ensure_image(self, image: str, build: policy.BuildSpec | None = None) -> None:
        """Guarantee ``image`` exists locally. If already present, return. If
        absent AND a ``build`` spec is set, build it from the org's Dockerfile
        (host-side Docker SDK, no compose); else pull. ``build.dockerfile`` is
        project-relative (or absolute); ``build.context`` defaults to the
        Dockerfile's directory when blank. The ``dockerfile`` arg passed to the
        SDK is the basename — it's resolved relative to the build context."""
        try:
            self.client.images.get(image)
            return
        except ImageNotFound:
            pass
        if build is not None:
            dockerfile_path = Path(build.dockerfile)
            context = build.context or str(dockerfile_path.parent)
            if not Path(context).is_absolute():
                context = str(Path(self.project_path) / context)
            log.info(
                "building image %s (dockerfile=%s context=%s)", image, build.dockerfile, context
            )
            try:
                self.client.images.build(path=context, dockerfile=dockerfile_path.name, tag=image)
            except (APIError, BuildError) as exc:
                raise ContainerError(f"build image {image}: {exc}") from exc
            return
        log.info("pulling image %s", image)
        try:
            self.client.images.pull(image)
        except APIError as exc:
            raise ContainerError(f"pull image {image}: {exc}") from exc

    def _exec(self, container: docker.models.containers.Container, cmd: str) -> None:
        """Best-effort exec after start; non-zero exit is logged not fatal."""
        try:
            result = container.exec_run(["bash", "-c", cmd], tty=False, demux=False)
        except APIError as exc:
            log.warning("post-start exec failed: %s", exc)
            return
        code = result.exit_code if result.exit_code is not None else 0
        if code != 0:
            out = result.output
            if isinstance(out, (bytes, bytearray)):
                out = out.decode("utf-8", "replace")
            log.warning("post-start exec exit %d: %s", code, (out or "").strip()[:200])

    def _restore_persisted(self, container: docker.models.containers.Container) -> None:
        # Verbatim port of manager.go::restorePersistedState.
        self._exec(container, _RESTORE_SCRIPT)

    def _install_deps(
        self, container: docker.models.containers.Container, pol: policy.Policy
    ) -> None:
        """Install the org's declared ``sandbox.deps`` (apt + pip) into the
        container. Apt and pip are independent best-effort steps (each via
        ``_exec``, which logs non-zero exit as a warning, never fatal) — a
        blocked Debian mirror does NOT also block a pypi install attempt. No-op
        when both lists are empty (today's default for every org)."""
        deps = pol.sandbox.deps
        if not deps.apt and not deps.pip:
            return
        log.info(
            "installing sandbox deps (%s): apt=%s pip=%s",
            pol.sandbox.image or DEFAULT_IMAGE,
            deps.apt,
            deps.pip,
        )
        if deps.apt:
            quoted = " ".join(shlex.quote(p) for p in deps.apt)
            self._exec(
                container,
                f"apt-get update -qq && apt-get install -y --no-install-recommends {quoted}",
            )
        if deps.pip:
            quoted = " ".join(shlex.quote(p) for p in deps.pip)
            self._exec(container, f"python3 -m pip install --no-cache-dir {quoted}")

    def _scaffold_workspace(self, container: docker.models.containers.Container) -> None:
        for d in ("/sandbox/workspace/memos", "/sandbox/workspace/.pux/sessions"):
            self._exec(container, f"mkdir -p {d}")
        # Chown to the host project dir owner so host-side tools can read/write.
        try:
            uid = os.stat(self.project_path).st_uid
        except OSError:
            return
        for d in ("/sandbox/workspace/memos", "/sandbox/workspace/.pux/sessions"):
            self._exec(container, f"chown -R {uid}:{uid} {d}")

    # -- public entry points ----------------------------------------------------

    def ensure(self) -> str:
        """Return the running sandbox container name, creating it if absent.

        The single-tenant gate: an already-running container for this project is
        reused (whoever booted it); we only create when none is running.

        Fail-closed egress: when the org's policy declares an ``egress.allow``
        block, the reused container MUST have ``NET_ADMIN`` (without it the
        deny-by-default iptables rules can't be installed → traffic flows
        unrestricted, defeating the policy). A stale container from before the
        policy was added (or created by a path that skipped the cap) is REJECTED
        here rather than silently reused — the operator destroys it and a fresh
        one is created with the right caps.

        Cross-session wayfinding: on entry we emit a project-switch banner if
        this session is binding a DIFFERENT host path than the last session
        (``projects.warn_if_switched``). ``/sandbox/workspace`` is a bind-mount
        window onto a host dir, not storage — launching from a different CWD
        silently points that window at a different project, which has caused
        users to believe prior work was lost (it wasn't; the window moved).
        After a successful boot/reuse we record the project so the next switch
        is detectable. Both calls are best-effort and never block boot.
        """
        # Lazy import — keeps the module-load graph acyclic and matches the
        # inline-import style used elsewhere in the harness.
        from pux_harness.sandbox import projects as _projects  # noqa: PLC0415

        if self._name:
            return self._name
        _projects.warn_if_switched(self.project_path)
        running = self._running_for_project()
        if running:
            self._validate_reused_container(running)
            self._name = running
            log.info("reusing running container %s", running)
            _projects.record(self.project_path, self.sandbox_id, self.org)
            return running
        try:
            name = self.create()
        except ContainerError as exc:
            # Auto-recovery for the name-conflict race: two ensure() calls in
            # the SAME project booted simultaneously — one won create(), the
            # other hit the refuse-to-kill guard (the winner's container is
            # RUNNING under this project's name). Re-check: if the winner has
            # now registered, reuse it; else retry create() once (the winner
            # may have died). Escalate on a second failure so we never loop.
            # Scoped by construction: the container name is derived from THIS
            # project's path (_derive_sandbox_id), so this can NEVER reach
            # across to a sibling project's container — the refuse-to-kill
            # guard's safety property is preserved.
            if not _is_running_name_conflict(exc):
                raise
            log.warning(
                "name conflict on create() (race?) — re-checking for a now-"
                "running container, then one retry: %s", exc,
            )
            running = self._running_for_project()
            if running:
                self._validate_reused_container(running)
                self._name = running
                log.info("recovered via reuse after create() race: %s", running)
                _projects.record(self.project_path, self.sandbox_id, self.org)
                return running
            name = self.create()  # exactly one retry
        _projects.record(self.project_path, self.sandbox_id, self.org)
        return name

    def _validate_reused_container(self, name: str) -> None:
        """Reject a reused container whose security posture doesn't match the
        org's policy. Currently checks egress caps — extend as new invariants
        are identified."""
        pol, effective_tier = self._resolve_policy()
        needs_egress = (
            effective_tier != "bridged"
            and pol is not None
            and bool(pol.egress.allow)
        )
        if not needs_egress:
            return
        try:
            c = self.client.containers.get(name)
        except NotFound:
            return  # vanished between list + get — create() will make a new one
        host_config = c.attrs.get("HostConfig", {})
        cap_add = host_config.get("CapAdd") or []
        if "NET_ADMIN" not in cap_add:
            raise ContainerError(
                f"reused container {name!r} lacks NET_ADMIN but the org policy "
                f"declares an egress.allow block — the deny-by-default firewall "
                f"cannot be enforced. Destroy the stale container "
                f"(``docker rm -f {name}``) and re-run so a fresh one is created "
                f"with the correct capabilities."
            )

    def destroy(self) -> None:
        """Stop + remove the sandbox container, ALWAYS saving persisted state.

        Data-loss fix (2026-07-12, gap 6 of the persistence audit): the previous
        version skipped ``_save_persisted`` when the container was already
        stopped — a stopped container can't be exec'd into. That silently
        dropped Chrome profile changes, new apt installs, and dotfile updates
        made during the session whenever a caller did ``stop`` then ``destroy``
        (e.g. ``pux sandbox stop``). The save now starts a stopped container
        briefly so the exec can run, then stops + removes it. Start failures
        surface as :class:`ContainerError` (no silent skip — verify or die)."""
        name = self._name or self.name
        try:
            container = self.client.containers.get(name)
        except NotFound:
            self._name = None
            return
        if container.status != "running":
            self._start_for_save(container, name)
        self._save_persisted(container)
        try:
            container.stop(timeout=10)
        except APIError as exc:
            log.warning("stop %s failed (non-fatal): %s", name, exc)
        self._safe_remove(container, name=name)
        log.info("sandbox destroyed: %s", name)
        self._name = None

    def reset(self) -> None:
        """Force-remove the sandbox container WITHOUT saving persisted state —
        a fast, unconditional teardown for a stuck/broken sandbox.

        Unlike ``destroy()`` (which execs into the container to save the Chrome
        profile + installed packages and can therefore HANG on a container that
        isn't responding), ``reset()`` skips the save and force-removes directly
        via the hardened ``_safe_remove`` (tolerant of a concurrent 409 removal
        race). The next ``ensure()`` recreates a fresh container from scratch.
        Use for recovery when a sandbox is wedged; prefer ``destroy()`` for a
        clean, state-preserving shutdown.
        """
        name = self.name
        try:
            container = self.client.containers.get(name)
        except NotFound:
            self._name = None
            return  # already gone — nothing to reset
        self._safe_remove(container, name=name)
        log.info("sandbox reset (force-removed, no save): %s", name)
        self._name = None

    # -- pause / unpause (true session preservation without teardown) --------

    def pause(self) -> None:
        """Freeze the sandbox container's processes in place (cgroup freezer).

        Unlike ``destroy()`` (which ``pux sandbox stop`` calls), pause keeps the
        container ALIVE — every process (Chrome with its open tabs, the Xvfb
        display, any long-running agent loop) is frozen at the exact instruction
        pointer, NOT killed. ``unpause`` resumes them as if nothing happened.

        This is the right answer to "I want to keep my session but free the
        CPU": the container stays in memory, ready to resume in milliseconds.
        Stop/start cycles rebuild the container from the image + restore the
        named volume (slower, and the running-state layer is reset)."""
        name = self._name or self.name
        try:
            container = self.client.containers.get(name)
        except NotFound as exc:
            raise ContainerError(
                f"cannot pause {name!r} — no such container. "
                f"Use `pux sandbox start` first."
            ) from exc
        if container.status != "running":
            raise ContainerError(
                f"cannot pause {name!r} — container is {container.status!r}, "
                f"not 'running'. Only running containers can be paused."
            )
        try:
            container.pause()
        except APIError as exc:
            raise ContainerError(f"pause {name}: {exc}") from exc
        log.info("sandbox paused: %s (processes frozen, memory resident)", name)

    def unpause(self) -> None:
        """Thaw a paused sandbox — every frozen process resumes in place."""
        name = self._name or self.name
        try:
            container = self.client.containers.get(name)
        except NotFound as exc:
            raise ContainerError(
                f"cannot unpause {name!r} — no such container."
            ) from exc
        if container.status != "paused":
            raise ContainerError(
                f"cannot unpause {name!r} — container is {container.status!r}, "
                f"not 'paused'."
            )
        try:
            container.unpause()
        except APIError as exc:
            raise ContainerError(f"unpause {name}: {exc}") from exc
        log.info("sandbox unpaused: %s (processes resumed)", name)

    def _start_for_save(
        self, container: docker.models.containers.Container, name: str,
    ) -> None:
        """Start a stopped container so ``_save_persisted`` can exec into it.

        Polls ``container.status`` for up to 5 s after ``start()`` returns (start
        is async at the daemon level). Raises :class:`ContainerError` if the
        container won't run — never silently skips the save."""
        import time  # noqa: PLC0415
        try:
            container.start()
        except APIError as exc:
            raise ContainerError(
                f"cannot start stopped container {name} to save persisted "
                f"state during destroy (data would be lost): {exc}"
            ) from exc
        for _ in range(10):
            container.reload()
            if container.status == "running":
                return
            time.sleep(0.5)
        raise ContainerError(
            f"container {name} did not reach running state after start "
            f"(status={container.status}); cannot save persisted state."
        )

    def _save_persisted(self, container: docker.models.containers.Container) -> None:
        # Verbatim port of manager.go::savePersistedState.
        self._exec(container, _SAVE_SCRIPT)

    # -- persist volume dump (gap 5: named-volume extraction) ------------------

    def persist_volume_name(self) -> str:
        """The named Docker volume that backs ``/sandbox/persist``."""
        return f"sandbox-{self.sandbox_id}-persist"

    def dump_persist(self, output_path: str) -> str:
        """Stream the named persist volume to a host tarball.

        Uses ``alpine:latest`` as a throwaway container with the volume mounted
        read-only, tars ``/p``, streams the gzipped bytes to ``output_path``.
        Works whether the sandbox is running or not — operates on the volume,
        not the sandbox container. Raises :class:`ContainerError` if the volume
        is absent (the sandbox never started) or the dump command fails.

        Persists the user-recoverable bits named-volume-side that the
        bind-mount does NOT cover: the Chrome profile (cookies + sessions),
        the recorded apt install list (so a rebuild is fast), and dotfiles
        written under ``/root`` during the session."""
        persist = self.persist_volume_name()
        try:
            self.client.volumes.get(persist)
        except NotFound as exc:
            raise ContainerError(
                f"persist volume {persist!r} does not exist; nothing to dump "
                f"(the sandbox has never been started)."
            ) from exc
        # Pull alpine lazily; it's ~8 MB. If the pull fails AND the image is
        # absent, containers.run below will raise NotFound — surface it.
        try:
            self.client.images.get("alpine:latest")
        except ImageNotFound:
            try:
                self.client.images.pull("alpine:latest")
            except APIError as exc:
                raise ContainerError(
                    f"alpine:latest is absent and pull failed (needed to read "
                    f"the persist volume): {exc}"
                ) from exc
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        try:
            chunks = self.client.containers.run(
                image="alpine:latest",
                command=["sh", "-c", "tar czf - -C /p ."],
                volumes={persist: {"bind": "/p", "mode": "ro"}},
                remove=True,
                detach=False,
                stream=True,
                stderr=False,
            )
        except APIError as exc:
            raise ContainerError(f"dump-persist run failed: {exc}") from exc
        with open(output_path, "wb") as f:
            n = 0
            for chunk in chunks:
                if chunk:
                    f.write(chunk)
                    n += len(chunk)
        if n == 0:
            raise ContainerError(
                f"dump-persist produced 0 bytes for volume {persist!r} "
                f"(the volume may be empty or the tar command failed silently)."
            )
        log.info("dumped persist volume %s → %s (%d bytes)", persist, output_path, n)
        return output_path


# --- persisted-state scripts (verbatim from manager.go) -----------------------

_RESTORE_SCRIPT = r"""# Restore Chrome profile
if [ -d /tmp/chrome-profile ] && [ -z "$(ls -A /tmp/chrome-profile 2>/dev/null)" ] && [ -d /sandbox/persist/chrome-profile ] && [ -n "$(ls -A /sandbox/persist/chrome-profile 2>/dev/null)" ]; then
  cp -a /sandbox/persist/chrome-profile/. /tmp/chrome-profile/
  echo "Restored Chrome profile"
fi
# Restore home dotfiles
if [ -d /sandbox/persist/home ] && [ -n "$(ls -A /sandbox/persist/home 2>/dev/null)" ]; then
  cp -a /sandbox/persist/home/. /root/ 2>/dev/null
  echo "Restored home dotfiles"
fi
# Reinstall previously installed packages (only those not in base image)
if [ -f /sandbox/persist/installed-packages.txt ] && [ -s /sandbox/persist/installed-packages.txt ]; then
  dpkg-query -W -f='${Package}\n' 2>/dev/null | sort > /tmp/base-packages.txt
  comm -23 /sandbox/persist/installed-packages.txt /tmp/base-packages.txt > /tmp/extra-packages.txt
  if [ -s /tmp/extra-packages.txt ]; then
    EXTRA=$(cat /tmp/extra-packages.txt | tr '\n' ' ')
    apt-get update -qq 2>/dev/null
    apt-get install -y $EXTRA 2>/dev/null
    echo "Restored $(wc -w < /tmp/extra-packages.txt) extra packages"
  fi
fi"""

_SAVE_SCRIPT = r"""# Save Chrome profile
if [ -d /tmp/chrome-profile ]; then
  mkdir -p /sandbox/persist/chrome-profile
  cp -a /tmp/chrome-profile/. /sandbox/persist/chrome-profile/
  echo "Saved Chrome profile"
fi
# Save home dotfiles
if [ -d /root ]; then
  mkdir -p /sandbox/persist/home
  cp -a /root/.bashrc /root/.profile /root/.wget-hsts /root/.config /sandbox/persist/home/ 2>/dev/null
  echo "Saved home dotfiles"
fi
# Save list of installed packages (for reinstallation on next start)
dpkg-query -W -f='${Package}\n' 2>/dev/null | sort > /sandbox/persist/installed-packages.txt
echo "Saved $(wc -l < /sandbox/persist/installed-packages.txt) packages list"
"""


# --- prepare (post-create, pre-agent jobs) ----------------------------------


def prepare(
    org: str,
    project_path: str | None = None,
    exec_client: Any | None = None,
    universal_warmup: bool = False,
) -> list[dict[str, Any]]:
    """Run post-create, pre-agent prep jobs inside the sandbox container.

    Called by the prepare() entry points (``main.py`` for ``pux direct``; the
    Aegra runtime via ``PrepareWarmupMiddleware`` for prod) after ``ensure()``,
    before ``graph.ainvoke()``. Returns a list of result dicts for logging/display.

    Idempotency is delegated to the scripts themselves (file caches +
    SurrealDB UPSERTs make repeat runs cheap).

    ``universal_warmup``: when True (the serve path), additionally run
    ``warmup_webhook`` for EVERY org — not a policy job, so it covers orgs that
    declare no ``jobs:`` — verifying the run-completion event endpoint
    (the Aegra runtime's ``/events/health``) is reachable from this sandbox, so a
    webhook-less client (Hermes) can observe completions. The ``direct`` path
    leaves it False (no serve in direct mode => the check would always fail).

    Uses lazy imports to avoid circular dependency with docker_exec.
    """
    import time  # noqa: PLC0415

    from pux_harness.sandbox.docker_exec import DockerExecClient  # noqa: PLC0415
    from pux_harness.sandbox.jobs import JobResult, run_jobs  # noqa: PLC0415

    if not project_path:
        project_path = resolve_project_path()

    try:
        pol = policy.load(org, project_path)
    except policy.NoPolicy:
        pol = None

    specs = policy.job_specs(pol) if pol is not None else []

    # Universal warmup needs a container even when the org declares no policy
    # jobs (it is the "for ALL sandboxes" step). Otherwise the historical
    # short-circuit stands: no declared jobs => no container touched here.
    if not specs and not universal_warmup:
        return []

    if exec_client is None:
        sb = SandboxContainer(project_path=project_path, org=org)
        exec_client = DockerExecClient(container=sb.ensure())

    results: list[JobResult] = list(run_jobs(pol, exec_client)) if specs else []

    if universal_warmup:
        # ALL serve sandboxes: prove the run-completion notification path is
        # alive from this container before the agent loop. Warn-and-continue.
        t0 = time.monotonic()
        try:
            out, rc = exec_client.exec(
                "python3 orgs/_shared/sandbox/warmup_webhook.py", timeout=30
            )
            results.append(JobResult(
                name="warmup_webhook",
                status="ok" if rc == 0 else "failed",
                error=None if rc == 0 else (out[-500:] if out else f"exit {rc}"),
                duration=time.monotonic() - t0,
            ))
        except Exception as exc:  # noqa: BLE001 - prep must never block the run
            results.append(JobResult(
                name="warmup_webhook", status="failed",
                error=str(exc)[:500], duration=time.monotonic() - t0,
            ))

    return [
        {"name": r.name, "status": r.status, "error": r.error, "duration": round(r.duration, 1)}
        for r in results
    ]
