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
  env          ``SANDBOX_POLICY``/``NETWORK_ALLOW``/``FS_READONLY``/
               ``FS_READWRITE``/``DOCKER_HOST``/``HOST_GATEWAY`` + policy creds.
  resources    mem 2048MB / cpu 2.0 cores / pids 512 (``PUX_SANDBOX_MEMORY_MB``/
               ``PUX_SANDBOX_CPU_CORES``/``PUX_SANDBOX_PIDS`` env knobs).
  runtime      ``runsc`` when TierIsolated + installed; ``PUX_SANDBOX_RUNTIME``
               overrides (``none`` opts out). Bridged tier never overrides.
  network      ``shared-infra`` (``OPENSHELL_NETWORK``); host net for Bridged.
  extra_hosts  ``host.docker.internal:host-gateway``.
  caps         ``NET_ADMIN`` only when policy stages an egress allowlist.

Policy enforcement — the part that was blocked while the Go binary owned the
container (the *resolver* was ported; the *enforcer* stayed Go because it
mutates ``container.HostConfig`` at create) — runs here now. It mirrors
``backend/internal/sandbox/policy_hook.go::applyOrgPolicy`` step for step:
load → validate required creds → resolve ``${VAR}`` mounts → inject creds/cookies
env → ``run_as_host_user`` → ``UID:GID`` → stage ``<project>/.pux/egress.conf``
+ grant ``NET_ADMIN`` (skipped for effective TierBridged — host net makes
iptables-in-container meaningless).

``ensure()`` is the single-tenant gate: discover a running container by the
project label and reuse it; create+start only when none is running (removing a
stopped stale namesake first). ``destroy()`` saves persisted state, stops
(10s grace), removes (force).
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import docker
from docker.errors import APIError, BuildError, ImageNotFound, NotFound

from pux_harness.kit._paths import project_root
from pux_harness.sandbox import host_setup, policy

log = logging.getLogger("pux.container")

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
DEFAULT_SANDBOX_ID = "mcp-default"
DEFAULT_NETWORK_ALLOW = "github.com,api.anthropic.com,api.openai.com,api.openrouter.com"

DEFAULT_MEMORY_MB = 2048
DEFAULT_CPU_CORES = 2.0
DEFAULT_PIDS = 512

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
    """
    p = os.environ.get("PUX_PROJECT_PATH") or str(project_root())
    if "://" in p:  # any URL scheme — ssh://, file://, http://, ...
        raise ContainerError(
            f"sandboxes require a local filesystem path; received a URL: {p!r}"
        )
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
        self.sandbox_id = _env_str("PUX_SANDBOX_ID", DEFAULT_SANDBOX_ID) if sandbox_id is None else sandbox_id
        self.org = org if org is not None else os.environ.get("PUX_ORG", "")
        self._client = client
        self._name: str | None = None

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
        try:
            c.remove(force=True)
        except APIError as exc:
            raise ContainerError(f"remove stale {self.name}: {exc}") from exc

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
        env = [
            f"SANDBOX_POLICY={DEFAULT_POLICY}",
            f"NETWORK_ALLOW={_env_str('PUX_NETWORK_ALLOW', DEFAULT_NETWORK_ALLOW)}",
            "FS_READONLY=/etc,/usr,/bin,/lib,/lib64",
            "FS_READWRITE=/sandbox/workspace,/sandbox/tmp",
            "DOCKER_HOST=unix:///var/run/docker.sock",
            "HOST_GATEWAY=host.docker.internal",
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
                log.info("host_setup exported %d env var(s): %s",
                         len(exports), sorted(exports))

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

        # Resource limits (env-overridable defaults).
        mem_bytes = _env_int("PUX_SANDBOX_MEMORY_MB", DEFAULT_MEMORY_MB) * 1024 * 1024
        nano_cpus = int(_env_float("PUX_SANDBOX_CPU_CORES", DEFAULT_CPU_CORES) * 1e9)
        pids = _env_int("PUX_SANDBOX_PIDS", DEFAULT_PIDS)

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
            log.info("staged egress allowlist (%d rules) + NET_ADMIN → %s",
                     len(pol.egress.allow), egress_path)

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
        )
        if user:
            create_kwargs["user"] = user
        if runtime:
            create_kwargs["runtime"] = runtime
        if cap_add:
            create_kwargs["cap_add"] = cap_add

        # Network: shared-infra for isolated; host net for bridged (skips ACLs).
        if effective_tier == "bridged":
            host_display = os.environ.get("DISPLAY", "")
            if host_display:
                create_kwargs.setdefault("environment", [])
                create_kwargs["environment"] = [*env, f"DISPLAY={host_display}"]
                create_kwargs["volumes"] = [*binds, "/tmp/.X11-unix:/tmp/.X11-unix"]
            create_kwargs["network_mode"] = "host"
        else:
            create_kwargs["network"] = _env_str("OPENSHELL_NETWORK", DEFAULT_NETWORK)

        self._remove_stale()
        log.info("creating container %s (image=%s tier=%s runtime=%s)",
                 self.name, image, effective_tier, runtime or "default")
        try:
            container = self.client.containers.create(**create_kwargs)
        except APIError as exc:
            # Name conflict on a stopped stale container we couldn't see → retry once.
            msg = str(exc)
            if "already in use" in msg or "Conflict" in msg:
                log.info("name conflict on %s — force-removing stale + retry", self.name)
                try:
                    self.client.containers.get(self.name).remove(force=True)
                except NotFound:
                    pass
                container = self.client.containers.create(**create_kwargs)
            else:
                raise ContainerError(f"create {self.name}: {exc}") from exc

        try:
            container.start()
        except APIError as exc:
            container.remove(force=True)
            raise ContainerError(f"start {self.name}: {exc}") from exc

        # Restore persisted state (Chrome profile, dotfiles, packages). Fire-and-
        # forget — failures don't block the sandbox (Go treats them the same way).
        self._restore_persisted(container)
        # Scaffold writable workspace dirs + chown to the host project owner so
        # host-side tools can read artifacts the agent writes.
        self._scaffold_workspace(container)

        log.info("sandbox ready: %s (container %s)", self.name, container.id[:12])
        self._name = self.name
        return self.name

    def _ensure_image(
        self, image: str, build: policy.BuildSpec | None = None
    ) -> None:
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
            log.info("building image %s (dockerfile=%s context=%s)",
                     image, build.dockerfile, context)
            try:
                self.client.images.build(
                    path=context, dockerfile=dockerfile_path.name, tag=image
                )
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
        """
        if self._name:
            return self._name
        running = self._running_for_project()
        if running:
            self._name = running
            log.info("reusing running container %s", running)
            return running
        return self.create()

    def destroy(self) -> None:
        """Stop + remove the sandbox container (save persisted state first).

        The save only runs when the container is still running — a stopped
        container can't be exec'd into (Go's savePersistedState ignores that
        exec error silently; we skip explicitly)."""
        name = self._name or self.name
        try:
            container = self.client.containers.get(name)
        except NotFound:
            self._name = None
            return
        if container.status == "running":
            self._save_persisted(container)
        try:
            container.stop(timeout=10)
        except APIError as exc:
            log.warning("stop %s failed (non-fatal): %s", name, exc)
        try:
            container.remove(force=True)
        except APIError as exc:
            raise ContainerError(f"remove {name}: {exc}") from exc
        log.info("sandbox destroyed: %s", name)
        self._name = None

    def _save_persisted(self, container: docker.models.containers.Container) -> None:
        # Verbatim port of manager.go::savePersistedState.
        self._exec(container, _SAVE_SCRIPT)


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
echo "Saved $(wc -l < /sandbox/persist/installed-packages.txt) packages list"""


# --- prepare (post-create, pre-agent jobs) ----------------------------------


def prepare(
    org: str,
    project_path: str | None = None,
    exec_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Run post-create, pre-agent prep jobs inside the sandbox container.

    Called by entry points (main.py, server.py) after ``ensure()``, before
    ``graph.ainvoke()``. Returns a list of result dicts for logging/display.

    Idempotency is delegated to the scripts themselves (file caches +
    SurrealDB UPSERTs make repeat runs cheap).

    Uses lazy imports to avoid circular dependency with docker_exec.
    """
    from pux_harness.sandbox.docker_exec import DockerExecClient  # noqa: PLC0415
    from pux_harness.sandbox.jobs import run_jobs  # noqa: PLC0415

    if not project_path:
        project_path = resolve_project_path()

    try:
        pol = policy.load(org, project_path)
    except policy.NoPolicy:
        return []

    specs = policy.job_specs(pol)
    if not specs:
        return []

    if exec_client is None:
        sb = SandboxContainer(project_path=project_path, org=org)
        container_name = sb.ensure()
        exec_client = DockerExecClient(container=container_name)

    results = run_jobs(pol, exec_client)
    return [
        {"name": r.name, "status": r.status, "error": r.error,
         "duration": round(r.duration, 1)}
        for r in results
    ]
