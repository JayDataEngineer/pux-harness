"""PuxSandboxProvider — dcode ``SandboxProvider`` bridge over pux's Docker sandbox.

dcode's sandbox registry instantiates this class from user config::

    # ~/.deepagents/config.toml
    [sandboxes.providers.pux]
    class_path = "pux_harness.sandbox.provider:PuxSandboxProvider"
    working_dir = "/sandbox/workspace"

…and calls ``get_or_create()`` / ``delete()`` via the unified
``deepagents_code.integrations.sandbox_factory.create_sandbox`` entry point.

The bridge wires two existing pux pieces together:

* :class:`pux_harness.sandbox.container.SandboxContainer` — the lifecycle owner
  (create / ensure / destroy the one persistent Docker container).
* :class:`pux_harness.sandbox.backend.PuxSandboxBackend` — a
  :class:`deepagents.backends.sandbox.BaseSandbox` whose four abstract
  primitives (``execute`` / ``id`` / ``upload_files`` / ``download_files``)
  run over :class:`pux_harness.sandbox.docker_exec.DockerExecClient`.

So this module adds NO new sandbox semantics — only the thin adapter that makes
pux's container loadable as ``dcode --sandbox pux``.

Why ``delete()`` is a no-op
---------------------------
dcode's ``create_sandbox()`` is a context manager that calls
``provider.delete(sandbox_id=backend.id)`` on exit for any fresh sandbox. That is
correct for dcode's built-in *ephemeral* providers (a throwaway cloud VM per
invocation). pux's container is the opposite — **single-tenant and persistent**,
reused across sessions (the workspace, the warmed browser, installed deps all
survive). Destroying on every agent run would defeat the entire model.

Teardown therefore stays explicit: ``pux sandbox destroy`` (or
``SandboxContainer.destroy()`` directly). This is a legitimate provider design —
the contract permits it; ``delete()`` simply declines to tear down a shared
resource it does not uniquely own.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deepagents_code.integrations.sandbox_provider import (
    SandboxProvider,
    SandboxProviderMetadata,
)

from pux_harness.sandbox.backend import PuxSandboxBackend
from pux_harness.sandbox.container import SandboxContainer
from pux_harness.sandbox.docker_exec import DockerExecClient

if TYPE_CHECKING:
    from deepagents.backends.protocol import SandboxBackendProtocol

logger = logging.getLogger(__name__)

__all__ = ["PuxSandboxProvider"]

#: Working directory inside the pux container — the project bind-mount and the
#: image ``WORKDIR``. dcode surfaces this to the model as the sandbox root.
WORKING_DIR = "/sandbox/workspace"


class PuxSandboxProvider(SandboxProvider):
    """dcode provider that drives pux's persistent Docker sandbox.

    ``get_or_create`` reuses a running container (booting one via
    :meth:`SandboxContainer.ensure` if none exists) and returns a
    :class:`PuxSandboxBackend` over it. ``delete`` is a no-op — see the module
    docstring for the persistence rationale.
    """

    @property
    def metadata(self) -> SandboxProviderMetadata:
        """Static description used by the registry without instantiating deps.

        ``backend_module`` is probe-imported by dcode's pre-flight dependency
        check; pointing it at the backend ensures a clear error if the pux
        package (or its ``docker`` SDK) is missing from the launching venv.
        """
        return SandboxProviderMetadata(
            name="pux",
            working_dir=WORKING_DIR,
            supports_sandbox_id=True,
            supports_snapshot_name=False,
            backend_module="pux_harness.sandbox.backend",
        )

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Return a backend over the (reused or freshly booted) pux container.

        Args:
            sandbox_id: Optional existing container id to reattach to. When
                ``None`` (the dcode default), pux derives a stable id from the
                project path — so repeated launches against the same project
                land on the *same* persistent container.
            **kwargs: Forwarded ``params`` from
                ``[sandboxes.providers.pux.params]``. Recognized keys:
                ``org`` (the pux org name, used to resolve sandbox policy;
                falls back to ``$PUX_ORG``).

        Returns:
            A :class:`PuxSandboxBackend` wired to the running container.

        Raises:
            pux_harness.sandbox.container.ContainerError: If the container
                cannot be created or started.
        """
        org = kwargs.get("org")
        container = SandboxContainer(sandbox_id=sandbox_id, org=org)
        name = container.ensure()
        exec_client = DockerExecClient(container=name, boot=False)
        backend = PuxSandboxBackend(exec_client)
        # Pin ``id`` to the container name so dcode's factory cleanup +
        # ``delete(sandbox_id=backend.id)`` address a real handle, and so we
        # avoid the lazy ``cat /etc/hostname`` exec the base property would
        # otherwise run on first read.
        backend._id = name
        logger.info("pux provider: backend ready on container %s", name)
        return backend

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        """Decline to destroy the container — it is shared and persistent.

        See the module docstring. Real teardown is ``pux sandbox destroy``.
        """
        logger.info(
            "pux provider: delete(%s) is a no-op — the container is persistent "
            "and reused across sessions (teardown: `pux sandbox destroy`)",
            sandbox_id,
        )
