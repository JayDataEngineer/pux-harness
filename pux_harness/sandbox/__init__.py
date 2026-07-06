"""Docker sandbox layer: the container lifecycle + policy enforcement
(``container``, ``policy``), the exec client + backend (``docker_exec``,
``backend``), and every native specialist tool (``tools/`` package).

Self-contained — imports nothing from ``agent`` or ``context``. The agent
layer wires these into the deepagents graph."""
