"""cli.main() bootstrap wiring — the universal seam for serve/direct.

**NOT YET COMMITTED** — rides with the ``cli.py`` change (``main()`` calls
``bootstrap_env_and_logging``) so the two land in the same commit. The shared
helper itself + its unit tests are committed in ``test_bootstrap.py``; this file
is the integration proof that the REAL ``pux`` console-script path loads
``./.env`` from a foreign CWD. Once ``cli.py``'s bootstrap line is committed
(with the exportable session's ``--project-root`` coordination), this file
commits alongside it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def test_cli_main_loads_dotenv_from_foreign_cwd(tmp_path: Path, monkeypatch) -> None:
    # End-to-end: the REAL cli.main() (the ``pux`` console script) loads
    # ``./.env`` from the launch CWD before argparse runs. The universal seam
    # that wires serve/direct/acp seamlessly in a FOREIGN codebase — the
    # consumer's .env is picked up with no ``export``. ``--help`` exits during
    # parse_args, but bootstrap has already run by then.
    import pytest

    from pux_harness import cli

    monkeypatch.delenv("PUX_E2E_SENTINEL", raising=False)
    (tmp_path / ".env").write_text("PUX_E2E_SENTINEL=loaded-by-cli\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["pux", "--help"])

    with pytest.raises(SystemExit):
        cli.main()
    assert os.environ["PUX_E2E_SENTINEL"] == "loaded-by-cli"
