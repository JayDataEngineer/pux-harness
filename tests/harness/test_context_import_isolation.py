"""Source-level tripwire: the ``context/`` cluster is upstream-portable.

Every module under ``pux_harness/context/`` must import ONLY from the standard
library, third-party packages (langchain/langgraph/deepagents/anthropic), or
its own cluster (``pux_harness.context.*``). No ``kit`` / ``agent`` / ``sandbox``
imports — those couple the cluster to the pux application layer and block
extraction into a standalone ``deepagents-context`` package.

The ONE exception is ``exec_tools.py`` (imports ``sandbox.docker_exec`` for
``ExecTimeout``); it is pending extraction to the pux side. When it moves, the
allowlist shrinks to empty and the cluster is fully pure.

Sibling of ``tests/kit/test_kit_import_isolation.py``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_CONTEXT_DIR = (
    Path(__file__).resolve().parents[2] / "pux_harness" / "context"
)

# pux-application-layer modules the context cluster must not reach.
_FORBIDDEN_PREFIXES = ("pux_harness.kit", "pux_harness.agent", "pux_harness.sandbox")

# Files still carrying a pux-application import (pending extraction).
_ALLOWLIST: dict[str, set[str]] = {
    "exec_tools.py": {"pux_harness.sandbox.docker_exec"},
}


def _pux_imports(tree: ast.AST) -> set[str]:
    """All ``pux_harness.*`` module paths imported by this module."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("pux_harness"):
            out.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("pux_harness"):
                    out.add(alias.name)
    return out


@pytest.mark.parametrize("path", sorted(_CONTEXT_DIR.glob("*.py")), ids=lambda p: p.name)
def test_context_cluster_import_pure(path: Path) -> None:
    """No ``context/*.py`` may import ``kit`` / ``agent`` / ``sandbox``."""
    if path.name == "__init__.py":
        return
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = _pux_imports(tree)
    forbidden = {
        mod for mod in imports
        if any(mod.startswith(pfx) for pfx in _FORBIDDEN_PREFIXES)
    }
    allowed = _ALLOWLIST.get(path.name, set())
    violators = forbidden - allowed
    assert not violators, (
        f"{path.name} imports pux-application modules {sorted(violators)} — "
        f"the context cluster must stay upstream-portable. Move the dependency "
        f"to the caller or inject it through the constructor."
    )
