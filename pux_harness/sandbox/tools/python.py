"""pux_sandbox_python — execute Python code inside the sandbox."""

from __future__ import annotations

from deepagents.backends.sandbox import BaseSandbox

import shlex

from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

from ._shared import _result, _exec


class _PythonArgs(BaseModel):
    code: str = Field(..., description="Python code to execute. Print output is captured and returned.")


_PYTHON_DESC = (
    "Execute Python code inside the sandbox. Print output is captured. "
    "Whatever the sandbox image ships with is available. Runs via docker exec "
    "(python3 -c)."
)


def _python_tool(sandbox: BaseSandbox) -> StructuredTool:
    def _run(code: str) -> str:
        if not code:
            return _result({"success": False, "error": "no code provided"})
        out, exit_code = _exec(sandbox, f"python3 -c {shlex.quote(code)}")
        if exit_code != 0:
            return _result({"success": False, "error": f"python exited {exit_code}", "output": out})
        return _result({"success": True, "output": out})

    return StructuredTool(
        name="python", description=_PYTHON_DESC,
        args_schema=_PythonArgs, func=_run,
    )
