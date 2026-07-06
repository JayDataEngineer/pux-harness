"""pux_grader_* — RubricMiddleware sandbox-bound evidence-gathering tools."""

from __future__ import annotations

import shlex

from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

from pux_harness.sandbox.docker_exec import DockerExecClient, ExecTimeout
from pux_harness.sandbox.tools._shared import PUX_GRADER_PREFIX, _tail, _result


class _GraderExecuteArgs(BaseModel):
    command: str = Field(
        ..., description="Shell command to run inside the sandbox (tests, lint, "
        "typecheck, build). Run from /sandbox/workspace. Cite the exit code in "
        "your verdict."
    )


_GRADER_EXECUTE_DESC = (
    "Run a shell command inside the sandbox container to gather EVIDENCE for a "
    "rubric verdict — run the test suite, lint, typecheck, or build, then read "
    "the exit code + output. The workspace is at /sandbox/workspace. Do not "
    "grade from the agent's summary — run the real check and cite what it said."
)


def _grader_execute_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(command: str) -> str:
        if not command:
            return _result({"success": False, "error": "no command provided"})
        out, exit_code = exec_client.exec(command)
        return _result({
            "success": True, "exit_code": exit_code, "output": out,
        })

    return StructuredTool(
        name=PUX_GRADER_PREFIX + "execute", description=_GRADER_EXECUTE_DESC,
        args_schema=_GraderExecuteArgs, func=_run,
    )


class _GraderReadFileArgs(BaseModel):
    path: str = Field(
        ..., description="Path to a file inside the sandbox (read the diff, "
        "inspect touched source). Project-relative paths resolve under "
        "/sandbox/workspace."
    )


_GRADER_READ_FILE_DESC = (
    "Read a file's contents inside the sandbox to gather EVIDENCE for a rubric "
    "verdict — inspect the changed files, read the diff, confirm the "
    "implementation exists and reads like the surrounding code. Do not take the "
    "agent's word that a file was changed — read it."
)


def _grader_read_file_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(path: str) -> str:
        if not path:
            return _result({"success": False, "error": "no path provided"})
        out, exit_code = exec_client.exec(f"cat {shlex.quote(path)}")
        if exit_code != 0:
            return _result({
                "success": False, "error": f"cat exited {exit_code}", "output": out,
            })
        return _result({"success": True, "path": path, "content": out})

    return StructuredTool(
        name=PUX_GRADER_PREFIX + "read_file", description=_GRADER_READ_FILE_DESC,
        args_schema=_GraderReadFileArgs, func=_run,
    )


class _GraderGrepArgs(BaseModel):
    pattern: str = Field(..., description="Regex or literal to search for.")
    path: str = Field(
        "/sandbox/workspace", description="File or directory to search "
        "(default: the workspace root)."
    )
    include: str | None = Field(
        None, description="Optional glob filter, e.g. '*.py' or '*.go'."
    )


_GRADER_GREP_DESC = (
    "Search file contents inside the sandbox to gather EVIDENCE for a rubric "
    "verdict — locate a symbol, check a regression marker didn't reappear, "
    "confirm a removed API has no remaining callers. Recursive by default."
)


def _grader_grep_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(pattern: str, path: str = "/sandbox/workspace",
             include: str | None = None) -> str:
        if not pattern:
            return _result({"success": False, "error": "no pattern provided"})
        cmd = f"grep -rn {shlex.quote(pattern)} {shlex.quote(path)}"
        if include:
            cmd += f" --include={shlex.quote(include)}"
        out, exit_code = exec_client.exec(cmd)
        if exit_code == 2:
            return _result({"success": False, "error": f"grep error: {out or 'bad pattern/path'}"})
        return _result({
            "success": True,
            "matches": out if exit_code == 0 else "",
            "match_count": out.count("\n") + 1 if (exit_code == 0 and out) else 0,
        })

    return StructuredTool(
        name=PUX_GRADER_PREFIX + "grep", description=_GRADER_GREP_DESC,
        args_schema=_GraderGrepArgs, func=_run,
    )


# ``build_grader_tools`` lives in ``registry.py`` now — it is a thin category
# filter over ``build_tools`` there. This module stays a leaf (factories only,
# no import of the registry) so the registry can import these factories
# one-way without a cycle.
