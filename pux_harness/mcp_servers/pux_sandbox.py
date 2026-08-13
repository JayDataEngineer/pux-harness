"""pux-sandbox MCP server — specialist tools as MCP (Phase 1 simplification).

The engine's 44 ``pux_sandbox_*`` specialist tools (python, describe_image,
multimodal, desktop, browser, etc.) are served via MCP stdio. Each agent
process that needs file/shell/code/browser tools spawns this server as a
child subprocess. The server connects to the shared sandbox backend
(``shared_exec`` / ``shared_backend`` — OpenShell by default) and registers
every specialist tool with FastMCP.

WHY: Phase 1 of the engine simplification plan. Tools move OUT of the engine
(stack.py / registry.py / ToolDeps / Requirements) into a standalone MCP
server. The engine becomes a thin compiler; orgs declare tools via
``{kind: mcp, ref: pux_sandbox}`` in org.yaml. The tool code itself is
UNCHANGED — ``build_native_specialists`` builds the same StructuredTool
instances; this server wraps each as an MCP tool.

Architecture (Pattern 2 — one MCP server per agent process):
    agent framework → stdio → [this server] → shared BaseSandbox → OpenShell

The server owns ONE ``ExecClient`` for its lifetime. When the agent
disconnects (stdin EOF), the server exits and the backend connection closes.

Env:
    PUX_ORG  — the org name (scopes skills tools)

Logging: stdout is RESERVED for MCP JSON-RPC. All logs go to stderr.
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from pux_harness.sandbox.exec import shared_backend, shared_exec
from pux_harness.sandbox.tools import make_specialist_tools

# ── logging (stderr only) ────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stderr,
    level=os.environ.get("PUX_MCP_LOG_LEVEL", "INFO"),
    format="[pux-sandbox-mcp] %(levelname)s %(message)s",
)
log = logging.getLogger("pux-sandbox-mcp")


# ── FastMCP server ───────────────────────────────────────────────────────────

mcp = FastMCP("pux-sandbox")


# ── StructuredTool → MCP function adapter ────────────────────────────────────

# JSON schema type → Python type mapping for MCP parameter annotations.
_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _wrap_structured_tool(tool: Any) -> Any:
    """Create a plain async function wrapping a langchain StructuredTool.

    FastMCP's ``add_tool`` accepts a function and derives the JSON schema from
    its type-annotated signature. We build that signature dynamically from the
    StructuredTool's ``args_schema`` (a Pydantic model) so the MCP wire schema
    matches exactly what the agent already knows.
    """
    name = tool.name
    description = tool.description or name
    prefix = "pux_sandbox_"
    short_name = name[len(prefix):] if name.startswith(prefix) else name

    # Extract the argument schema.
    schema = tool.args_schema.model_json_schema()
    properties: dict[str, dict] = schema.get("properties", {})
    required: set[str] = set(schema.get("required", []))

    # Build function parameters.
    params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        ptype = prop_schema.get("type", "string")
        py_type = _JSON_TYPE_MAP.get(ptype, str)
        annotations[prop_name] = py_type
        has_default = "default" in prop_schema
        default = prop_schema.get("default") if has_default else (
            inspect.Parameter.empty if prop_name in required else None
        )
        params.append(
            inspect.Parameter(
                prop_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=py_type,
            )
        )

    # The wrapper function — calls the StructuredTool asynchronously.
    async def _wrapper(**kwargs: Any) -> str:
        result = await tool.ainvoke(kwargs)
        # MCP tools return text content; stringify structured results.
        if isinstance(result, str):
            return result
        import json
        return json.dumps(result, default=str, indent=2)

    # Apply metadata so FastMCP generates the correct schema.
    _wrapper.__name__ = short_name.replace("-", "_")
    _wrapper.__doc__ = description
    _wrapper.__annotations__ = {"return": str, **annotations}
    _wrapper.__signature__ = inspect.Signature(parameters=params)  # type: ignore[attr-defined]

    return _wrapper


# ── bootstrap ────────────────────────────────────────────────────────────────

def _register_all() -> int:
    """Build every specialist StructuredTool + register each with FastMCP."""
    backend = shared_backend()
    org = os.environ.get("PUX_ORG") or None

    tools = make_specialist_tools(
        backend,
        vision_model=None,
        org=org,
    )
    log.info("built %d specialist tools from REGISTRY", len(tools))

    for tool in tools:
        fn = _wrap_structured_tool(tool)
        mcp.add_tool(fn, name=tool.name, description=tool.description)
        log.debug("registered: %s", tool.name)

    return len(tools)


_tool_count = _register_all()
log.info("pux-sandbox MCP server ready — %d tools", _tool_count)


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
