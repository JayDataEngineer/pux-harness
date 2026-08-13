"""The ``pux_sandbox_*`` specialist tool package.

Every tool is a ``StructuredTool`` named ``pux_sandbox_<name>`` (graders:
``pux_grader_<name>``). This package replaces the monolithic ``tools.py`` —
each tool group lives in its own module:

- ``python.py``        — execute Python code in the sandbox
- ``skills.py``        — list / load org skills (host FS, no exec)
- ``describe_image.py`` — multimodal-model-primary + ONNX fallback
- ``multimodal.py``    — ``multimodal`` + ``multimodal_mega`` (media + prompt)
- ``browser.py``       — all ``pux_sandbox_browser_*`` tools (SeleniumBase Chrome)
- ``desktop.py``       — all ``pux_sandbox_desktop_*`` tools (xdotool + Xvfb)
- ``grader.py``        — ``pux_grader_*`` evidence-gathering factory functions
- ``registry.py``      — the single ``REGISTRY`` of ``ToolSpec`` + derived
  surface sets + ``make_specialist_tools`` / ``build_grader_tools``.

Two "private" modules provide shared infrastructure:
- ``_shared.py`` — ZERO-pux-import generic helpers (``_exec``, ``_tail``, ``_result``,
  ``_NoArgs``). Portable alongside the specialist tools.
- ``_pux.py`` — pux-specific constants (``PUX_PREFIX``, ``PUX_GRADER_PREFIX``,
  ``SKILL_FILE``, ``_skills_dirs``) for the org-coupled tools.
- ``_media.py`` — shared media-acquisition + ONNX-inference plumbing (consumed
  by ``describe_image`` and ``multimodal``).

Public API
----------
``make_specialist_tools`` — the main function ``graph.py`` calls.
``build_grader_tools`` — the three ``pux_grader_*`` evidence tools.
``REGISTRY`` / ``ToolSpec`` / ``Category`` / ``Requirements`` — the declarative
tool surface (single source of truth; everything below derives from it).
``SPECIALIST_TOOL_NAMES`` / ``SPECIALISTS`` — derived name whitelist the org
contract resolves agent ``tools:`` lists against.
``NATIVE_FS_TOOLS`` — the fs/shell surface ``FilesystemMiddleware`` injects
(declared in ``REGISTRY`` as ``factory=None`` natives).
``GRADER_TOOL_NAMES`` — the grader evidence-tool names.
``LEGACY_TOOL_NAMES`` — forbidden legacy names (permanent tripwire denylist).
``classify_slug`` / ``prefixed`` — shared classifier + prefixer so the contract
and runtime resolver can never drift.
"""

from pux_harness.sandbox.tools.registry import (
    Category,
    REGISTRY,
    Requirements,
    SPECIALIST_GROUPS,
    TOOL_GROUP_TOOLS,
    ToolSpec,
    build_grader_tools,
    make_specialist_tools,
    classify_slug,
    prefixed,
    resolve_tool_allowlist,
)
from pux_harness.sandbox.tools.registry import (
    GRADER_TOOL_NAMES,
    LEGACY_TOOL_NAMES,
    NATIVE_FS_TOOLS,
    SPECIALIST_TOOL_NAMES,
    SPECIALISTS,
)

__all__ = [
    "Category",
    "GRADER_TOOL_NAMES",
    "LEGACY_TOOL_NAMES",
    "NATIVE_FS_TOOLS",
    "REGISTRY",
    "Requirements",
    "SPECIALIST_GROUPS",
    "SPECIALIST_TOOL_NAMES",
    "SPECIALISTS",
    "TOOL_GROUP_TOOLS",
    "ToolSpec",
    "build_grader_tools",
    "make_specialist_tools",
    "classify_slug",
    "prefixed",
    "resolve_tool_allowlist",
]
