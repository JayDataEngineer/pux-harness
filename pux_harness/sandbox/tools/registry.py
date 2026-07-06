"""The single source of truth for the harness's action-tool surface.

Every ``pux_sandbox_*`` specialist, every ``pux_grader_*`` evidence tool, and
every native fs/shell tool the deepagents ``FilesystemMiddleware`` injects is
declared ONCE here as a ``ToolSpec`` in ``REGISTRY``. Everything else — the
specialist/native/grader name sets the contract validates against, the prefix
helpers the runtime resolver uses, and the builder functions ``graph.py``
calls — DERIVES from that list. There is no second hand-maintained copy
anywhere: adding a tool means adding one ``ToolSpec`` line, and every
frozenset + resolver + builder updates automatically.

Why a declarative list and not decorator auto-discovery: the whole surface is
visible at a glance, there are no import-time side effects, and a stale
reference fails loud (the contract and the runtime resolver share
``classify_slug`` / ``prefixed``, so they can no longer drift — see
``agent/contract.py`` rule 4 and ``agent/orgs.py`` ``_resolve_tools``).

Per-tool ``Requirements`` (which Docker client / backend / vision model / org
scope a factory needs, and which sandbox capabilities — ffmpeg, xdotool — it
expects) are DECLARED here but NOT acted on: a tool whose capability is
unsatisfied is still registered and reports an honest error at call time (the
Phase-18 multimodal philosophy — present + honest, never a silent drop).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from pux_harness.sandbox.docker_exec import DockerExecClient
from pux_harness.sandbox.backend import PuxSandboxBackend
from pux_harness.sandbox.tools._shared import PUX_PREFIX, PUX_GRADER_PREFIX
from pux_harness.sandbox.tools.python import _python_tool
from pux_harness.sandbox.tools.skills import _list_skills_tool, _load_skill_tool
from pux_harness.sandbox.tools.describe_image import _describe_image_tool
from pux_harness.sandbox.tools.multimodal import _multimodal_tool, _multimodal_mega_tool
from pux_harness.sandbox.tools.browser import (
    _browser_navigate_tool,
    _browser_click_tool,
    _browser_type_tool,
    _browser_screenshot_tool,
    _browser_evaluate_tool,
    _browser_search_tool,
    _browser_scroll_tool,
    _browser_go_back_tool,
    _browser_wait_tool,
    _browser_find_text_tool,
    _browser_extract_tool,
    _browser_extract_images_tool,
    _browser_save_screenshot_tool,
    _browser_download_tool,
    _browser_upload_tool,
    _browser_tabs_tool,
    _browser_new_tab_tool,
    _browser_switch_tab_tool,
    _browser_close_tab_tool,
    _browser_dropdown_options_tool,
    _browser_select_dropdown_tool,
    _browser_save_session_tool,
    _browser_restore_session_tool,
    _browser_drag_tool,
    _browser_hover_tool,
    _browser_press_tool,
    _browser_click_at_tool,
    _browser_scroll_into_view_tool,
    _browser_a11y_tool,
    _browser_iframe_tool,
)
from pux_harness.sandbox.tools.desktop import (
    _desktop_screenshot_tool,
    _desktop_click_tool,
    _desktop_type_tool,
    _desktop_key_tool,
)
from pux_harness.sandbox.tools.grader import (
    _grader_execute_tool,
    _grader_read_file_tool,
    _grader_grep_tool,
)


# --- the vocabulary -------------------------------------------------------

class Category(Enum):
    """Which surface a tool belongs to.

    Plain ``Enum`` (not ``StrEnum``) so a category never compares equal to a
    bare slug string — ``Category.NATIVE != "native"``."""

    NATIVE = "native"        # injected by FilesystemMiddleware; factory is None
    SPECIALIST = "specialist"  # pux_sandbox_* — agent-whitelistable
    GRADER = "grader"        # pux_grader_* — RubricMiddleware evidence tools


@dataclass(frozen=True)
class Requirements:
    """What a tool's factory needs to bind, and what it expects at call time.

    The first four fields select which ``ToolDeps`` members ``build_tools``
    threads into the factory as keyword args (the factory param names match
    these exactly: ``exec_client`` / ``backend`` / ``vision_model`` / ``org``).
    ``caps`` is DECLARE-ONLY metadata — sandbox binaries (ffmpeg, xdotool) the
    tool expects to find at call time; never gated on, surfaced for grep + a
    future informational contract warning."""

    exec_client: bool = False
    backend: bool = False
    vision: bool = False
    org: bool = False
    caps: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        # Accept a plain set literal (``caps={"ffmpeg"}``) and coerce — frozen
        # dataclass, so assign through object.__setattr__.
        if not isinstance(self.caps, frozenset):
            object.__setattr__(self, "caps", frozenset(self.caps))


@dataclass(frozen=True)
class ToolSpec:
    """One tool, declared once. The single unit of the tool surface."""

    slug: str
    category: Category
    needs: Requirements
    factory: Callable[..., StructuredTool] | None  # None for NATIVE (middleware)


@dataclass(frozen=True)
class ToolDeps:
    """The dependency bundle ``build_tools`` fans out to factories by need."""

    exec_client: Any = None
    backend: Any = None
    vision_model: Any = None
    org: str | None = None


# --- THE registry ---------------------------------------------------------
# Order is preserved by build_tools → matches the historical build order.

REGISTRY: list[ToolSpec] = [
    # python + skills
    ToolSpec("python", Category.SPECIALIST, Requirements(exec_client=True), _python_tool),
    ToolSpec("list_skills", Category.SPECIALIST, Requirements(org=True), _list_skills_tool),
    ToolSpec("load_skill", Category.SPECIALIST, Requirements(org=True), _load_skill_tool),

    # media (model-primary, declare their caps)
    ToolSpec("describe_image", Category.SPECIALIST,
             Requirements(backend=True, exec_client=True, vision=True, caps={"onnx"}),
             _describe_image_tool),
    ToolSpec("multimodal", Category.SPECIALIST,
             Requirements(backend=True, exec_client=True, vision=True),
             _multimodal_tool),
    ToolSpec("multimodal_mega", Category.SPECIALIST,
             Requirements(backend=True, exec_client=True, vision=True, caps={"ffmpeg"}),
             _multimodal_mega_tool),

    # browser (SeleniumBase Chrome via sb_server)
    ToolSpec("browser_navigate", Category.SPECIALIST, Requirements(exec_client=True), _browser_navigate_tool),
    ToolSpec("browser_click", Category.SPECIALIST, Requirements(exec_client=True), _browser_click_tool),
    ToolSpec("browser_type", Category.SPECIALIST, Requirements(exec_client=True), _browser_type_tool),
    ToolSpec("browser_screenshot", Category.SPECIALIST, Requirements(exec_client=True), _browser_screenshot_tool),
    ToolSpec("browser_evaluate", Category.SPECIALIST, Requirements(exec_client=True), _browser_evaluate_tool),
    ToolSpec("browser_search", Category.SPECIALIST, Requirements(exec_client=True), _browser_search_tool),
    ToolSpec("browser_scroll", Category.SPECIALIST, Requirements(exec_client=True), _browser_scroll_tool),
    ToolSpec("browser_go_back", Category.SPECIALIST, Requirements(exec_client=True), _browser_go_back_tool),
    ToolSpec("browser_wait", Category.SPECIALIST, Requirements(exec_client=True), _browser_wait_tool),
    ToolSpec("browser_find_text", Category.SPECIALIST, Requirements(exec_client=True), _browser_find_text_tool),
    ToolSpec("browser_extract", Category.SPECIALIST, Requirements(exec_client=True), _browser_extract_tool),
    ToolSpec("browser_extract_images", Category.SPECIALIST, Requirements(exec_client=True), _browser_extract_images_tool),
    ToolSpec("browser_save_screenshot", Category.SPECIALIST, Requirements(exec_client=True), _browser_save_screenshot_tool),
    ToolSpec("browser_download", Category.SPECIALIST, Requirements(exec_client=True), _browser_download_tool),
    ToolSpec("browser_upload", Category.SPECIALIST, Requirements(exec_client=True), _browser_upload_tool),
    ToolSpec("browser_tabs", Category.SPECIALIST, Requirements(exec_client=True), _browser_tabs_tool),
    ToolSpec("browser_new_tab", Category.SPECIALIST, Requirements(exec_client=True), _browser_new_tab_tool),
    ToolSpec("browser_switch_tab", Category.SPECIALIST, Requirements(exec_client=True), _browser_switch_tab_tool),
    ToolSpec("browser_close_tab", Category.SPECIALIST, Requirements(exec_client=True), _browser_close_tab_tool),
    ToolSpec("browser_dropdown_options", Category.SPECIALIST, Requirements(exec_client=True), _browser_dropdown_options_tool),
    ToolSpec("browser_select_dropdown", Category.SPECIALIST, Requirements(exec_client=True), _browser_select_dropdown_tool),
    ToolSpec("browser_save_session", Category.SPECIALIST, Requirements(exec_client=True), _browser_save_session_tool),
    ToolSpec("browser_restore_session", Category.SPECIALIST, Requirements(exec_client=True), _browser_restore_session_tool),
    # Phase 19 — SOTA mouse/keyboard/DnD (still pure-CDP SeleniumBase, no new deps)
    ToolSpec("browser_drag", Category.SPECIALIST, Requirements(exec_client=True), _browser_drag_tool),
    ToolSpec("browser_hover", Category.SPECIALIST, Requirements(exec_client=True), _browser_hover_tool),
    ToolSpec("browser_press", Category.SPECIALIST, Requirements(exec_client=True), _browser_press_tool),
    ToolSpec("browser_click_at", Category.SPECIALIST, Requirements(exec_client=True), _browser_click_at_tool),
    ToolSpec("browser_scroll_into_view", Category.SPECIALIST, Requirements(exec_client=True), _browser_scroll_into_view_tool),
    ToolSpec("browser_a11y", Category.SPECIALIST, Requirements(exec_client=True), _browser_a11y_tool),
    ToolSpec("browser_iframe", Category.SPECIALIST, Requirements(exec_client=True), _browser_iframe_tool),

    # desktop (xdotool + Xvfb)
    ToolSpec("desktop_screenshot", Category.SPECIALIST,
             Requirements(exec_client=True, caps={"xdotool", "x11"}), _desktop_screenshot_tool),
    ToolSpec("desktop_click", Category.SPECIALIST,
             Requirements(exec_client=True, caps={"xdotool", "x11"}), _desktop_click_tool),
    ToolSpec("desktop_type", Category.SPECIALIST,
             Requirements(exec_client=True, caps={"xdotool", "x11"}), _desktop_type_tool),
    ToolSpec("desktop_key", Category.SPECIALIST,
             Requirements(exec_client=True, caps={"xdotool", "x11"}), _desktop_key_tool),

    # native fs/shell — injected by FilesystemMiddleware, no factory here.
    ToolSpec("ls", Category.NATIVE, Requirements(), None),
    ToolSpec("read_file", Category.NATIVE, Requirements(), None),
    ToolSpec("write_file", Category.NATIVE, Requirements(), None),
    ToolSpec("edit_file", Category.NATIVE, Requirements(), None),
    ToolSpec("glob", Category.NATIVE, Requirements(), None),
    ToolSpec("grep", Category.NATIVE, Requirements(), None),
    ToolSpec("execute", Category.NATIVE, Requirements(), None),

    # grader — RubricMiddleware evidence tools (own prefix). NOTE: the bare
    # slugs ``execute``/``read_file``/``grep`` collide with native slugs above;
    # classify_slug resolves NATIVE first, which is correct — graders never
    # appear in an agent whitelist. GRADERS/GRADER_TOOL_NAMES stay well-defined
    # because they filter by category before prefixing.
    ToolSpec("execute", Category.GRADER, Requirements(exec_client=True), _grader_execute_tool),
    ToolSpec("read_file", Category.GRADER, Requirements(exec_client=True), _grader_read_file_tool),
    ToolSpec("grep", Category.GRADER, Requirements(exec_client=True), _grader_grep_tool),
]


# --- everything below DERIVES from REGISTRY — no hand-maintained copies ----

SPECIALISTS: frozenset[str] = frozenset(
    s.slug for s in REGISTRY if s.category is Category.SPECIALIST
)
SPECIALIST_TOOL_NAMES: frozenset[str] = frozenset(
    PUX_PREFIX + s for s in SPECIALISTS
)
NATIVE_FS_TOOLS: frozenset[str] = frozenset(
    s.slug for s in REGISTRY if s.category is Category.NATIVE
)
GRADERS: frozenset[str] = frozenset(
    s.slug for s in REGISTRY if s.category is Category.GRADER
)
GRADER_TOOL_NAMES: frozenset[str] = frozenset(
    PUX_GRADER_PREFIX + s for s in GRADERS
)


# Forbidden legacy tool names — the frozen bash/file pux_sandbox_* surface that
# the native flip replaced. A DENYLIST (not derived from REGISTRY — these names
# are deliberately absent). Permanent tripwire per the no-legacy-left-behind
# rule: co-located + named here so the dev-bot forcing surface check
# (``main.py``) imports one constant instead of re-declaring the literal.
LEGACY_TOOL_NAMES: frozenset[str] = frozenset({
    "pux_sandbox_bash", "pux_sandbox_file_read", "pux_sandbox_file_write",
    "pux_sandbox_file_edit", "pux_sandbox_file_glob", "pux_sandbox_file_grep",
})


_PREFIX_BY_CATEGORY: dict[Category, str] = {
    Category.SPECIALIST: PUX_PREFIX,
    Category.GRADER: PUX_GRADER_PREFIX,
    # NATIVE: no prefix (bare slug, middleware-provided).
}


def prefixed(slug: str, category: Category) -> str:
    """The fully-qualified tool name for ``slug`` in ``category``. One place
    that knows the prefix → category mapping; both the contract and the runtime
    resolver call this, so the prefix can never drift between them."""
    return _PREFIX_BY_CATEGORY.get(category, "") + slug


def classify_slug(slug: str) -> Category | None:
    """Which surface a bare agent-whitelist slug belongs to, or ``None`` if it
    resolves to nothing. Shared by the offline contract (rule 4) and the
    runtime ``_resolve_tools`` — the single classification so the two paths can
    no longer disagree (the old contract permitted a native slug in a whitelist
    while the runtime raised KeyError on it)."""
    if slug in NATIVE_FS_TOOLS:
        return Category.NATIVE
    if slug in SPECIALISTS:
        return Category.SPECIALIST
    if slug in GRADERS:
        return Category.GRADER
    return None


def build_tools(deps: ToolDeps, category: Category) -> list[StructuredTool]:
    """Instantiate every ``REGISTRY`` tool of ``category``, threading only the
    deps each tool declared in its ``Requirements`` (by keyword — factory param
    names match the need fields). Skips NATIVE (factory is None; the deepagents
    ``FilesystemMiddleware`` injects those)."""
    out: list[StructuredTool] = []
    for spec in REGISTRY:
        if spec.category is not category or spec.factory is None:
            continue
        kw: dict[str, Any] = {}
        if spec.needs.exec_client:
            kw["exec_client"] = deps.exec_client
        if spec.needs.backend:
            kw["backend"] = deps.backend
        if spec.needs.vision:
            kw["vision_model"] = deps.vision_model
        if spec.needs.org:
            kw["org"] = deps.org
        out.append(spec.factory(**kw))
    return out


def build_native_specialists(
    exec_client: DockerExecClient, vision_model: object | None = None,
    org: str | None = None, backend: PuxSandboxBackend | None = None,
) -> list[StructuredTool]:
    """Every native ``pux_sandbox_*`` specialist. Thin category filter over
    ``build_tools`` — signature preserved so ``graph.py`` / ``main.py`` and the
    monkeypatching test sites are untouched.

    ``vision_model`` threads the MULTIMODAL LLM into ``describe_image`` /
    ``multimodal`` / ``multimodal_mega`` (model-primary, ONNX/ffmpeg fallback).
    ``org`` scopes the skills tools. ``backend`` is the PuxSandboxBackend the
    three media tools read sandbox files through. See each tool's
    ``Requirements`` entry in ``REGISTRY``."""
    return build_tools(
        ToolDeps(exec_client=exec_client, backend=backend,
                 vision_model=vision_model, org=org),
        Category.SPECIALIST,
    )


def build_grader_tools(exec_client: DockerExecClient) -> list[StructuredTool]:
    """The three ``pux_grader_*`` evidence tools for ``RubricMiddleware``'s
    grader. Relocated here from ``grader.py`` so the registry owns every
    builder (and ``grader.py`` stays a leaf module, avoiding a
    ``grader -> registry -> grader`` import cycle). Signature preserved."""
    return build_tools(ToolDeps(exec_client=exec_client), Category.GRADER)
