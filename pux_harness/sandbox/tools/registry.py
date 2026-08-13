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
``agent/org_validation.py`` rule 4 and ``agent/orgs.py`` ``_resolve_tools``).

Per-tool ``Requirements`` (which optional kwargs — vision model / org
scope — a factory needs, and which sandbox capabilities — ffmpeg, xdotool — it
expects) are DECLARED here but NOT acted on: a tool whose capability is
unsatisfied is still registered and reports an honest error at call time (the
multimodal philosophy — present + honest, never a silent drop).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

from langchain_core.tools import StructuredTool

from deepagents.backends.sandbox import BaseSandbox
from ._pux import PUX_PREFIX, PUX_GRADER_PREFIX
from .python import _python_tool
from .skills import _list_skills_tool
from .describe_image import _describe_image_tool
from .multimodal import _multimodal_tool, _multimodal_mega_tool
from .browser import (
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
    _browser_uc_tool,
    _browser_accept_cookies_tool,
    _browser_warmup_history_tool,
    _browser_solve_captcha_tool,
    _browser_reset_tool,
)
from .desktop import (
    _desktop_screenshot_tool,
    _desktop_click_tool,
    _desktop_type_tool,
    _desktop_key_tool,
)
from .grader import (
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
    """What a tool's factory needs beyond the standard ``sandbox: BaseSandbox``.

    Every specialist factory takes ``sandbox`` as its first param — that's the
    portable BaseSandbox contract. ``vision`` and ``org`` flag the two OPTIONAL
    kwargs ``make_specialist_tools`` threads in (``vision_model`` / ``org``).
    ``caps`` is DECLARE-ONLY metadata — sandbox binaries (ffmpeg, xdotool) the
    tool expects to find at call time; never gated on, surfaced for grep + a
    future informational contract warning."""

    vision: bool = False
    org: bool = False
    caps: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.caps, frozenset):
            object.__setattr__(self, "caps", frozenset(self.caps))


@dataclass(frozen=True)
class ToolSpec:
    """One tool, declared once. The single unit of the tool surface."""

    slug: str
    category: Category
    needs: Requirements
    factory: Callable[..., StructuredTool] | None  # None for NATIVE (middleware)
    # Capability GROUP — the coarse knob orgs scope their SUPERVISOR surface by
    # (``policy.yaml`` ``tool_surface.groups``). Specialist slugs roll up to one
    # group; native/grader tools carry ``None`` (they are never group-scoped).
    # Adding a tool means setting its group once here, so the per-org surface
    # (``resolve_tool_allowlist``) and the contract can't drift from REGISTRY.
    group: str | None = None


# --- THE registry ---------------------------------------------------------
# Order is preserved by make_specialist_tools → matches the historical build order.

REGISTRY: list[ToolSpec] = [
    # python + skills. Note: there is no ``load_skill`` — skill bodies
    # are peeked via the native ``read_file`` (SkillsMiddleware advertises each
    # skill's name + description). The ``skills-peek-via-read-file`` contract
    # tripwire makes a re-introduction a HARD failure.
    ToolSpec("python", Category.SPECIALIST, Requirements(), _python_tool, "code"),
    ToolSpec("list_skills", Category.SPECIALIST, Requirements(org=True), _list_skills_tool, "skills"),

    # media (model-primary, declare their caps)
    ToolSpec("describe_image", Category.SPECIALIST,
             Requirements(vision=True, caps={"onnx"}),
             _describe_image_tool, "media"),
    ToolSpec("multimodal", Category.SPECIALIST,
             Requirements(vision=True),
             _multimodal_tool, "media"),
    ToolSpec("multimodal_mega", Category.SPECIALIST,
             Requirements(vision=True, caps={"ffmpeg"}),
             _multimodal_mega_tool, "media"),

    # browser (SeleniumBase Chrome via sb_server)
    ToolSpec("browser_navigate", Category.SPECIALIST, Requirements(), _browser_navigate_tool, "browser"),
    ToolSpec("browser_click", Category.SPECIALIST, Requirements(), _browser_click_tool, "browser"),
    ToolSpec("browser_type", Category.SPECIALIST, Requirements(), _browser_type_tool, "browser"),
    ToolSpec("browser_screenshot", Category.SPECIALIST, Requirements(), _browser_screenshot_tool, "browser"),
    ToolSpec("browser_evaluate", Category.SPECIALIST, Requirements(), _browser_evaluate_tool, "browser"),
    ToolSpec("browser_search", Category.SPECIALIST, Requirements(), _browser_search_tool, "browser"),
    ToolSpec("browser_scroll", Category.SPECIALIST, Requirements(), _browser_scroll_tool, "browser"),
    ToolSpec("browser_go_back", Category.SPECIALIST, Requirements(), _browser_go_back_tool, "browser"),
    ToolSpec("browser_wait", Category.SPECIALIST, Requirements(), _browser_wait_tool, "browser"),
    ToolSpec("browser_find_text", Category.SPECIALIST, Requirements(), _browser_find_text_tool, "browser"),
    ToolSpec("browser_extract", Category.SPECIALIST, Requirements(), _browser_extract_tool, "browser"),
    ToolSpec("browser_extract_images", Category.SPECIALIST, Requirements(), _browser_extract_images_tool, "browser"),
    ToolSpec("browser_save_screenshot", Category.SPECIALIST, Requirements(), _browser_save_screenshot_tool, "browser"),
    ToolSpec("browser_download", Category.SPECIALIST, Requirements(), _browser_download_tool, "browser"),
    ToolSpec("browser_upload", Category.SPECIALIST, Requirements(), _browser_upload_tool, "browser"),
    ToolSpec("browser_tabs", Category.SPECIALIST, Requirements(), _browser_tabs_tool, "browser"),
    ToolSpec("browser_new_tab", Category.SPECIALIST, Requirements(), _browser_new_tab_tool, "browser"),
    ToolSpec("browser_switch_tab", Category.SPECIALIST, Requirements(), _browser_switch_tab_tool, "browser"),
    ToolSpec("browser_close_tab", Category.SPECIALIST, Requirements(), _browser_close_tab_tool, "browser"),
    ToolSpec("browser_dropdown_options", Category.SPECIALIST, Requirements(), _browser_dropdown_options_tool, "browser"),
    ToolSpec("browser_select_dropdown", Category.SPECIALIST, Requirements(), _browser_select_dropdown_tool, "browser"),
    ToolSpec("browser_save_session", Category.SPECIALIST, Requirements(), _browser_save_session_tool, "browser"),
    ToolSpec("browser_restore_session", Category.SPECIALIST, Requirements(), _browser_restore_session_tool, "browser"),
    # SOTA mouse/keyboard/DnD (still pure-CDP SeleniumBase, no new deps)
    ToolSpec("browser_drag", Category.SPECIALIST, Requirements(), _browser_drag_tool, "browser"),
    ToolSpec("browser_hover", Category.SPECIALIST, Requirements(), _browser_hover_tool, "browser"),
    ToolSpec("browser_press", Category.SPECIALIST, Requirements(), _browser_press_tool, "browser"),
    ToolSpec("browser_click_at", Category.SPECIALIST, Requirements(), _browser_click_at_tool, "browser"),
    ToolSpec("browser_scroll_into_view", Category.SPECIALIST, Requirements(), _browser_scroll_into_view_tool, "browser"),
    ToolSpec("browser_a11y", Category.SPECIALIST, Requirements(), _browser_a11y_tool, "browser"),
    ToolSpec("browser_iframe", Category.SPECIALIST, Requirements(), _browser_iframe_tool, "browser"),
    # Captcha bypass + fingerprint-legitimacy tools (SeleniumBase UC mode + curated consent)
    ToolSpec("browser_uc", Category.SPECIALIST, Requirements(), _browser_uc_tool, "browser"),
    ToolSpec("browser_accept_cookies", Category.SPECIALIST, Requirements(), _browser_accept_cookies_tool, "browser"),
    ToolSpec("browser_warmup_history", Category.SPECIALIST, Requirements(), _browser_warmup_history_tool, "browser"),
    ToolSpec("browser_solve_captcha", Category.SPECIALIST, Requirements(), _browser_solve_captcha_tool, "browser"),
    ToolSpec("browser_reset", Category.SPECIALIST, Requirements(), _browser_reset_tool, "browser"),

    # desktop (xdotool + Xvfb)
    ToolSpec("desktop_screenshot", Category.SPECIALIST,
             Requirements(caps={"xdotool", "x11"}), _desktop_screenshot_tool, "desktop"),
    ToolSpec("desktop_click", Category.SPECIALIST,
             Requirements(caps={"xdotool", "x11"}), _desktop_click_tool, "desktop"),
    ToolSpec("desktop_type", Category.SPECIALIST,
             Requirements(caps={"xdotool", "x11"}), _desktop_type_tool, "desktop"),
    ToolSpec("desktop_key", Category.SPECIALIST,
             Requirements(caps={"xdotool", "x11"}), _desktop_key_tool, "desktop"),

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
    ToolSpec("execute", Category.GRADER, Requirements(), _grader_execute_tool),
    ToolSpec("read_file", Category.GRADER, Requirements(), _grader_read_file_tool),
    ToolSpec("grep", Category.GRADER, Requirements(), _grader_grep_tool),
]


# --- everything below DERIVES from REGISTRY — no hand-maintained copies ----

SPECIALISTS: frozenset[str] = frozenset(
    s.slug for s in REGISTRY if s.category is Category.SPECIALIST
)
SPECIALIST_TOOL_NAMES: frozenset[str] = frozenset(
    PUX_PREFIX + s for s in SPECIALISTS
)

# Capability GROUP -> the specialist slugs in it. DERIVED from REGISTRY (every
# group'd ToolSpec rolls up here), so a new tool with a group auto-joins its
# group with no second list to maintain. The contract + ``resolve_tool_allowlist``
# read ONLY this, so the per-org surface can never drift from REGISTRY.
TOOL_GROUP_TOOLS: dict[str, frozenset[str]] = {}
for _spec in REGISTRY:
    if _spec.category is Category.SPECIALIST and _spec.group is not None:
        TOOL_GROUP_TOOLS[_spec.group] = TOOL_GROUP_TOOLS.get(_spec.group, frozenset()) | frozenset({_spec.slug})
# A canonical listing of group names (for validation + docs).
SPECIALIST_GROUPS: frozenset[str] = frozenset(TOOL_GROUP_TOOLS)


def resolve_tool_allowlist(entries: Sequence[str]) -> frozenset[str]:
    """Resolve a ``tool_surface.groups`` list into the set of SPECIALIST slugs
    the SUPERVISOR may carry. OPT-IN: an org without a ``tool_surface`` block
    gets ``frozenset()`` — NO specialists on the supervisor (they still reach
    it via subagents, which resolve their own ``tools:`` against the FULL
    surface). An org opts in by listing the groups/bare-slugs it wants on the
    supervisor's own surface.

    Each ``entries`` item is EITHER a known group name (``code`` / ``skills`` /
    ``media`` / ``browser`` / ``desktop``) — which expands to every tool in that
    group — OR a bare specialist slug (``python``) — which enables just that one
    tool. This lets an org say ``groups: [code, skills, media]`` (carry code +
    skills + media on the supervisor, drop browser/desktop) or ``groups:
    [python]`` (only python). An unknown entry fails loud (PolicyError) so a
    typo'd group can't silently ship the wrong surface. Empty list ->
    ``frozenset()`` (no specialists on the supervisor — the opt-in default)."""
    if not entries:
        return frozenset()
    allowed: set[str] = set()
    for entry in entries:
        if entry in TOOL_GROUP_TOOLS:
            allowed |= set(TOOL_GROUP_TOOLS[entry])
        elif entry in SPECIALISTS:
            allowed.add(entry)
        else:
            raise ValueError(
                f"tool_surface.groups: {entry!r} is neither a known group "
                f"({sorted(SPECIALIST_GROUPS)}) nor a specialist tool slug "
                f"({sorted(SPECIALISTS)})"
            )
    return frozenset(allowed)
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
# rule: co-located + named here so the coder forcing surface check
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


def make_specialist_tools(
    sandbox: BaseSandbox, *, vision_model: object | None = None,
    org: str | None = None, apply_prefix: bool = True,
) -> list[StructuredTool]:
    """Build every native ``pux_sandbox_*`` specialist tool.

    Takes a standard ``BaseSandbox`` — the SAME interface any LangChain
    project provides (OpenShell, LocalShell, E2B, …). The specialist tools
    (python/browser/desktop/grader/media) are zero-pux-import factories that
    accept ``BaseSandbox`` directly; they are portable to any langchain project.

    Each factory creates a StructuredTool with a PLAIN name (``python``,
    ``browser_navigate``, …). When ``apply_prefix`` is True (default), the
    ``pux_sandbox_`` prefix is applied so org configs and the contract layer
    (``resolve_tool_allowlist`` / ``prefixed``) continue matching.

    ``vision_model`` threads the multimodal LLM into ``describe_image`` /
    ``multimodal`` / ``multimodal_mega`` (skipped when None).
    ``org`` scopes the skills tools (skipped when None)."""
    out: list[StructuredTool] = []
    for spec in REGISTRY:
        if spec.category is not Category.SPECIALIST or spec.factory is None:
            continue
        kw: dict[str, Any] = {"sandbox": sandbox}
        if spec.needs.vision:
            kw["vision_model"] = vision_model
        if spec.needs.org:
            kw["org"] = org
        out.append(spec.factory(**kw))

    if apply_prefix:
        for t in out:
            t.name = PUX_PREFIX + t.name

    return out


def build_grader_tools(sandbox: BaseSandbox) -> list[StructuredTool]:
    """The three ``pux_grader_*`` evidence tools for ``RubricMiddleware``'s
    grader. Takes a ``BaseSandbox`` — the tools are portable."""
    out: list[StructuredTool] = []
    for spec in REGISTRY:
        if spec.category is not Category.GRADER or spec.factory is None:
            continue
        tool = spec.factory(sandbox)
        tool.name = PUX_GRADER_PREFIX + tool.name
        out.append(tool)
    return out
