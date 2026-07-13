"""BrowserVisionMiddleware — surface post-action screenshots as native image
blocks so a multimodal driver (the shipped default, mimo-v2.5) can SEE the page
after each action and decide the next step. This is the vision-in-the-loop
"look → reason → act → look" loop the SOTA browser agents (computer-use,
browser-use) run.

THE BLOCK
    deepagents' canonical image ContentBlock is ``{"type":"image","base64":...,
    "mime_type":...}`` — the same shape ``read_file`` emits and
    ``_media._media_content_block`` builds. ``ChatOpenAI`` translates it to the
    provider ``image_url`` form. We reuse that EXACT shape.

WHY A COMPANION HumanMessage, NOT AN IMAGE IN THE ToolMessage
    The OpenAI-style tool-result role accepts a STRING content widely, but
    MULTIMODAL tool-result content is provider/gateway-dependent: the shipped
    OpenCode-Zen-Go gateway upstream-rejects (HTTP 400) an ``image_url`` block
    inside a tool-role message, even though it accepts the SAME image block in
    a user-role message (proven live against mimo-v2.5: it reads example.com
    correctly from a HumanMessage image, 400s from a ToolMessage image). So we
    keep the ToolMessage text-only (the tool_call still gets its paired result)
    and emit a SECOND message — a HumanMessage carrying the image — right after
    it. The model sees ``[...tool_result(text), human(screenshot)]``, which the
    gateway accepts and the model reads. This is the universally-compatible
    form (images-in-user-messages work on every vision provider); image-in-tool
    does not.

THE SEAM
    ``wrap_tool_call(request, handler)`` runs AFTER the framework bound the
    correct ``tool_call_id`` to the result ``ToolMessage``. We return a
    ``Command(update={"messages": [text_tool_message, human_image_message]})``
    — the canonical "tool emits multiple messages" shape
    (deepagents/middleware/subagents.py::_return_command_with_state_update).
    The text ToolMessage keeps its tool_call_id so the reducer still pairs it
    with the pending tool call; the HumanMessage is appended after. A Command
    is not a text ToolMessage, so ``ContextMiddleware._is_text_tm`` returns
    False and it passes through offload untouched. For the same reason this
    middleware mounts INNERMOST (last in the stack): ``handler(request)`` then
    returns the RAW tool string before ``ContextMiddleware`` could offload it,
    so ``screenshot_path`` is still inline to find.

DATA-DRIVEN
    We attach iff the tool's JSON result carries a ``screenshot_path`` — every
    page-mutating browser endpoint already returns one. No per-tool allowlist.

HONEST FAILURE
    A fetch failure leaves the ORIGINAL text ToolMessage in place (no Command,
    no image). The image is an enhancement, never a replacement; a missing
    image is honest, never a silent fallback that pretends vision works.

GATING / MODE
    Default ON regardless of driver — vision is always wired. ``PUX_BROWSER_VISION=0``
    fully disables it (clean absent-from-list). The MODE is selected per-scope by
    the driver's capability (``model.driver_multimodal``, threaded in by
    ``stack._build_browser_vision``): a MULTIMODAL driver (e.g. the ``fast`` tier's
    mimo-v2.5) gets the native image block above; a TEXT-only driver (the shipped
    DEFAULT tier's glm-5.2 supervisor) gets a TEXT POINTER — "screenshot saved at
    <path>; call ``describe_image(image_path=<path>, prompt=...)`` to inspect it"
    — so a non-multimodal driver reaches vision through the ``multimodal`` role
    (``describe_image`` routes there) instead of an image block it cannot read.
    No base64 fetch in the text-pointer mode (cheaper — the image never leaves the
    container; the typed tool re-reads it on demand).
"""
from __future__ import annotations

import json
import os
import shlex
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

# Every browser specialist tool is named ``pux_sandbox_browser_<slug>``.
_BROWSER_PREFIX = "pux_sandbox_browser_"

# Per-action screenshot policy. Attaching a screenshot (~1–1.7K vision tokens)
# after EVERY browser action is the SOTA "look → reason → act" loop, but most
# actions are deterministic — the agent typed text it already knows, scrolled a
# known amount, or ran a read query whose RETURN VALUE is the ground truth, not
# the pixels. Auto-screenshotting those wastes ~60% of the browser's vision
# tokens. So only tools where the page's visual state may change UNPREDICTABLY
# (navigation, click-may-navigate, hover-reveals-menu, drag-moved-something)
# auto-attach. For everything else the agent calls ``browser_screenshot``
# explicitly when it wants to look. New browser tools default to text-only
# (add them here iff vision is needed after them).
_SCREENSHOT_SLUGS = frozenset({
    "navigate", "search", "go_back", "new_tab", "switch_tab",
    "click", "click_at", "hover", "drag", "select_dropdown",
    "accept_cookies", "uc", "screenshot",
})

# Cap the fetch so a pathological path can't hang the agent — a screenshot over
# this is dropped (text result still ships).
_MAX_SCREENSHOT_BYTES = 4 * 1024 * 1024  # 4 MiB


def _screenshot_b64(exec_client: Any, path: str) -> str | None:
    """Read ``path`` out of the sandbox container as base64.

    ``base64 -w0`` (coreutils, always present in the image) streams the file to
    stdout with no line wrapping; the host decodes nothing here — the gateway
    wants the raw base64 in the image block. Returns ``None`` on any failure so
    the caller skips the image honestly."""
    if not path or not isinstance(path, str):
        return None
    try:
        out, code = exec_client.exec(f"base64 -w0 {shlex.quote(path)}")
    except Exception:
        return None
    if code:
        return None
    b64 = (out or "").strip()
    if not b64:
        return None
    # base64 inflates ~1.33×; reject anything decoding past the cap.
    if len(b64) > _MAX_SCREENSHOT_BYTES * 4 // 3:
        return None
    return b64


def _mime_for(path: str) -> str:
    p = path.lower()
    if p.endswith(".jpg") or p.endswith(".jpeg"):
        return "image/jpeg"
    if p.endswith(".webp"):
        return "image/webp"
    return "image/png"  # sb_server only ever writes .png screenshots


class BrowserVisionMiddleware(AgentMiddleware):
    """Attach each browser tool's screenshot as a native image block, delivered
    as a companion HumanMessage alongside the text ToolMessage result."""

    def __init__(
        self,
        exec_client: Any,
        *,
        enabled: bool = True,
        multimodal_driver: bool = True,
    ) -> None:
        self.exec_client = exec_client
        self.enabled = enabled
        # True  -> attach the screenshot as a native image block (the driver can
        #          read images directly). False -> emit a text pointer to
        #          describe_image (the driver is text-only; vision is delegated
        #          to the multimodal role). See the GATING/MODE note above. The
        #          default (True) preserves the pre-fallback behavior for any
        #          direct construction; stack._build_browser_vision threads the
        #          real per-scope capability in.
        self.multimodal_driver = multimodal_driver

    # request is langchain's ToolCallRequest (or a SimpleNamespace stand-in in
    # tests) — read tool_call defensively, mirroring ContextMiddleware.
    @staticmethod
    def _tool_name(request: Any) -> str:
        tc = getattr(request, "tool_call", None) or {}
        if isinstance(tc, dict):
            name = tc.get("name")
            return str(name) if name is not None else "tool"
        return "tool"

    def _enrich(self, result: Any) -> Any:
        """Return ``Command([text_tool_message, companion])`` iff ``result`` is a
        browser ToolMessage whose JSON carries a ``screenshot_path``. The
        companion is a native image block when the driver is multimodal, or a
        text pointer to ``describe_image`` when the driver is text-only.
        Otherwise return ``result`` unchanged."""
        if not isinstance(result, ToolMessage):
            return result
        if isinstance(result.content, list):
            return result  # already multimodal — another layer enriched it
        content = result.content if isinstance(result.content, str) else ""
        if not content:
            return result
        try:
            payload = json.loads(content)
        except (ValueError, TypeError):
            return result  # not JSON (e.g. an error string) — no image to attach
        if not isinstance(payload, dict):
            return result
        path = payload.get("screenshot_path")
        if not isinstance(path, str) or not path:
            return result
        if not self.multimodal_driver:
            # Text-only driver: it cannot read an image block. Point it at the
            # screenshot path + steer it toward the RIGHT verification channel.
            # The SoM element map (numbered labels) is already in the tool
            # result JSON — the model can read those. For ASSERTING page state
            # (element exists, text rendered, value correct), browser_evaluate
            # is exact DOM truth — steer there FIRST. describe_image (the vision
            # proxy) is for visual-only checks (layout, color, "does it look
            # right") where a DOM assertion can't capture the property — it's
            # hallucination-prone on fine detail, so it's the fallback, not the
            # default. No base64 fetch — the image stays in the container;
            # describe_image re-reads it on demand.
            human = HumanMessage(content=[{"type": "text", "text": (
                f"[screenshot result for {result.name} tool_call "
                f"{result.tool_call_id} at {path}] You cannot view it directly. "
                f"Prefer browser_evaluate for assertions (DOM truth: "
                f"document.querySelector, innerText, pixel counts). "
                f"For visual-only checks (layout, color), call "
                f"describe_image(image_path={path!r}, "
                f"prompt=\"<what to check>\")."
            )}])
            return Command(update={"messages": [result, human]})
        b64 = _screenshot_b64(self.exec_client, path)
        if not b64:
            return result  # fetch failed — ship the text result unchanged
        # The companion HumanMessage: a short label + the native image block.
        # Label names the tool_call so the model reads this as "the screenshot
        # that tool produced", not a fresh user interjection.
        human = HumanMessage(content=[
            {"type": "text",
             "text": f"[screenshot result for {result.name} tool_call {result.tool_call_id}]"},
            {"type": "image", "base64": b64, "mime_type": _mime_for(path)},
        ])
        # Keep the text ToolMessage (tool_call_id paired) THEN append the image.
        return Command(update={"messages": [result, human]})

    def _should_screenshot(self, request: Any) -> bool:
        """True iff this browser tool's slug is in the screenshot policy set.
        Non-browser tools short-circuit before this is called."""
        name = self._tool_name(request)
        if not name.startswith(_BROWSER_PREFIX):
            return False
        return name[len(_BROWSER_PREFIX):] in _SCREENSHOT_SLUGS

    def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled or not self._should_screenshot(request):
            return handler(request)
        return self._enrich(handler(request))

    async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled or not self._should_screenshot(request):
            return await handler(request)
        result = await handler(request)
        return self._enrich(result)


def browser_vision_enabled() -> bool:
    """Default ON — vision is always wired (a text-only driver gets the
    describe_image text-pointer mode, not a silent drop). A cloner who wants
    vision fully off sets ``PUX_BROWSER_VISION=0``."""
    return os.getenv("PUX_BROWSER_VISION", "1") != "0"
