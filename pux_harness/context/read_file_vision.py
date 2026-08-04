"""ReadFileVisionMiddleware — automatic image/binary fallback for non-multimodal
drivers. The MISSING automatic fallback system: when a non-multimodal model
calls ``read_file`` on an image (or any binary file), deepagents'
``FilesystemMiddleware`` returns a ``ToolMessage`` whose ``content`` is a list
of content blocks — ``[{"type":"image","base64":...,"mime_type":...}]``. That
goes straight to the model. A non-multimodal model (e.g. the default tier's
glm-5.2 supervisor, or any text-only orchestrator/coder) cannot handle image
content blocks → the gateway returns HTTP 400 → the run crashes.

This was the user's report: "Read img causes a fail as it's only allowed to
read text? We had a full fucking in place. Where there was a AUTOMATIC
fallbacks system. When a model that wasn't multimodal read an image, the
request would be routed to the 'read_media' tool, which was our multi-model
model doing a oneshot read of the file and describing it for the calling
model."

THREE LAYERS OF DEFENSE
    This middleware uses THREE independent layers to ensure no binary content
    block EVER reaches a non-multimodal model:

    Layer 1 — ``wrap_tool_call`` INTERCEPTION (primary):
        When a tool returns a ToolMessage with binary content blocks, the
        middleware replaces the binary with auto-describe text (calls the
        vision model) or a text pointer (fallback). This catches the image at
        the SOURCE — it never enters message history.

    Layer 2 — ``wrap_model_call`` SAFETY NET (secondary):
        Right BEFORE each model call, the middleware scans ALL messages in the
        request. If ANY message still has binary content blocks (from a path
        that bypassed Layer 1 — a Command return, a pre-existing thread, a
        HumanMessage injected by another middleware, etc.), they are stripped
        and replaced with text placeholders. This is the LAST LINE of defense:
        nothing non-text can reach the model.

    Layer 3 — auto-describe fallback:
        When the vision model call itself fails (transient provider error,
        rate limit, etc.), the middleware falls back to a text pointer to
        ``describe_image``, so the model always gets a steer.

MULTIMODAL DRIVERS
    When ``multimodal_driver=True`` ALL layers are no-ops — the image passes
    through untouched so a multimodal driver (mimo-v2.5) reads it natively.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

# Content block types that carry binary payload a text-only model cannot
# consume. We use a DENYLIST approach: ``text`` blocks pass through, EVERYTHING
# else (``image``, ``image_url``, ``file``, ``video_frame``, or any future
# type) is stripped. This is safer than an allowlist because ChatOpenAI
# converts ``{"type": "image"}`` → ``{"type": "image_url"}`` internally, and
# providers may invent new block types. Only ``"text"`` is safe for a
# text-only model.
_ALLOWED_BLOCK_TYPES = frozenset({"text"})

# The prompt sent to the vision model for auto-describe.
_DESCRIBE_PROMPT = (
    "Briefly describe this image in 2-3 sentences. Focus on the key visual "
    "elements: what objects, text, colors, or UI elements are visible. "
    "Be concise — this description is for a text-only model that cannot see "
    "the image."
)


def _is_non_text_block(block: Any) -> bool:
    """True iff a content block is a dict whose ``type`` is NOT ``text``.
    Covers ``image``, ``image_url``, ``file``, ``video_frame``, and any future
    type a provider might introduce. Non-dict blocks (strings, None) are NOT
    non-text (they're raw content, not blocks)."""
    if not isinstance(block, dict):
        return False
    return block.get("type") not in _ALLOWED_BLOCK_TYPES


def _has_binary_blocks(content: Any) -> bool:
    """True iff ``content`` is a list of content blocks containing at least one
    non-text block (image, image_url, file, video_frame, etc.)."""
    if not isinstance(content, list):
        return False
    return any(_is_non_text_block(b) for b in content)


def _strip_binary_blocks(content: Any) -> tuple[list, bool]:
    """Replace ALL non-text content blocks with text placeholders. Returns
    ``(new_content, changed)``. Uses a denylist: anything that's not
    ``{"type": "text", ...}`` is replaced. Non-list content passes through."""
    if not isinstance(content, list):
        return content, False
    changed = False
    new_blocks = []
    for block in content:
        if _is_non_text_block(block):
            b64 = block.get("base64", "")
            # Keep a SHORT fingerprint so the model can tell which image this
            # was (first 20 chars of base64), without the full payload.
            fingerprint = (b64[:20] + "...") if len(b64) > 20 else b64
            new_blocks.append({
                "type": "text",
                "text": f"[binary/image content removed — {fingerprint}]",
            })
            changed = True
        else:
            new_blocks.append(block)
    if not changed:
        return content, False  # identity — no copy needed
    return new_blocks, True


def _extract_path(result: ToolMessage) -> str:
    """Pull the file path from the ToolMessage's ``additional_kwargs``."""
    kwargs = result.additional_kwargs or {}
    path = kwargs.get("read_file_path") or kwargs.get("path") or ""
    return str(path) if path else ""


def _extract_image_data(result: ToolMessage) -> tuple[str, str] | None:
    """Extract ``(base64, mime_type)`` from the first non-text content block
    (image, image_url, file, etc.)."""
    if not isinstance(result.content, list):
        return None
    for block in result.content:
        if not isinstance(block, dict):
            continue
        if not _is_non_text_block(block):
            continue
        b64 = block.get("base64") or block.get("source", {}).get("data")
        # Also handle image_url format: {"type": "image_url", "image_url": {"url": "data:..."}}
        if not b64 and block.get("type") == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                # Parse data URI: data:<mime>;base64,<data>
                import re
                m = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
                if m:
                    return m.group(2), m.group(1)
        mime = block.get("mime_type") or "image/png"
        if b64:
            return str(b64), str(mime)
    return None


def _build_text_pointer(result: ToolMessage) -> ToolMessage:
    """Construct a replacement ToolMessage with a text pointer to
    ``describe_image``."""
    path = _extract_path(result)
    if path:
        text = (
            f"[Binary/image file read: {path}] "
            f"You cannot view images or binary files directly. "
            f"Call describe_image(image_path='{path}', "
            f"prompt=\"<what you want to know about this image>\") "
            f"to get a text description of its contents."
        )
    else:
        text = (
            f"[Binary/image content from tool '{result.name}' "
            f"(call {result.tool_call_id})] "
            f"You cannot view images or binary files directly. "
            f"Call describe_image(image_path='<path>', prompt=...) "
            f"to inspect image files."
        )
    return ToolMessage(
        content=text,
        name=result.name or "read_file",
        tool_call_id=result.tool_call_id,
        status=getattr(result, "status", "success"),
    )


def _build_description_msg(
    result: ToolMessage, description: str
) -> ToolMessage:
    """Build a ToolMessage carrying the vision model's description."""
    path = _extract_path(result)
    prefix = f"[Image file: {path}]" if path else f"[Image from {result.name}]"
    return ToolMessage(
        content=f"{prefix} {description}",
        name=result.name or "read_file",
        tool_call_id=result.tool_call_id,
        status=getattr(result, "status", "success"),
    )


class ReadFileVisionMiddleware(AgentMiddleware):
    """Three-layer defense ensuring no binary content block EVER reaches a
    non-multimodal model. See module docstring for the three layers.

    Mounted on ALL agents (supervisor + every subagent). No-op when the driver
    is multimodal (all layers short-circuit).
    """

    def __init__(
        self,
        *,
        multimodal_driver: bool = True,
        enabled: bool = True,
        vision_model: Any = None,
    ) -> None:
        self.multimodal_driver = multimodal_driver
        self.enabled = enabled
        self.vision_model = vision_model

    # --- Layer 1: wrap_tool_call — intercept at the source ---

    def _auto_describe(self, result: ToolMessage) -> ToolMessage | None:
        """Call the vision model to describe the image. Returns a ToolMessage
        with the description, or ``None`` to trigger text-pointer fallback."""
        if self.vision_model is None:
            return None
        image_data = _extract_image_data(result)
        if image_data is None:
            return None
        b64, mime = image_data
        try:
            # Use the deepagents canonical image format — the SAME format
            # read_file emits and BrowserVisionMiddleware uses. ChatOpenAI
            # translates it to the provider's image_url form. Using image_url
            # directly caused "content.type is invalid" errors on some
            # providers (code 1210) that only accept the translated form.
            msg = HumanMessage(content=[
                {"type": "text", "text": _DESCRIBE_PROMPT},
                {"type": "image", "base64": b64, "mime_type": mime},
            ])
            response = self.vision_model.invoke([msg])
            description = (
                response.content.strip()
                if hasattr(response, "content")
                else str(response).strip()
            )
            if description:
                return _build_description_msg(result, description)
        except Exception as e:
            logger.warning("read_file_vision auto-describe failed: %s", e)
        return None

    async def _auto_describe_async(self, result: ToolMessage) -> ToolMessage | None:
        """Async variant of ``_auto_describe``."""
        if self.vision_model is None:
            return None
        image_data = _extract_image_data(result)
        if image_data is None:
            return None
        b64, mime = image_data
        try:
            msg = HumanMessage(content=[
                {"type": "text", "text": _DESCRIBE_PROMPT},
                {"type": "image", "base64": b64, "mime_type": mime},
            ])
            response = await self.vision_model.ainvoke([msg])
            description = (
                response.content.strip()
                if hasattr(response, "content")
                else str(response).strip()
            )
            if description:
                return _build_description_msg(result, description)
        except Exception as e:
            logger.warning("read_file_vision auto-describe failed: %s", e)
        return None

    def _intercept(self, result: Any) -> Any:
        """Layer 1: intercept tool results with binary blocks. Tries
        auto-describe first, falls back to text pointer."""
        if not self.enabled or self.multimodal_driver:
            return result
        if not isinstance(result, ToolMessage):
            return result
        if not _has_binary_blocks(result.content):
            return result
        described = self._auto_describe(result)
        if described is not None:
            return described
        return _build_text_pointer(result)

    async def _intercept_async(self, result: Any) -> Any:
        if not self.enabled or self.multimodal_driver:
            return result
        if not isinstance(result, ToolMessage):
            return result
        if not _has_binary_blocks(result.content):
            return result
        described = await self._auto_describe_async(result)
        if described is not None:
            return described
        return _build_text_pointer(result)

    def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled or self.multimodal_driver:
            return handler(request)
        return self._intercept(handler(request))

    async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled or self.multimodal_driver:
            return await handler(request)
        return await self._intercept_async(await handler(request))

    # --- Layer 2: wrap_model_call — safety net before model sees messages ---

    def _sanitize_messages(self, messages: list) -> tuple[list, bool]:
        """Scan ALL messages and strip binary content blocks from ANY message
        type (ToolMessage, HumanMessage, AIMessage, SystemMessage). Returns
        ``(new_messages, changed)``. When nothing changed, the original list
        is returned (identity check allows the caller to skip override)."""
        changed = False
        new_messages = []
        for msg in messages:
            content = getattr(msg, "content", None)
            if content is None:
                new_messages.append(msg)
                continue
            new_content, did_change = _strip_binary_blocks(content)
            if did_change:
                changed = True
                # Reconstruct the message with cleaned content. We can't
                # mutate frozen dataclasses, so create a new one via
                # model_copy (pydantic) or simple reconstruction.
                try:
                    new_msg = msg.model_copy(update={"content": new_content})
                except Exception:
                    # Fallback: direct attribute set (works for non-frozen)
                    try:
                        msg.content = new_content
                        new_msg = msg
                    except Exception:
                        new_msg = msg  # last resort — leave unchanged
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        return new_messages, changed

    def wrap_model_call(self, request, handler):  # type: ignore[no-untyped-def]
        """Layer 2 safety net: strip ANY remaining binary blocks from ALL
        messages right before the model sees them. This catches leaks from
        paths that bypassed Layer 1 (Command returns, pre-existing threads,
        HumanMessages injected by other middlewares, etc.)."""
        if not self.enabled or self.multimodal_driver:
            return handler(request)
        messages = getattr(request, "messages", None)
        if not messages:
            return handler(request)
        sanitized, changed = self._sanitize_messages(messages)
        if changed:
            request = request.override(messages=sanitized)
        return handler(request)

    async def awrap_model_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled or self.multimodal_driver:
            return await handler(request)
        messages = getattr(request, "messages", None)
        if not messages:
            return await handler(request)
        sanitized, changed = self._sanitize_messages(messages)
        if changed:
            request = request.override(messages=sanitized)
        return await handler(request)
