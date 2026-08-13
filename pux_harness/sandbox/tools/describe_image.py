"""pux_sandbox_describe_image — primary multimodal model with ONNX fallback."""

from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

from deepagents.backends.sandbox import BaseSandbox
from ._shared import _tail, _result
from ._media import _read_media, _invoke_primary_media, _onnx_describe, _model_name


class _DescribeImageArgs(BaseModel):
    image_path: str | None = Field(
        None, description="Absolute path to image file inside the sandbox "
        "(e.g. /sandbox/workspace/foo.png)"
    )
    image_url: str | None = Field(
        None, description="URL of image to download and describe. Mutually "
        "exclusive with image_path."
    )
    prompt: str | None = Field(
        None, description="Optional instruction for the model (default: generic "
        "description). e.g. 'what text is on the sign?'"
    )


_DESCRIBE_IMAGE_DESC = (
    "Describe an image. PRIMARY path: the driving model (mimo-v2.5) reads the "
    "image natively via multimodal input — fast, no local model load. FALLBACK "
    "path: if the driving model can't see the image (non-multimodal model, API "
    "error, empty output), an in-sandbox ONNX vision model "
    "(Qwen3.5-2B-ONNX-OPT) describes it locally. Pass either an in-sandbox "
    "image path OR a URL. The result's `source` field reports which path "
    "produced the description (`primary` | `fallback` | `onnx`)."
)


def _describe_image_tool(
    sandbox: BaseSandbox,
    vision_model: object | None = None,
) -> StructuredTool:
    def _run(
        image_path: str | None = None,
        image_url: str | None = None,
        prompt: str | None = None,
    ) -> str:
        if not image_path and not image_url:
            return _result({"success": False, "error": "one of image_path or image_url is required"})
        if image_path and image_url:
            return _result({"success": False, "error": "image_path and image_url are mutually exclusive"})

        primary_error: str | None = None
        if vision_model is not None:
            try:
                b64, mime = _read_media(sandbox, image_path, image_url)
                desc = _invoke_primary_media(vision_model, b64, mime, prompt, "image")
                return _result({
                    "success": True,
                    "description": desc,
                    "model": _model_name(vision_model),
                    "source": "primary",
                })
            except Exception as exc:
                primary_error = str(exc)
        {"primary_error": _tail(primary_error, 300)} if primary_error else {}

        return _result(_onnx_describe(
            sandbox, image_path=image_path, image_url=image_url,
            prompt=prompt, primary_error=primary_error,
        ))

    return StructuredTool(
        name="describe_image", description=_DESCRIBE_IMAGE_DESC,
        args_schema=_DescribeImageArgs, func=_run,
    )
