"""pux_sandbox_multimodal / pux_sandbox_multimodal_mega — media + prompt tools."""

from __future__ import annotations

import shlex
from pathlib import Path

from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

from deepagents.backends.sandbox import BaseSandbox
from ._shared import _tail, _result, _exec
from ._media import (
    _read_media, _invoke_primary_media, _onnx_describe,
    _media_kind, _extract_video_keyframes, _model_name, _IMAGE_FETCH_TIMEOUT,
    _MediaNonAnswer,
)


class _MultimodalArgs(BaseModel):
    media_path: str | None = Field(
        None, description="Absolute path to a media file inside the sandbox "
        "(image: png/jpg/jpeg/gif/webp/bmp, audio: wav/mp3/flac/ogg/m4a/aac, "
        "video: mp4/webm/avi/mov/mkv)."
    )
    media_url: str | None = Field(
        None, description="URL of a media file to download + analyze. Mutually "
        "exclusive with media_path."
    )
    prompt: str | None = Field(
        None, description="Optional instruction for the model (default: generic "
        "description / transcription / moment-by-moment). e.g. 'what is the "
        "person saying?' or 'read the text on the sign in this frame'."
    )


_MULTIMODAL_DESC = (
    "Send an image, audio clip, OR video clip plus a PROMPT to the multimodal "
    "model and get its reasoning back. The PROMPT is the point — this tool "
    "exists for questions a dedicated transcriber/describer can't answer: 'is "
    "this audio intelligible?', 'does this chart show an upward trend?', 'is "
    "anything in this frame unsafe for work?'. The model judges; you get its "
    "answer. It does NOT silently fall back to ONNX/whisper/keyframes — a "
    "silent downgrade would hand you a generic description in place of the "
    "judgment you asked for, indistinguishable from the real answer. If the "
    "model can't (no multimodal model configured, API error, empty output) you "
    "get an HONEST error with `reason` + `primary_error`; retry, switch to "
    "`multimodal_mega` for an offline-capable waterfall, or `describe_image` "
    "for an image-only ONNX path. Pass either an in-sandbox media path OR a URL."
)

_MULTIMODAL_MEGA_DESC = (
    "Resilient sibling of `multimodal`: same media + prompt → multimodal model "
    "first, but on any model failure it falls back per media type (a WATERFALL). "
    "image -> in-sandbox ONNX vision (describe_image.py); audio -> NO offline "
    "audio fallback exists, so it returns success:false audio_unavailable_offline "
    "(honest — we don't fabricate a transcript); video -> ffmpeg extracts up to 8 "
    "keyframes, each analyzed through the image waterfall, and the per-frame "
    "descriptions are stitched. The result's `source` field reports which tier "
    "produced it (`primary` | `fallback:onnx` | `fallback:keyframes`); "
    "`primary_error` is echoed when a fallback fired. Use this when you want "
    "SOMETHING back even if the model is down — but prefer `multimodal` when you "
    "need the model's prompt-conditioned JUDGMENT (the fallbacks describe, they "
    "don't reason about your prompt)."
)


def _multimodal_validate(media_path: str | None, media_url: str | None) -> dict | None:
    """Shared arg validation for both multimodal tools. Returns an error envelope
    dict if invalid, else ``None``."""
    if not media_path and not media_url:
        return {"success": False,
                "error": "one of media_path or media_url is required"}
    if media_path and media_url:
        return {"success": False,
                "error": "media_path and media_url are mutually exclusive"}
    return None


def _multimodal_unsupported(name: str) -> dict:
    return {
        "success": False,
        "error": f"unsupported media type: {Path(name).suffix!r}",
        "supported": ("image (png/jpg/jpeg/gif/webp/bmp), "
                      "audio (wav/mp3/flac/ogg/m4a/aac), "
                      "video (mp4/webm/avi/mov/mkv)"),
    }


def _multimodal_tool(
    sandbox: BaseSandbox,
    vision_model: object | None = None,
) -> StructuredTool:
    def _run(
        media_path: str | None = None,
        media_url: str | None = None,
        prompt: str | None = None,
    ) -> str:
        bad = _multimodal_validate(media_path, media_url)
        if bad is not None:
            return _result(bad)
        kind = _media_kind(media_path or media_url or "")
        if kind == "unknown":
            return _result(_multimodal_unsupported(media_path or media_url or ""))

        if vision_model is None:
            return _result({
                "success": False, "media_type": kind, "reason": "no_model",
                "explanation": (
                    "No multimodal model is configured, and this tool does not "
                    "fall back. Use `multimodal_mega` for an offline-capable "
                    "waterfall, or `describe_image` for an image-only ONNX path."),
            })
        try:
            b64, mime = _read_media(sandbox, media_path, media_url)
            desc = _invoke_primary_media(vision_model, b64, mime, prompt, kind)
        except _MediaNonAnswer as exc:
            # The model replied but confessed it didn't receive the media (e.g.
            # 'no audio was attached') — distinct from a 429/error: the call
            # succeeded but the wiring didn't land. Surface it as such so the
            # agent doesn't mistake the non-answer for the real judgment.
            return _result({
                "success": False, "media_type": kind, "reason": "model_non_answer",
                "primary_error": _tail(str(exc), 300),
            })
        except Exception as exc:
            return _result({
                "success": False, "media_type": kind, "reason": "model_failed",
                "primary_error": _tail(str(exc), 300),
            })
        return _result({
            "success": True, "description": desc,
            "model": _model_name(vision_model),
            "media_type": kind, "source": "primary",
        })

    return StructuredTool(
        name="multimodal", description=_MULTIMODAL_DESC,
        args_schema=_MultimodalArgs, func=_run,
    )


def _multimodal_mega_tool(
    sandbox: BaseSandbox,
    vision_model: object | None = None,
) -> StructuredTool:
    def _run(
        media_path: str | None = None,
        media_url: str | None = None,
        prompt: str | None = None,
    ) -> str:
        bad = _multimodal_validate(media_path, media_url)
        if bad is not None:
            return _result(bad)
        source_name = media_path or media_url or ""
        kind = _media_kind(source_name)
        if kind == "unknown":
            return _result(_multimodal_unsupported(source_name))

        primary_error: str | None = None
        if vision_model is not None:
            try:
                b64, mime = _read_media(sandbox, media_path, media_url)
                desc = _invoke_primary_media(vision_model, b64, mime, prompt, kind)
                return _result({
                    "success": True, "description": desc,
                    "model": _model_name(vision_model),
                    "media_type": kind, "source": "primary",
                })
            except Exception as exc:
                primary_error = str(exc)
        pe = {"primary_error": _tail(primary_error, 300)} if primary_error else {}

        if kind == "image":
            d = _onnx_describe(
                sandbox, image_path=media_path, image_url=media_url,
                prompt=prompt, primary_error=primary_error,
            )
            d["media_type"] = "image"
            if d.get("success"):
                d["source"] = "fallback:onnx"
            return _result(d)

        if kind == "audio":
            return _result({
                "success": False, "media_type": "audio",
                "reason": "audio_unavailable_offline",
                "explanation": (
                    "The multimodal model could not process this audio clip, "
                    "and no offline audio fallback (e.g. whisper) is installed "
                    "in the sandbox. Retry if the failure looked transient; "
                    "otherwise point a model at it that accepts audio."),
                **pe,
            })

        if media_url and not media_path:
            dl = ("curl -s -L --max-time 60 -o /tmp/pux_mm_video "
                  + shlex.quote(media_url))
            out, exit_code = _exec(sandbox, dl, timeout=_IMAGE_FETCH_TIMEOUT)
            if exit_code != 0:
                return _result({"success": False, "media_type": "video",
                                "reason": "video_download_failed",
                                "error": _tail(out, 200), **pe})
            video_file = "/tmp/pux_mm_video"
        else:
            video_file = media_path or ""

        frames, ferr = _extract_video_keyframes(sandbox, video_file)
        if ferr:
            return _result({"success": False, "media_type": "video",
                            "reason": ferr, **pe})

        per_frame: list[dict] = []
        for fp in frames:
            frame_error: str | None = None
            if vision_model is not None:
                try:
                    b64, _ = _read_media(sandbox, fp, None)
                    desc = _invoke_primary_media(
                        vision_model, b64, "image/png", prompt, "video_frame")
                    per_frame.append({"frame": fp, "success": True,
                                      "description": desc, "source": "primary"})
                    continue
                except Exception as exc:
                    frame_error = str(exc)
            d = _onnx_describe(sandbox, image_path=fp, image_url=None,
                               prompt=prompt, primary_error=frame_error)
            d["frame"] = fp
            per_frame.append(d)

        stitched = "\n\n".join(
            f"[frame {i + 1}] {pf.get('description', '')}"
            for i, pf in enumerate(per_frame) if pf.get("description")
        )
        any_success = any(pf.get("description") for pf in per_frame)
        return _result({
            "success": bool(any_success), "media_type": "video",
            "source": "fallback:keyframes", "frame_count": len(per_frame),
            "frames": per_frame, "description": stitched, **pe,
        })

    return StructuredTool(
        name="multimodal_mega", description=_MULTIMODAL_MEGA_DESC,
        args_schema=_MultimodalArgs, func=_run,
    )
