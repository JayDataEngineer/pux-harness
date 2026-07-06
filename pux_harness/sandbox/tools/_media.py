"""Shared media utilities for describe_image and multimodal tools.

ONNX fallback, base64 media acquisition, media type detection, primary
multimodal model invocation, and video keyframe extraction. Every public
function here is consumed by ``describe_image.py`` and/or ``multimodal.py``.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import shlex
import tempfile
from pathlib import Path
from datetime import datetime

from langchain_core.messages import HumanMessage

from pux_harness.sandbox.docker_exec import DockerExecClient, ExecTimeout
from pux_harness.sandbox.backend import PuxSandboxBackend
from pux_harness.sandbox.tools._shared import _tail, _result

log = logging.getLogger(__name__)

# --- constants ---------------------------------------------------------------

_DESCRIBE_IMAGE_SCRIPT = "/usr/local/bin/describe_image.py"
_DESCRIBE_IMAGE_TIMEOUT = 120
_IMAGE_FETCH_TIMEOUT = 60

_VISION_UNAVAILABLE = (
    "Vision model is not downloaded. Run scripts/bootstrap-vision.sh from the "
    "host to enable. (This message means BOTH paths failed: the driving model "
    "could not describe the image, and the local ONNX fallback is not "
    "bootstrapped.)"
)
_VISION_DEPS_MISSING = (
    "Sandbox image is missing onnxruntime-genai. Rebuild with `task build` "
    "after pulling latest sandbox/Dockerfile."
)

_MEDIA_KIND_BY_EXT: dict[str, str] = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".heic": "image", ".heif": "image", ".bmp": "image",
    ".wav": "audio", ".mp3": "audio", ".aiff": "audio", ".aac": "audio",
    ".ogg": "audio", ".flac": "audio", ".m4a": "audio",
    ".mp4": "video", ".mpeg": "video", ".mov": "video", ".avi": "video",
    ".flv": "video", ".mpg": "video", ".webm": "video", ".wmv": "video",
    ".3gpp": "video", ".mkv": "video",
}

_VIDEO_KEYFRAMES = 8
_KEYFRAME_TIMEOUT = 120


def _guess_mime(path: str) -> str:
    """MIME type for any file path — via stdlib."""
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def _model_name(model: object) -> str:
    """The model id of a ChatOpenAI instance (``.model_name`` / ``.model``)."""
    return getattr(model, "model_name", None) or getattr(model, "model", None) or "model"


def _media_kind(path: str) -> str:
    """``image`` | ``audio`` | ``video`` | ``unknown`` by extension."""
    return _MEDIA_KIND_BY_EXT.get(Path(path).suffix.lower(), "unknown")


def _default_media_prompt(kind: str) -> str:
    if kind == "audio":
        return ("Transcribe and describe this audio clip. Note any speech "
                "(quote it), music, and key sounds.")
    if kind == "video":
        return "Describe what happens in this video clip, moment by moment."
    if kind == "video_frame":
        return ("Describe this video frame concisely — the action, any text, "
                "and key visual features.")
    return ("Describe this media concisely. Focus on text, key elements, and "
            "notable features.")


def _read_media(
    backend: PuxSandboxBackend,
    exec_client: DockerExecClient,
    path: str | None,
    url: str | None,
) -> tuple[str, str]:
    """Acquire media bytes → ``(base64, mime)``. Uses ``backend.read()`` for
    sandbox paths and curl for URLs (sandbox egress policy applies)."""
    if path:
        rr = backend.read(path)
        if rr.error:
            raise RuntimeError(rr.error)
        if rr.file_data is None:
            raise RuntimeError(f"no data for {path}")
        return rr.file_data["content"], _guess_mime(path)
    cmd = f"curl -s -L --max-time 30 {shlex.quote(url or '')} | base64 -w0"
    out, exit_code = exec_client.exec(cmd, timeout=_IMAGE_FETCH_TIMEOUT)
    b64 = (out or "").strip()
    if exit_code != 0 or not b64:
        raise RuntimeError(f"url fetch exit {exit_code}: {_tail(out, 200)}")
    return b64, _guess_mime(url or "")


def _media_content_block(kind: str, b64: str, mime: str) -> dict:
    """The multimodal content block to send to the model for ``kind``.

    NATIVE for image + audio: the canonical deepagents ContentBlock
    ``{"type": "image"|"audio", "base64": <b64>, "mime_type": <mime>}`` — the
    same shape ``read_file`` produces (deepagents/middleware/filesystem.py:1133).
    ``ChatOpenAI`` translates it downstream into the provider wire form (image →
    ``image_url`` data-URI; audio → ``input_audio`` with ``format`` DERIVED from
    ``mime_type``), verified via ``convert_to_openai_data_block``. This replaced
    the hand-rolled ``image_url`` / ``input_audio`` dicts and was the probable
    fix for the gateway 400 on image input and the "no audio attached" miss.

    HAND-ROLLED for video: langchain-openai has NO native video path —
    ``convert_to_openai_data_block`` raises ``ValueError("Block of type video
    is not supported")`` on ``type == "video"`` (block_translators/openai.py:149;
    ``is_data_content_block`` still returns True, so it routes there and dies).
    A native ``video`` block would ALWAYS crash at serialization, so the only
    shape that can attempt a whole-clip call is the provider-specific
    ``video_url`` data-URI. If the gateway rejects it, the failure surfaces as a
    caught exception (honest ``model_failed`` / mega keyframe fallback) — never
    a silent miss. A ``video_frame`` is a keyframe PNG → IMAGE → native block.

    Raises ``ValueError`` on an unsupported ``kind`` — a real bug, not a skip."""
    if kind in ("image", "video_frame"):
        return {"type": "image", "base64": b64, "mime_type": mime}
    if kind == "audio":
        return {"type": "audio", "base64": b64, "mime_type": mime}
    if kind == "video":
        return {"type": "video_url",
                "video_url": {"url": f"data:{mime};base64,{b64}"}}
    raise ValueError(f"unsupported media kind for content block: {kind!r}")


class _MediaNonAnswer(RuntimeError):
    """The model replied but signalled it didn't actually receive the media
    (e.g. 'no audio was attached'). Distinct from empty output: the model DID
    return text, just text that confesses the media never arrived. Treated as a
    Tier-1 failure so the default ``multimodal`` tool surfaces a
    ``model_non_answer`` reason and ``multimodal_mega`` falls through to its
    terminal tier — instead of returning a misleading success whose
    'description' is the model's own 'I got nothing'."""


# Substrings (lowercased) that strongly indicate the model did not receive an
# AUDIO clip — the known failure mode ("no audio was attached", returned 200
# but with no audio actually wired through). Curated for precision: a real
# description of a silent clip ("I hear only background hum") is NOT here.
_AUDIO_NON_ANSWER_PHRASES: tuple[str, ...] = (
    "no audio was attached", "no audio attached", "no audio provided",
    "no audio included", "no audio received", "no audio file",
    "no audio content", "no audio is present", "no audio present",
    "audio was not attached", "audio is not attached",
    "audio is missing", "audio was missing",
    "didn't receive any audio", "did not receive any audio",
    "no sound was provided", "no sound attached", "no sound included",
    "i don't hear any", "i cannot hear", "i can't hear",
    "could not hear any", "couldn't hear any", "unable to hear",
)


def _looks_like_audio_non_answer(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in _AUDIO_NON_ANSWER_PHRASES)


def _model_text_or_raise(resp: object, kind: str) -> str:
    """Extract the model's reply text; raise on empty or a media non-answer.

    Empty content → ``RuntimeError`` (a generic model failure). An audio
    non-answer → ``_MediaNonAnswer`` so callers can surface it distinctly.
    The non-answer guard is audio-scoped: image/video non-answers ('I don't
    see…') overlap too heavily with legitimate descriptions to detect
    reliably, and the known failure mode was specifically audio."""
    content = getattr(resp, "content", None)
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") for b in content if isinstance(b, dict))
    out = (content or "").strip() if isinstance(content, str) else ""
    if not out:
        raise RuntimeError("primary model returned empty content")
    if kind == "audio" and _looks_like_audio_non_answer(out):
        raise _MediaNonAnswer(out)
    return out


def _invoke_primary_media(
    model: object, b64: str, mime: str, prompt: str | None, kind: str,
) -> str:
    """Send the media to the multimodal model + return its text. Raises on any
    failure (caller falls back / surfaces an honest error).

    Builds the content block via ``_media_content_block`` — native for image +
    audio (the canonical ``read_file`` shape; ``ChatOpenAI`` does the provider
    wire translation downstream) and the provider-specific ``video_url``
    data-URI for video (no native video path exists). Reply text is vetted by
    ``_model_text_or_raise`` so a 'no audio attached'-style non-answer is
    treated as a failure, not a silent success."""
    text = prompt or _default_media_prompt(kind)
    block = _media_content_block(kind, b64, mime)
    msg = HumanMessage(content=[{"type": "text", "text": text}, block])
    resp = model.invoke([msg])
    return _model_text_or_raise(resp, kind)


def _onnx_describe(
    exec_client: DockerExecClient, *,
    image_path: str | None, image_url: str | None,
    prompt: str | None, primary_error: str | None = None,
) -> dict:
    """The in-sandbox ONNX fallback (Qwen3.5-2B-ONNX-OPT via
    ``describe_image.py``). Returns a RESULT DICT (caller wraps with
    ``_result``). Shared by ``describe_image`` and ``multimodal_mega`` so the
    two stay byte-equivalent on the ONNX path.

    Exit-code dispatch: 0=success, 2=model missing (NOT an error — the model is
    optional), 3=onnxruntime-genai absent. ``primary_error`` (set when the
    driving model was tried and failed) flips ``source`` from ``onnx`` to
    ``fallback`` and is echoed back so the fallback is observable."""
    pe = {"primary_error": _tail(primary_error, 300)} if primary_error else {}
    parts = [f"python3 {_DESCRIBE_IMAGE_SCRIPT}"]
    parts += ["--image", shlex.quote(image_path)] if image_path else ["--image-url", shlex.quote(image_url)]
    if prompt:
        parts += ["--prompt", shlex.quote(prompt)]
    cmd = " ".join(parts)
    try:
        out, exit_code = exec_client.exec(cmd, timeout=_DESCRIBE_IMAGE_TIMEOUT)
    except ExecTimeout:
        return {"success": False, "reason": "timeout",
                "error": f"describe_image timed out after {_DESCRIBE_IMAGE_TIMEOUT}s",
                **pe}
    except Exception as exc:  # container vanished / docker API error
        return {"success": False, "reason": "exec_failed", "error": str(exc), **pe}
    if exit_code == 0:
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            return {"success": False, "reason": "malformed_output",
                    "error": f"describe_image returned non-JSON: {_tail(out, 400)}",
                    **pe}
        return {"success": True,
                "description": parsed.get("description", ""),
                "model": parsed.get("model", ""),
                "source": "fallback" if primary_error else "onnx",
                **pe}
    if exit_code == 2:
        return {"success": False, "reason": "unavailable",
                "explanation": _VISION_UNAVAILABLE, "detail": _tail(out), **pe}
    if exit_code == 3:
        return {"success": False, "reason": "deps_missing",
                "explanation": _VISION_DEPS_MISSING, "detail": _tail(out), **pe}
    return {"success": False, "reason": "inference_failed", "error": _tail(out), **pe}


def _extract_video_keyframes(
    exec_client: DockerExecClient, video_path: str, n: int = _VIDEO_KEYFRAMES,
) -> tuple[list[str], str | None]:
    """Probe ``video_path`` (in-sandbox) and extract up to ``n`` evenly-spaced
    frames to ``/tmp/pux_multimodal_kf/kf_*.png`` via ffmpeg. Returns
    ``(frame_paths, None)`` on success or ``([], reason)`` on failure."""
    kf_dir = "/tmp/pux_multimodal_kf"
    exec_client.exec(f"rm -rf {kf_dir} && mkdir -p {kf_dir}", timeout=30)
    probe = ("ffprobe -v error -show_entries format=duration "
             "-of default=noprint_wrappers=1:nokey=1 " + shlex.quote(video_path))
    out, exit_code = exec_client.exec(probe, timeout=_KEYFRAME_TIMEOUT)
    if exit_code != 0:
        return [], "ffmpeg_missing" if exit_code == 127 else f"ffprobe_failed: {_tail(out, 200)}"
    try:
        duration = float((out or "").strip())
    except ValueError:
        return [], f"no_duration: {_tail(out, 120)}"
    if duration <= 0:
        return [], "empty_video"
    interval = max(1.0, duration / max(1, n))
    extract = (
        f"ffmpeg -hide_banner -loglevel error -i {shlex.quote(video_path)} "
        f"-vf fps={1 / interval:.4f} -frames:v {n} -y {kf_dir}/kf_%03d.png"
    )
    out, exit_code = exec_client.exec(extract, timeout=_KEYFRAME_TIMEOUT)
    if exit_code != 0:
        return [], "ffmpeg_extract_failed" if exit_code == 127 else f"ffmpeg_extract_failed: {_tail(out, 200)}"
    ls, _ = exec_client.exec(f"ls -1 {kf_dir}/*.png 2>/dev/null | sort", timeout=30)
    frames = [ln.strip() for ln in (ls or "").splitlines() if ln.strip()]
    if not frames:
        return [], "no_keyframes_extracted"
    return frames, None
