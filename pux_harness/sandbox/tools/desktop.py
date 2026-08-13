"""pux_sandbox_desktop_* — X11 desktop via xdotool + desktop_observe.py."""

from __future__ import annotations

from deepagents.backends.sandbox import BaseSandbox

import json
import shlex

from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

from ._shared import _tail, _result, _NoArgs, _exec


_DISPLAY_ENV = "DISPLAY=:99"
_DESKTOP_TIMEOUT = 15
_DESKTOP_OBSERVE = "/usr/local/bin/desktop_observe.py"


def _exec_desktop(sandbox: BaseSandbox, op: str, cmd: str,
                  *, timeout: int = _DESKTOP_TIMEOUT):
    """Run a desktop command; return ``(error_envelope | None, out, exit_code)``.
    On timeout / docker failure / non-zero exit, ``error_envelope`` is a
    ready-to-return JSON string (and the caller short-circuits). On success it
    is ``None`` and the caller synthesizes the result from ``out`` / its args."""
    try:
        out, exit_code = _exec(sandbox, cmd, timeout=timeout)
    except Exception as exc:
        return _result({"success": False, "reason": "exec_failed",
                        "error": f"desktop {op}: {exc}"}), "", 0
    if exit_code != 0:
        return _result({"success": False, "reason": "exec_failed",
                        "error": f"desktop {op}: exit {exit_code}",
                        "detail": _tail(out, 400)}), out, exit_code
    return None, out, exit_code


# --- screenshot -------------------------------------------------------------

_DESKTOP_SCREENSHOT_DESC = (
    "Capture the sandbox desktop (X11 DISPLAY=:99) as a base64 PNG with OCR-"
    "detected text elements + window list. Each element has cx/cy (center "
    "pixel coords) — pass those to desktop_click. Use to orient before "
    "clicking or to read on-screen text."
)


def _desktop_screenshot_tool(sandbox: BaseSandbox) -> StructuredTool:
    def _run() -> str:
        cmd = f"{_DISPLAY_ENV} python3 {_DESKTOP_OBSERVE}"
        err, out, _ = _exec_desktop(sandbox, "screenshot", cmd)
        if err is not None:
            return err
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            return _result({"success": False, "reason": "malformed_response",
                            "error": "desktop_screenshot: non-JSON output",
                            "detail": _tail(out, 400)})
        parsed["ok"] = False if parsed.get("error") else True
        return _result(parsed)

    return StructuredTool(
        name="desktop_screenshot", description=_DESKTOP_SCREENSHOT_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- click ------------------------------------------------------------------

_DESKTOP_CLICK_DESC = (
    "Click at pixel coordinates on the sandbox desktop. Pick coords from "
    "desktop_screenshot's element.cx/element.cy or the visible image. Optional "
    "button: 1=left (default), 2=middle, 3=right."
)


class _DesktopClickArgs(BaseModel):
    x: int = Field(..., description="X pixel coordinate (0 = left edge)")
    y: int = Field(..., description="Y pixel coordinate (0 = top edge)")
    button: int = Field(1, description="Mouse button: 1=left (default), 2=middle, 3=right")


def _desktop_click_tool(sandbox: BaseSandbox) -> StructuredTool:
    def _run(x: int, y: int, button: int = 1) -> str:
        if button < 1 or button > 3:
            return _result({"success": False, "error": "button must be 1, 2, or 3"})
        cmd = f"{_DISPLAY_ENV} xdotool mousemove --sync {x} {y} click {button}"
        err, _, _ = _exec_desktop(sandbox, "click", cmd)
        if err is not None:
            return err
        return _result({"ok": True, "x": x, "y": y, "button": button})

    return StructuredTool(
        name="desktop_click", description=_DESKTOP_CLICK_DESC,
        args_schema=_DesktopClickArgs, func=_run,
    )


# --- type -------------------------------------------------------------------

_DESKTOP_TYPE_DESC = (
    "Type text into the focused desktop window via xdotool. Optional clear "
    "(default true) Ctrl+A + Delete's existing field content first. Characters "
    "are sent as real X11 key events — works in any app."
)


class _DesktopTypeArgs(BaseModel):
    text: str = Field(..., description="Text to type")
    clear: bool = Field(True, description="Clear field first (default true)")


def _desktop_type_tool(sandbox: BaseSandbox) -> StructuredTool:
    def _run(text: str, clear: bool = True) -> str:
        if not text:
            return _result({"success": False, "error": "text is required"})
        parts = [_DISPLAY_ENV, "xdotool"]
        if clear:
            parts.append("key ctrl+a Delete")
        parts += ["type", "--clearmodifiers", shlex.quote(text)]
        cmd = " ".join(parts)
        err, _, _ = _exec_desktop(sandbox, "type", cmd)
        if err is not None:
            return err
        return _result({"ok": True, "text": text, "clear": clear})

    return StructuredTool(
        name="desktop_type", description=_DESKTOP_TYPE_DESC,
        args_schema=_DesktopTypeArgs, func=_run,
    )


# --- key --------------------------------------------------------------------

_DESKTOP_KEY_DESC = (
    "Press a key combo on the sandbox desktop via xdotool key. Examples: "
    "'Return', 'ctrl+c', 'alt+Tab', 'Escape', 'super'. For text input use "
    "desktop_type instead."
)


class _DesktopKeyArgs(BaseModel):
    keys: str = Field(..., description="xdotool key combo (e.g. 'Return', 'ctrl+c', 'alt+Tab')")


def _desktop_key_tool(sandbox: BaseSandbox) -> StructuredTool:
    def _run(keys: str) -> str:
        if not keys:
            return _result({"success": False, "error": "keys is required"})
        cmd = f"{_DISPLAY_ENV} xdotool key {shlex.quote(keys)}"
        err, _, _ = _exec_desktop(sandbox, "key", cmd)
        if err is not None:
            return err
        return _result({"ok": True, "keys": keys})

    return StructuredTool(
        name="desktop_key", description=_DESKTOP_KEY_DESC,
        args_schema=_DesktopKeyArgs, func=_run,
    )
