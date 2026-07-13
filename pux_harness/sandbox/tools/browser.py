"""pux_sandbox_browser_* — in-sandbox SeleniumBase Chrome via sb_server.py."""

from __future__ import annotations

import json
import os
import random
import shlex
import time

from pydantic import BaseModel, Field, field_validator

from langchain_core.tools import StructuredTool

from pux_harness.sandbox.docker_exec import DockerExecClient, ExecTimeout
from pux_harness.sandbox.tools._shared import PUX_PREFIX, _tail, _result, _NoArgs


_SB_SERVER_ADDR = "http://127.0.0.1:9876"
_BROWSER_TIMEOUT = 60

# Human-like pacing: random delay before each browser tool call so the
# action cadence looks natural to antibot services. 250-700ms mimics
# human reaction lag. Set PUX_BROWSER_MIN_PACING=0 to disable.
_PACING_MIN_MS = int(os.environ.get("PUX_BROWSER_MIN_PACING", "250"))
_PACING_MAX_MS = int(os.environ.get("PUX_BROWSER_MAX_PACING", "700"))


def _pace():
    """Sleep a random human-like amount before each browser tool call."""
    if _PACING_MIN_MS > 0:
        delay = random.uniform(_PACING_MIN_MS / 1000.0, _PACING_MAX_MS / 1000.0)
        time.sleep(delay)


def _sb_post(exec_client: DockerExecClient, endpoint: str, body_obj: dict | None,
             *, timeout: int = _BROWSER_TIMEOUT) -> str:
    """POST ``body_obj`` to the in-sandbox sb_server.py endpoint, return the
    parsed JSON re-serialized via ``_result``."""
    _pace()
    max_time = max(1, timeout)
    parts = [
        "curl -s -S",
        f"--max-time {max_time}",
        "-X POST",
        f"{_SB_SERVER_ADDR}{endpoint}",
        "-H 'Content-Type: application/json'",
    ]
    body = ""
    if body_obj is not None:
        body = json.dumps(body_obj)
        parts += ["-d", shlex.quote(body)]
    cmd = " ".join(parts)
    try:
        out, exit_code = exec_client.exec(cmd, timeout=timeout)
    except ExecTimeout:
        return _result({"success": False, "reason": "timeout",
                        "error": f"browser {endpoint}: timed out after {timeout}s"})
    except Exception as exc:
        return _result({"success": False, "reason": "exec_failed",
                        "error": f"browser {endpoint}: {exc}"})
    if exit_code != 0:
        return _result({"success": False, "reason": "exec_failed",
                        "error": f"browser {endpoint}: curl exit {exit_code}",
                        "detail": _tail(out, 400)})
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return _result({"success": False, "reason": "malformed_response",
                        "error": f"browser {endpoint}: non-JSON response",
                        "detail": _tail(out, 400)})
    return _result(parsed)


# --- navigate ---------------------------------------------------------------

_BROWSER_NAVIGATE_DESC = (
    "Open a URL in the sandbox's persistent Chrome. Returns page title, URL, "
    "text snippet, and a base64 screenshot with Set-of-Marks labels on "
    "interactive elements. The session persists — subsequent browser_click / "
    "browser_type / browser_screenshot calls operate on this page until you "
    "navigate again."
)


class _BrowserNavigateArgs(BaseModel):
    url: str = Field(..., description="Absolute URL including scheme (https://example.com)")


def _browser_navigate_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(url: str) -> str:
        if not url:
            return _result({"success": False, "error": "url is required"})
        return _sb_post(exec_client, "/navigate", {"url": url})

    return StructuredTool(
        name=PUX_PREFIX + "browser_navigate", description=_BROWSER_NAVIGATE_DESC,
        args_schema=_BrowserNavigateArgs, func=_run,
    )


# --- click ------------------------------------------------------------------

_BROWSER_CLICK_DESC = (
    "Click an element on the current page. Pass either a SoM label (integer "
    "from the labeled screenshot) or a CSS selector string. Returns the "
    "post-click page state (URL, title, screenshot). For anti-bot sites "
    "(Cloudflare Turnstile, behavioral fingerprinting) that ignore normal "
    "clicks, set trusted=true to drive the real OS cursor via CDP "
    "(isTrusted=true) — only when a normal click silently no-ops."
)


class _BrowserClickArgs(BaseModel):
    index: int | None = Field(None, description="SoM label (numbered box on interactive elements from the last screenshot)")
    selector: str | None = Field(None, description="CSS selector (e.g. 'button#submit'). Used when index is omitted.")
    trusted: bool = Field(False, description="Trusted input: drive the real cursor via CDP Input (isTrusted=true) so anti-bot defenses register a genuine click. Default false (Selenium click). Use only when a normal click silently no-ops on a protected site.")


def _browser_click_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(index: int | None = None, selector: str | None = None,
             trusted: bool = False) -> str:
        body: dict = {}
        if index is not None:
            body["index"] = index
        if selector is not None:
            body["selector"] = selector
        if trusted:
            body["trusted"] = True
        return _sb_post(exec_client, "/click", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_click", description=_BROWSER_CLICK_DESC,
        args_schema=_BrowserClickArgs, func=_run,
    )


# --- type -------------------------------------------------------------------

_BROWSER_TYPE_DESC = (
    "Type text into a form field on the current page. Uses CDP character-by-"
    "character input (React-safe — fires real DOM events). Pass either a SoM "
    "label or CSS selector to identify the target input. For anti-bot sites "
    "with keystroke-fingerprinting, set trusted=true to type via CDP "
    "Input.insertText (isTrusted input events)."
)


class _BrowserTypeArgs(BaseModel):
    text: str = Field(..., description="Text to type into the field")
    index: int | None = Field(None, description="SoM label of the target input")
    selector: str | None = Field(None, description="CSS selector of the target input")
    trusted: bool = Field(False, description="Trusted input: type via CDP Input.insertText (isTrusted input events) instead of the native value-setter. Default false. Use for keystroke-fingerprinting anti-bot defenses.")


def _browser_type_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(text: str, index: int | None = None, selector: str | None = None,
             trusted: bool = False) -> str:
        if not text:
            return _result({"success": False, "error": "text is required"})
        if index is None and not selector:
            return _result({"success": False, "error": "either index or selector is required"})
        body = {"text": text}
        if index is not None:
            body["index"] = index
        if selector is not None:
            body["selector"] = selector
        if trusted:
            body["trusted"] = True
        return _sb_post(exec_client, "/type", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_type", description=_BROWSER_TYPE_DESC,
        args_schema=_BrowserTypeArgs, func=_run,
    )


# --- screenshot -------------------------------------------------------------

_BROWSER_SCREENSHOT_DESC = (
    "Capture the current browser state as a labeled screenshot. Returns base64 "
    "PNG + SoM-numbered boxes on interactive elements. Use to re-orient after "
    "page updates, or to get fresh label numbers for clicking."
)


def _browser_screenshot_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/read", {})

    return StructuredTool(
        name=PUX_PREFIX + "browser_screenshot", description=_BROWSER_SCREENSHOT_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- evaluate ---------------------------------------------------------------

_BROWSER_EVALUATE_DESC = (
    "Evaluate JavaScript on the current page, return the result. Power-tool "
    "escape hatch when navigate/click/type/screenshot don't fit (e.g. read "
    "window.__NEXT_DATA__, scroll to an element, fetch XHR). Runs in the page "
    "context — same-origin policy applies.\n\n"
    "GOTCHAS (both cost real debugging time):\n"
    "1. OBJECT-LITERAL RETURNS NEED PARENS. `return {a: 1}` is a SYNTAX ERROR "
    "— the JS parser treats `{` as a block statement, not an object literal. "
    "Wrap object literals: `return ({a: 1, b: 2})`. Primitives/strings/arrays "
    "are fine unwrapped: `return document.title`, `return [1, 2, 3]`.\n"
    "2. PERSISTENT REPL STATE. This is a long-lived JS REPL, not a fresh "
    "context — `let`/`var` declarations, DOM mutations, and global assignments "
    "from one call PERSIST into the next. A variable you declared in call #1 "
    "is still there in call #5 (re-declaring it throws `Identifier has already "
    "been declared`). Wrap one-shot scripts in an IIFE to avoid leaking state: "
    "`(function(){ ... })()`. Or reuse the persistence deliberately — declare a "
    "counter once, increment it across calls."
)


class _BrowserEvaluateArgs(BaseModel):
    code: str = Field(
        ...,
        description=(
            "JavaScript to evaluate. Use `return <value>` for the result. "
            "CRITICAL: wrap object-literal returns in parens — "
            "`return ({a: 1})` — `return {a: 1}` is a syntax error "
            "(parsed as a block statement). The REPL persists across calls "
            "(variables + mutations leak); wrap one-shot scripts in an IIFE "
            "to isolate."
        ),
    )


def _browser_evaluate_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(code: str) -> str:
        if not code:
            return _result({"success": False, "error": "code is required"})
        return _sb_post(exec_client, "/evaluate", {"code": code})

    return StructuredTool(
        name=PUX_PREFIX + "browser_evaluate", description=_BROWSER_EVALUATE_DESC,
        args_schema=_BrowserEvaluateArgs, func=_run,
    )


# --- search -----------------------------------------------------------------

_BROWSER_SEARCH_DESC = (
    "Search the web via DuckDuckGo and land on the results page. Returns the "
    "same labeled screenshot + page state as browser_navigate (the engine builds "
    "the DuckDuckGo URL for you). Use as the ENTRY POINT when you have a query "
    "but no URL. After searching, read the returned screenshot, pick a result by "
    "its SoM label, and browser_click it to open."
)


class _BrowserSearchArgs(BaseModel):
    query: str = Field(..., description="Natural-language search query (the engine URL-encodes it)")


def _browser_search_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(query: str) -> str:
        if not query:
            return _result({"success": False, "error": "query is required"})
        return _sb_post(exec_client, "/search", {"query": query})

    return StructuredTool(
        name=PUX_PREFIX + "browser_search", description=_BROWSER_SEARCH_DESC,
        args_schema=_BrowserSearchArgs, func=_run,
    )


# --- scroll -----------------------------------------------------------------

_BROWSER_SCROLL_DESC = (
    "Scroll the current page to reveal more content, then return a fresh "
    "labeled screenshot of the newly-visible region. Pass direction='down' or "
    "'up' for a viewport-sized jump; or set amount to a pixel count (e.g. 800) "
    "for a precise scroll. Essential on long pages — interactive elements below "
    "the fold have NO SoM label until you scroll them into view."
)


class _BrowserScrollArgs(BaseModel):
    direction: str = Field("down", description="'down' or 'up' (viewport-sized); ignored when amount>0")
    amount: int = Field(0, description="Pixel count to scroll (sign follows direction). 0 = use direction for a viewport jump.")


def _browser_scroll_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(direction: str = "down", amount: int = 0) -> str:
        return _sb_post(exec_client, "/scroll", {"direction": direction, "amount": amount})

    return StructuredTool(
        name=PUX_PREFIX + "browser_scroll", description=_BROWSER_SCROLL_DESC,
        args_schema=_BrowserScrollArgs, func=_run,
    )


# --- go_back ----------------------------------------------------------------

_BROWSER_GO_BACK_DESC = (
    "Navigate back to the previous page in history. Returns the prior page's "
    "labeled screenshot. Use when a navigation took you somewhere unhelpful and "
    "you want to undo it without re-searching or re-typing a URL."
)


def _browser_go_back_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/go_back", {})

    return StructuredTool(
        name=PUX_PREFIX + "browser_go_back", description=_BROWSER_GO_BACK_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- wait -------------------------------------------------------------------

_BROWSER_WAIT_DESC = (
    "Pause for up to 30 seconds (server clamps; default 2) for async content to "
    "load, then return a fresh labeled screenshot. Use after navigate/click/type "
    "when the page is still loading or a JS render is in flight — a cheap way to "
    "let the DOM settle before re-reading. Prefer this over guessing that a "
    "screenshot is current."
)


class _BrowserWaitArgs(BaseModel):
    seconds: int = Field(2, description="How long to wait; server clamps to 30")


def _browser_wait_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(seconds: int = 2) -> str:
        return _sb_post(exec_client, "/wait", {"seconds": seconds})

    return StructuredTool(
        name=PUX_PREFIX + "browser_wait", description=_BROWSER_WAIT_DESC,
        args_schema=_BrowserWaitArgs, func=_run,
    )


# --- find_text --------------------------------------------------------------

_BROWSER_FIND_TEXT_DESC = (
    "Scroll to and highlight the first occurrence of the given text on the "
    "current page (uses window.find). Returns a fresh labeled screenshot centered "
    "on the match. Use to locate specific information in a long page faster than "
    "scanning the whole screenshot."
)


class _BrowserFindTextArgs(BaseModel):
    text: str = Field(..., description="Substring to locate on the page")


def _browser_find_text_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(text: str) -> str:
        if not text:
            return _result({"success": False, "error": "text is required"})
        return _sb_post(exec_client, "/find_text", {"text": text})

    return StructuredTool(
        name=PUX_PREFIX + "browser_find_text", description=_BROWSER_FIND_TEXT_DESC,
        args_schema=_BrowserFindTextArgs, func=_run,
    )


# --- extract ----------------------------------------------------------------

_BROWSER_EXTRACT_DESC = (
    "Extract structured text data from the current page: title, url, headings, "
    "paragraphs, lists, tables, and forms. The query is a free-text note of "
    "intent (defaults to 'extract all text content'). Returns {extracted:{...}}. "
    "Use to pull CLEAN text from an article or enumerate form fields, instead of "
    "OCR-ing the screenshot."
)


class _BrowserExtractArgs(BaseModel):
    query: str = Field("extract all text content", description="Free-text note of what you want (the engine extracts the same DOM structures regardless)")


def _browser_extract_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(query: str = "extract all text content") -> str:
        return _sb_post(exec_client, "/extract", {"query": query})

    return StructuredTool(
        name=PUX_PREFIX + "browser_extract", description=_BROWSER_EXTRACT_DESC,
        args_schema=_BrowserExtractArgs, func=_run,
    )


# --- extract_images ---------------------------------------------------------

_BROWSER_EXTRACT_IMAGES_DESC = (
    "List every <img> on the current page with its src + alt text. Returns "
    "{images:[{src,alt}], url}. Use to collect image URLs for downloading (pass "
    "a src to browser_download) or to inventory page media without parsing the "
    "screenshot."
)


def _browser_extract_images_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/extract_images", {})

    return StructuredTool(
        name=PUX_PREFIX + "browser_extract_images", description=_BROWSER_EXTRACT_IMAGES_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- save_screenshot --------------------------------------------------------

_BROWSER_SAVE_SCREENSHOT_DESC = (
    "Save the current page as a clean PNG file at the given path (e.g. "
    "/tmp/evidence.png). DISTINCT from browser_screenshot (which returns a "
    "base64 SoM-labeled view for ACTING on the page): this writes an archival "
    "image to disk for evidence, attachments, or later describe_image analysis. "
    "Returns {screenshot_path, url}."
)


class _BrowserSaveScreenshotArgs(BaseModel):
    path: str | None = Field(None, description="Absolute sandbox path incl. .png extension. If omitted the engine generates one and returns it.")


def _browser_save_screenshot_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(path: str | None = None) -> str:
        body: dict = {}
        if path:
            body["path"] = path
        return _sb_post(exec_client, "/screenshot", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_save_screenshot", description=_BROWSER_SAVE_SCREENSHOT_DESC,
        args_schema=_BrowserSaveScreenshotArgs, func=_run,
    )


# --- download ---------------------------------------------------------------

_BROWSER_DOWNLOAD_DESC = (
    "Download a file from a direct URL to a path inside the sandbox (e.g. "
    "/tmp/report.pdf). Both url and path are required. Returns {url, path, size}. "
    "Use for direct file URLs (discovered via browser_extract_images or link "
    "hrefs) — NOT for pages that require interaction to produce the file."
)


class _BrowserDownloadArgs(BaseModel):
    url: str = Field(..., description="Direct file URL to fetch")
    path: str = Field(..., description="Absolute sandbox output path (incl. extension)")


def _browser_download_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(url: str, path: str) -> str:
        if not url or not path:
            return _result({"success": False, "error": "url and path are both required"})
        return _sb_post(exec_client, "/download", {"url": url, "path": path})

    return StructuredTool(
        name=PUX_PREFIX + "browser_download", description=_BROWSER_DOWNLOAD_DESC,
        args_schema=_BrowserDownloadArgs, func=_run,
    )


# --- upload -----------------------------------------------------------------

_BROWSER_UPLOAD_DESC = (
    "Upload a local file into an <input type='file'> on the current page. "
    "Identify the input by CSS selector and pass a sandbox-absolute file_path "
    "(which must already exist). Returns {uploaded, selector, file}. Use to "
    "attach a resume/photo/document to a form whose upload UI can't be driven by "
    "browser_type."
)


class _BrowserUploadArgs(BaseModel):
    selector: str = Field(..., description="CSS selector of the <input type='file'>")
    file_path: str = Field(..., description="Absolute sandbox path of the file to upload (must exist)")


def _browser_upload_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(selector: str, file_path: str) -> str:
        if not selector or not file_path:
            return _result({"success": False, "error": "selector and file_path are both required"})
        return _sb_post(exec_client, "/upload", {"selector": selector, "file_path": file_path})

    return StructuredTool(
        name=PUX_PREFIX + "browser_upload", description=_BROWSER_UPLOAD_DESC,
        args_schema=_BrowserUploadArgs, func=_run,
    )


# --- tabs -------------------------------------------------------------------

_BROWSER_TABS_DESC = (
    "List all open browser tabs with their index, url, title, and which is "
    "active. Returns {tabs:[{index,url,title,active}]}. Use before "
    "browser_switch_tab to find the index of the tab you want, or to confirm how "
    "many tabs are open."
)


def _browser_tabs_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/tabs", {})

    return StructuredTool(
        name=PUX_PREFIX + "browser_tabs", description=_BROWSER_TABS_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- new_tab ----------------------------------------------------------------

_BROWSER_NEW_TAB_DESC = (
    "Open a new browser tab to the given URL (default about:blank) and switch to "
    "it. Returns the new tab's labeled screenshot. Use to open a link without "
    "losing the current page, or to compare pages side-by-side."
)


class _BrowserNewTabArgs(BaseModel):
    url: str = Field("about:blank", description="URL to open in the new tab")


def _browser_new_tab_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(url: str = "about:blank") -> str:
        return _sb_post(exec_client, "/new_tab", {"url": url})

    return StructuredTool(
        name=PUX_PREFIX + "browser_new_tab", description=_BROWSER_NEW_TAB_DESC,
        args_schema=_BrowserNewTabArgs, func=_run,
    )


# --- switch_tab -------------------------------------------------------------

_BROWSER_SWITCH_TAB_DESC = (
    "Switch to the browser tab at the given 0-based index. Returns that tab's "
    "labeled screenshot with fresh SoM labels. Use browser_tabs first to learn "
    "the index→url mapping."
)


class _BrowserSwitchTabArgs(BaseModel):
    index: int = Field(0, description="0-based tab index (see browser_tabs)")


def _browser_switch_tab_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(index: int = 0) -> str:
        return _sb_post(exec_client, "/switch_tab", {"index": index})

    return StructuredTool(
        name=PUX_PREFIX + "browser_switch_tab", description=_BROWSER_SWITCH_TAB_DESC,
        args_schema=_BrowserSwitchTabArgs, func=_run,
    )


# --- close_tab --------------------------------------------------------------

_BROWSER_CLOSE_TAB_DESC = (
    "Close the current browser tab and switch to the last remaining one (the "
    "engine refuses to close the final tab). Returns the now-active tab's "
    "labeled screenshot. Use to clean up after browser_new_tab."
)


def _browser_close_tab_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/close_tab", {})

    return StructuredTool(
        name=PUX_PREFIX + "browser_close_tab", description=_BROWSER_CLOSE_TAB_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- dropdown_options -------------------------------------------------------

_BROWSER_DROPDOWN_OPTIONS_DESC = (
    "Read the options of a <select> dropdown. Identify the select element by SoM "
    "label (index) or CSS selector. Returns {selector, options, multiple, "
    "selected_count}. Call BEFORE browser_select_dropdown to learn the available "
    "option values and visible text."
)


class _BrowserDropdownOptionsArgs(BaseModel):
    index: int | None = Field(None, description="SoM label of the <select> element")
    selector: str | None = Field(None, description="CSS selector of the <select> element")


def _browser_dropdown_options_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(index: int | None = None, selector: str | None = None) -> str:
        if index is None and not selector:
            return _result({"success": False, "error": "either index or selector is required"})
        body: dict = {}
        if index is not None:
            body["index"] = index
        if selector is not None:
            body["selector"] = selector
        return _sb_post(exec_client, "/dropdown_options", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_dropdown_options", description=_BROWSER_DROPDOWN_OPTIONS_DESC,
        args_schema=_BrowserDropdownOptionsArgs, func=_run,
    )


# --- select_dropdown --------------------------------------------------------

_BROWSER_SELECT_DROPDOWN_DESC = (
    "Choose an option in a <select> dropdown. Identify the select by SoM label "
    "(index) or CSS selector, then specify the option by its value attribute OR "
    "its visible text (exactly one). Returns the post-selection labeled "
    "screenshot. Use browser_dropdown_options first to discover the right value "
    "or text."
)


class _BrowserSelectDropdownArgs(BaseModel):
    index: int | None = Field(None, description="SoM label of the <select> element")
    selector: str | None = Field(None, description="CSS selector of the <select> element")
    value: str | None = Field(None, description="value attribute of the option to select (use XOR with text)")
    text: str | None = Field(None, description="Visible text of the option to select (use XOR with value)")


def _browser_select_dropdown_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(index: int | None = None, selector: str | None = None,
             value: str | None = None, text: str | None = None) -> str:
        if index is None and not selector:
            return _result({"success": False, "error": "either index or selector is required"})
        if value is None and text is None:
            return _result({"success": False, "error": "either value or text is required"})
        body: dict = {}
        if index is not None:
            body["index"] = index
        if selector is not None:
            body["selector"] = selector
        if value is not None:
            body["value"] = value
        if text is not None:
            body["text"] = text
        return _sb_post(exec_client, "/select_dropdown", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_select_dropdown", description=_BROWSER_SELECT_DROPDOWN_DESC,
        args_schema=_BrowserSelectDropdownArgs, func=_run,
    )


# --- save_session -----------------------------------------------------------

_BROWSER_SAVE_SESSION_DESC = (
    "Save the current browser session (cookies + localStorage) to a JSON file "
    "(default /tmp/browser-session.json). Returns {saved, path, cookies, "
    "storage_items}. Call AFTER logging into an auth-heavy site so a later run "
    "can browser_restore_session without re-authenticating."
)


class _BrowserSaveSessionArgs(BaseModel):
    path: str = Field("/tmp/browser-session.json", description="Absolute sandbox path to write the session JSON")


def _browser_save_session_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(path: str = "/tmp/browser-session.json") -> str:
        return _sb_post(exec_client, "/save_session", {"path": path})

    return StructuredTool(
        name=PUX_PREFIX + "browser_save_session", description=_BROWSER_SAVE_SESSION_DESC,
        args_schema=_BrowserSaveSessionArgs, func=_run,
    )


# --- restore_session --------------------------------------------------------

_BROWSER_RESTORE_SESSION_DESC = (
    "Restore a previously-saved browser session (cookies + localStorage) from a "
    "JSON file (default /tmp/browser-session.json). Returns {restored, path, "
    "cookies, storage_items}. Call right after browser_navigate to the site's "
    "domain, BEFORE other actions, to reuse saved auth."
)


class _BrowserRestoreSessionArgs(BaseModel):
    path: str = Field("/tmp/browser-session.json", description="Absolute sandbox path of a session JSON written by browser_save_session")


def _browser_restore_session_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(path: str = "/tmp/browser-session.json") -> str:
        return _sb_post(exec_client, "/restore_session", {"path": path})

    return StructuredTool(
        name=PUX_PREFIX + "browser_restore_session", description=_BROWSER_RESTORE_SESSION_DESC,
        args_schema=_BrowserRestoreSessionArgs, func=_run,
    )


# --- drag (SOTA drag-and-drop) ------------------------------------------------

_BROWSER_DRAG_DESC = (
    "Drag-and-drop an element on the current page — the gap this fills vs older "
    "browser_click/type tooling. Works for sortable lists (Kanban boards, "
    "SortableJS/react-dnd/dnd-kit), file drop-zones, sliders, and custom "
    "draggables. Identify the SOURCE with a SoM index, CSS selector, or x/y "
    "coords; identify the TARGET with a SoM index/selector, x/y coords, OR a "
    "dx/dy pixel offset (offset mode is how you nudge a slider thumb). strategy: "
    "'auto' (default) picks HTML5-event drag for genuinely draggable elements "
    "and mouse-physics otherwise; 'html5' forces the synthetic dragstart/drop "
    "chain (best for sortable lists); 'physics' forces mousedown→mousemove(N)"
    "→mouseup (best for sliders/canvas). ALWAYS verify the result in the "
    "returned screenshot — if 'auto' picked wrong, retry with the other strategy."
)


class _BrowserDragArgs(BaseModel):
    from_index: int | None = Field(None, description="SoM label of the drag source")
    from_selector: str | None = Field(None, description="CSS selector of the drag source")
    from_x: float | None = Field(None, description="x coord of the drag source (use instead of index/selector)")
    from_y: float | None = Field(None, description="y coord of the drag source")
    to_index: int | None = Field(None, description="SoM label of the drop target")
    to_selector: str | None = Field(None, description="CSS selector of the drop target")
    to_x: float | None = Field(None, description="x coord of the drop target")
    to_y: float | None = Field(None, description="y coord of the drop target")
    dx: float | None = Field(None, description="drop this many px right(+) of the source (offset mode — sliders). Use XOR with to_*.")
    dy: float | None = Field(None, description="drop this many px down(+) of the source (offset mode — sliders)")
    strategy: str = Field("auto", description="'auto' | 'html5' | 'physics'")
    steps: int = Field(25, description="mouse-move interpolation steps for the physics path (ignored by html5)")


def _browser_drag_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(from_index: int | None = None, from_selector: str | None = None,
             from_x: float | None = None, from_y: float | None = None,
             to_index: int | None = None, to_selector: str | None = None,
             to_x: float | None = None, to_y: float | None = None,
             dx: float | None = None, dy: float | None = None,
             strategy: str = "auto", steps: int = 25) -> str:
        has_src = from_index is not None or from_selector or from_x is not None
        has_tgt = (to_index is not None or to_selector or to_x is not None
                   or dx is not None or dy is not None)
        if not has_src:
            return _result({"success": False, "error": "drag needs a source: from_index/from_selector OR from_x/from_y"})
        if not has_tgt:
            return _result({"success": False, "error": "drag needs a target: to_index/to_selector, to_x/to_y, or dx/dy"})
        body: dict = {"strategy": strategy, "steps": steps}
        if from_index is not None:
            body["from_index"] = from_index
        if from_selector:
            body["from_selector"] = from_selector
        if from_x is not None:
            body["from_x"] = from_x
        if from_y is not None:
            body["from_y"] = from_y
        if to_index is not None:
            body["to_index"] = to_index
        if to_selector:
            body["to_selector"] = to_selector
        if to_x is not None:
            body["to_x"] = to_x
        if to_y is not None:
            body["to_y"] = to_y
        if dx is not None:
            body["dx"] = dx
        if dy is not None:
            body["dy"] = dy
        return _sb_post(exec_client, "/drag", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_drag", description=_BROWSER_DRAG_DESC,
        args_schema=_BrowserDragArgs, func=_run,
    )


# --- hover ------------------------------------------------------------------

_BROWSER_HOVER_DESC = (
    "Hover the mouse over an element on the current page (dispatches "
    "mouseover/mousemove/mouseenter). Use to reveal dropdown menus, tooltips, "
    "fly-out panels, and hover-cards that only appear on mouseover — often a "
    "required precursor to clicking a menu item that has no SoM label until the "
    "menu opens. Identify the target with a SoM index, CSS selector, or x/y "
    "coords. Returns a fresh labeled screenshot so you can see what the hover "
    "revealed."
)


class _BrowserHoverArgs(BaseModel):
    index: int | None = Field(None, description="SoM label of the element to hover")
    selector: str | None = Field(None, description="CSS selector of the element to hover")
    x: float | None = Field(None, description="x coord to hover (use instead of index/selector)")
    y: float | None = Field(None, description="y coord to hover")


def _browser_hover_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(index: int | None = None, selector: str | None = None,
             x: float | None = None, y: float | None = None) -> str:
        has_el = index is not None or selector
        if not has_el and (x is None or y is None):
            return _result({"success": False, "error": "hover needs index/selector OR x,y"})
        body: dict = {}
        if index is not None:
            body["index"] = index
        if selector:
            body["selector"] = selector
        if x is not None:
            body["x"] = x
        if y is not None:
            body["y"] = y
        return _sb_post(exec_client, "/hover", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_hover", description=_BROWSER_HOVER_DESC,
        args_schema=_BrowserHoverArgs, func=_run,
    )


# --- press (keys / hotkeys) -------------------------------------------------

_BROWSER_PRESS_DESC = (
    "Press a key or hotkey combination on the current page (dispatches "
    "keydown/keypress/keyup with modifier flags). Examples: 'Enter', 'Escape', "
    "'Tab', 'ArrowDown', 'Control+a', 'Shift+ArrowDown', 'Control+Enter'. Use "
    "for non-character keys browser_type can't send — to submit/dismiss, move a "
    "slider thumb with arrow keys, select-all, copy/paste, navigate comboboxes "
    "and menus by keyboard, or close a modal. Optionally target a SoM "
    "index/selector to focus it first; otherwise the currently-focused element "
    "receives the press."
)


class _BrowserPressArgs(BaseModel):
    keys: str = Field(..., description="Key or '+'-joined combo, e.g. 'Enter', 'Control+a', 'ArrowDown'. Modifiers: Control/Shift/Alt/Cmd.")
    index: int | None = Field(None, description="SoM label of the element to focus before pressing")
    selector: str | None = Field(None, description="CSS selector of the element to focus before pressing")


def _browser_press_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(keys: str, index: int | None = None, selector: str | None = None) -> str:
        if not keys:
            return _result({"success": False, "error": "keys is required"})
        body: dict = {"keys": keys}
        if index is not None:
            body["index"] = index
        if selector:
            body["selector"] = selector
        return _sb_post(exec_client, "/press", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_press", description=_BROWSER_PRESS_DESC,
        args_schema=_BrowserPressArgs, func=_run,
    )


# --- click_at (coords / right / double) -------------------------------------

_BROWSER_CLICK_AT_DESC = (
    "Click at exact pixel coordinates on the current page — the vision-grounded "
    "click. Use when a target has no SoM label and no clean selector (a canvas, "
    "a chart point, an image map, a custom-drawn button), so you must click a "
    "screen position from the screenshot. Also covers right-click (open a "
    "context menu: right=true) and double-click (double=true). If you pass a "
    "SoM index/selector instead of coords, the engine resolves it to the "
    "element's center and clicks there. Returns the post-click labeled "
    "screenshot."
)


class _BrowserClickAtArgs(BaseModel):
    x: float | None = Field(None, description="x coord to click (omit to use index/selector center)")
    y: float | None = Field(None, description="y coord to click")
    index: int | None = Field(None, description="SoM label whose center to click (alternative to x,y)")
    selector: str | None = Field(None, description="CSS selector whose center to click")
    button: int = Field(0, description="mouse button: 0=left (default), 1=middle, 2=right")
    double: bool = Field(False, description="true → double-click")
    right: bool = Field(False, description="true → right-click (dispatches contextmenu)")
    trusted: bool = Field(False, description="Trusted input: drive the real cursor via CDP Input (isTrusted=true). Default false. Use for anti-bot sites that ignore synthetic clicks.")


def _browser_click_at_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(x: float | None = None, y: float | None = None,
             index: int | None = None, selector: str | None = None,
             button: int = 0, double: bool = False, right: bool = False,
             trusted: bool = False) -> str:
        has_target = x is not None or y is not None or index is not None or selector
        if not has_target:
            return _result({"success": False, "error": "click_at needs x,y OR index/selector"})
        body: dict = {"button": button, "double": double, "right": right}
        if x is not None:
            body["x"] = x
        if y is not None:
            body["y"] = y
        if index is not None:
            body["index"] = index
        if selector:
            body["selector"] = selector
        if trusted:
            body["trusted"] = True
        return _sb_post(exec_client, "/click_at", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_click_at", description=_BROWSER_CLICK_AT_DESC,
        args_schema=_BrowserClickAtArgs, func=_run,
    )


# --- scroll_into_view -------------------------------------------------------

_BROWSER_SCROLL_INTO_VIEW_DESC = (
    "Scroll a specific element into the visible viewport, centered, then return "
    "a fresh labeled screenshot. Use BEFORE clicking/typing an element that you "
    "know exists (by index/selector) but is off-screen and so has no SoM label. "
    "Distinct from browser_scroll (viewport jump / pixel scroll) — this targets "
    "one element. After it returns, the element's SoM label is fresh and "
    "clickable."
)


class _BrowserScrollIntoViewArgs(BaseModel):
    index: int | None = Field(None, description="SoM label of the element to bring into view")
    selector: str | None = Field(None, description="CSS selector of the element to bring into view")


def _browser_scroll_into_view_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(index: int | None = None, selector: str | None = None) -> str:
        if index is None and not selector:
            return _result({"success": False, "error": "either index or selector is required"})
        body: dict = {}
        if index is not None:
            body["index"] = index
        if selector:
            body["selector"] = selector
        return _sb_post(exec_client, "/scroll_into_view", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_scroll_into_view", description=_BROWSER_SCROLL_INTO_VIEW_DESC,
        args_schema=_BrowserScrollIntoViewArgs, func=_run,
    )


# --- a11y (accessibility tree) ----------------------------------------------

_BROWSER_A11Y_DESC = (
    "Read the current page as an accessibility tree: a compact list of "
    "{role, name, selector} for every interactive element. Far cheaper to "
    "reason over than a screenshot when the page is dense (a long form, a data "
    "table, a settings panel) — use it alongside browser_screenshot to find the "
    "right SoM label or selector by role/name ('button Submit', 'textbox Email') "
    "instead of scanning the image. Returns {items:[...], total}. The selectors "
    "are usable directly by browser_click / browser_type."
)


def _browser_a11y_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/a11y", {})

    return StructuredTool(
        name=PUX_PREFIX + "browser_a11y", description=_BROWSER_A11Y_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- iframe -----------------------------------------------------------------

_BROWSER_IFRAME_DESC = (
    "Act on elements inside <iframe>s on the current page. Many sites embed "
    "CAPTCHAs, payment forms, rich-text editors, and widgets in iframes — their "
    "contents are invisible to browser_click/type on the top page. "
    "action='list' enumerates iframes (index/name/id/src). "
    "action='click' clicks an element INSIDE a same-origin iframe: pass the "
    "iframe as index/selector and inner_selector for the in-frame target. "
    "action='evaluate' runs JS inside a same-origin iframe (pass code). "
    "Cross-origin iframes are blocked by same-origin policy — the tool returns a "
    "clear error for those (they need provider-level handling). The legacy "
    "'enter'/'exit' frame-switch actions are RETIRED (CDP has no global frame "
    "context like WebDriver's switch_to); use 'click'/'evaluate' instead."
)


class _BrowserIframeArgs(BaseModel):
    action: str = Field("list", description="'list' | 'click' | 'evaluate' (legacy 'enter'/'exit' retired)")
    index: int | None = Field(None, description="SoM label of the iframe element")
    selector: str | None = Field(None, description="CSS selector of the iframe element")
    inner_selector: str | None = Field(None, description="action='click': CSS selector inside the iframe of the element to click")
    code: str | None = Field(None, description="action='evaluate': JS to run inside the iframe (use 'return' for a value)")


def _browser_iframe_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(action: str = "list", index: int | None = None, selector: str | None = None,
             inner_selector: str | None = None, code: str | None = None) -> str:
        body: dict = {"action": action}
        if index is not None:
            body["index"] = index
        if selector:
            body["selector"] = selector
        if inner_selector:
            body["inner_selector"] = inner_selector
        if code:
            body["code"] = code
        return _sb_post(exec_client, "/iframe", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_iframe", description=_BROWSER_IFRAME_DESC,
        args_schema=_BrowserIframeArgs, func=_run,
    )


# --- UC mode (Cloudflare / Turnstile / hCaptcha bypass) ---------------------

_BROWSER_UC_DESC = (
    "Bypass Cloudflare Turnstile, hCaptcha, and reCAPTCHA on pages the persistent "
    "Chrome can't pass. Spawns a dedicated SeleniumBase SB(uc=True) Chrome and calls "
    "uc_gui_click_captcha — a REAL pyautogui mouse click on the checkbox, the only "
    "reliable way past cross-origin captcha iframes (CDP/JS clicks are blocked by SOP "
    "and lack the isTrusted=true signal anti-bot scripts require).\n\n"
    "WHEN TO USE: after browser_navigate, if the page shows 'Just a moment', 'Checking "
    "your browser', 'Verify you are human', a Cloudflare/hCaptcha challenge, OR if "
    "browser_solve_captcha returned captcha_solved=false. Also use pre-emptively on "
    "sites known to cage job applications behind Turnstile (Workday, some Greenhouse).\n\n"
    "LIFECYCLE: action=open creates a persistent UC session you can follow with "
    "action=click / type / read / evaluate on the SAME CF-cleared page, then "
    "action=close to tear it down. Cookies (including cf_clearance) are auto-handed "
    "off to the persistent browser so later browser_navigate calls to the same domain "
    "inherit the cleared state.\n\n"
    "ACTIONS:\n"
    "  open    {url, click_captcha=true, handoff=true} — open in UC Chrome, click any "
    "          captcha, hand cookies to persistent browser. Returns page data + cf_cleared.\n"
    "  click   {selector?, text?} — click in the UC session.\n"
    "  type    {selector, text, submit=false, clear=true} — type in the UC session.\n"
    "  read    {} — re-read the UC page (title, url, text, screenshot).\n"
    "  evaluate {code} — run JS in the UC session.\n"
    "  cookies {cookie_action='get'|'inject_persistent'} — cookie management.\n"
    "  close   {} — tear down the UC session."
)


class _BrowserUcArgs(BaseModel):
    action: str = Field("open", description="open|click|type|read|evaluate|cookies|close")
    url: str | None = Field(None, description="URL for action=open")
    click_captcha: bool | None = Field(True, description="action=open: run uc_gui_click_captcha")
    handoff: bool | None = Field(True, description="action=open: hand cookies to persistent browser")
    selector: str | None = Field(None, description="action=click/type: CSS selector")
    text: str | None = Field(None, description="action=click: link text | action=type: text to type")
    by: str | None = Field("css", description="action=click: selector strategy (css|text|xpath)")
    submit: bool | None = Field(False, description="action=type: press Enter after typing")
    clear: bool | None = Field(True, description="action=type: clear field first")
    code: str | None = Field(None, description="action=evaluate: JS to run")
    cookie_action: str | None = Field("get", description="action=cookies: get|inject_persistent")


def _browser_uc_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(action: str = "open", url: str | None = None, click_captcha: bool | None = True,
             handoff: bool | None = True, selector: str | None = None, text: str | None = None,
             by: str | None = "css", submit: bool | None = False, clear: bool | None = True,
             code: str | None = None, cookie_action: str | None = "get") -> str:
        body: dict = {"action": action}
        if url is not None:
            body["url"] = url
        if click_captcha is not None:
            body["click_captcha"] = click_captcha
        if handoff is not None:
            body["handoff"] = handoff
        if selector is not None:
            body["selector"] = selector
        if text is not None:
            body["text"] = text
        if by is not None:
            body["by"] = by
        if submit is not None:
            body["submit"] = submit
        if clear is not None:
            body["clear"] = clear
        if code is not None:
            body["code"] = code
        if cookie_action is not None:
            body["cookie_action"] = cookie_action
        # UC open can take ~30-60s (spawns Chrome + captcha click). read/click
        # are fast. Scale the curl max-time to the action.
        timeout = 150 if action == "open" else _BROWSER_TIMEOUT
        return _sb_post(exec_client, "/uc", body, timeout=timeout)
    return StructuredTool(
        name=PUX_PREFIX + "browser_uc", description=_BROWSER_UC_DESC,
        args_schema=_BrowserUcArgs, func=_run,
    )


# --- accept cookies (GDPR / CCPA banner dismissal) --------------------------

_BROWSER_ACCEPT_COOKIES_DESC = (
    "Dismiss GDPR/CCPA cookie-consent banners ('We use cookies', 'Accept all', "
    "'Manage preferences' popups). Call this IMMEDIATELY after browser_navigate lands "
    "on a site where a cookie banner is visible — before doing anything else on the "
    "page. Banners block the underlying UI and waste your turns if you try to SoM-click "
    "around them.\n\n"
    "Uses a curated selector list covering OneTrust, TrustArc, CookieBot, Quantcast, "
    "Didomi, SourcePoint, and BBC/legacy banners, plus a text-based fallback (matches "
    "'Accept all', 'I agree', 'Got it', etc.). More reliable than SoM-guessing: these "
    "banners often render in containers the SoM labeler skips.\n\n"
    "Returns cookies_accepted=true with the method/selector that worked, or "
    "cookies_accepted=false with method='no-banner-found' (which is HONEST — banners "
    "are geo-targeted; a US-egress browser won't see EU GDPR banners, and that's fine)."
)


class _BrowserAcceptCookiesArgs(BaseModel):
    pass


def _browser_accept_cookies_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/accept_cookies", {})
    return StructuredTool(
        name=PUX_PREFIX + "browser_accept_cookies", description=_BROWSER_ACCEPT_COOKIES_DESC,
        args_schema=_BrowserAcceptCookiesArgs, func=_run,
    )


# --- warmup history (fingerprint legitimacy) --------------------------------

_BROWSER_WARMUP_HISTORY_DESC = (
    "Build browsing-history legitimacy before a sensitive task (job applications, "
    "account login). Visits benign high-traffic sites (Wikipedia, Hacker News, GitHub, "
    "Stack Overflow) with realistic dwell times + scroll, so the browser's history + "
    "cookie jar + TLS fingerprint look like a real user rather than a fresh automation "
    "session that went straight about:blank → target-site.\n\n"
    "WHEN TO USE: ONCE at the start of a session that will touch sensitive targets "
    "(LinkedIn login, Workday applications, Twitter). Combats 'fresh automation' "
    "heuristics. Don't overuse — it burns ~15-30s.\n\n"
    "Optional: pass urls=[...] to customize the site list, dwell=N (seconds per site, "
    "default 3.0). Returns visited count + per-site status."
)


class _BrowserWarmupHistoryArgs(BaseModel):
    urls: list[str] | None = Field(None, description="Custom site list (default: 6 benign sites)")
    dwell: float | None = Field(3.0, description="Seconds per site (randomized ±50%)")

    @field_validator("urls", mode="before")
    @classmethod
    def _coerce_urls(cls, v):
        """Models frequently pass a JSON array as a STRING ('[\"a\",\"b\"]')
        rather than a real array, and Pydantic's strict list[str] rejects it
        with "Input should be a valid list" — costing the agent 3-4 retry
        turns before it gives up. Accept str | list, JSON-decode strings,
        comma-split fallback, drop blanks. Returns None for empty input so
        the server uses its default site list."""
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                v = json.loads(s)
            except json.JSONDecodeError:
                v = [u.strip() for u in s.split(",")]
        if not isinstance(v, list):
            return None
        out = [str(u).strip() for u in v if str(u).strip()]
        return out or None


def _browser_warmup_history_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(urls: list[str] | None = None, dwell: float | None = 3.0) -> str:
        body: dict = {}
        if urls is not None:
            body["urls"] = urls
        if dwell is not None:
            body["dwell"] = dwell
        # 6 sites × ~4s dwell = ~30s budget
        timeout = max(_BROWSER_TIMEOUT, int((len(urls) if urls else 6) * (dwell or 3) + 20))
        return _sb_post(exec_client, "/warmup_history", body, timeout=timeout)
    return StructuredTool(
        name=PUX_PREFIX + "browser_warmup_history", description=_BROWSER_WARMUP_HISTORY_DESC,
        args_schema=_BrowserWarmupHistoryArgs, func=_run,
    )


# --- solve captcha (honest best-effort on persistent browser) ----------------

_BROWSER_SOLVE_CAPTCHA_DESC = (
    "Best-effort captcha click on the persistent Chrome via JS. HONEST: verifies "
    "whether a challenge is still on screen after the attempt and returns "
    "captcha_solved=false when it can't pass (cross-origin Cloudflare/hCaptcha iframes "
    "cannot be clicked via CDP — SOP blocks access and the click lacks isTrusted).\n\n"
    "Use this FIRST for simple in-page captchas (old-style reCAPTCHA checkboxes that "
    "aren't in a cross-origin iframe). If it returns captcha_solved=false with a hint, "
    "switch to browser_uc (the SB(uc=True) + uc_gui_click_captcha path) which can pass "
    "cross-origin Turnstile/hCaptcha via real pyautogui clicks."
)


class _BrowserSolveCaptchaArgs(BaseModel):
    pass


def _browser_solve_captcha_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/solve_captcha", {})
    return StructuredTool(
        name=PUX_PREFIX + "browser_solve_captcha", description=_BROWSER_SOLVE_CAPTCHA_DESC,
        args_schema=_BrowserSolveCaptchaArgs, func=_run,
    )
