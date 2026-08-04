"""Exec-dependent context tools — the gap-1 + gap-2 surface.

Four tools that need a ``DockerExecClient`` (the sandbox bridge) and are only
built when ``build_context_tools`` receives one. They close the parity gap with
context-mode's ctx_execute / ctx_execute_file / ctx_batch_execute /
ctx_fetch_and_index.

The command-building logic is split into PURE functions (``_build_exec_command``,
``_build_file_command``, ``_HTML_TO_TEXT_CODE``) so tests can verify the exact
shell string generated for each language WITHOUT a running Docker container.
The tool closures call these builders then hand the command to ``exec_client``.

Design rules (no hacks):
- No shelling out to the host — every command runs inside the sandbox container
  via ``exec_client.exec()``.
- No HTML parser dependency — the HTML-to-text converter is stdlib-only
  (``html.parser``) and runs as ``python3 -c`` INSIDE the container, piped from
  curl. The host process never touches network.
- TTL cache is a real blob lookup (``EventStore.get_blob_by_tool``), not an
  in-memory dict that evaporates on restart.
"""
from __future__ import annotations

import json
import shlex
import textwrap
import uuid

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from pux_harness.context.events import EventStore
from pux_harness.sandbox.docker_exec import ExecTimeout


# -- language → command maps (pure, testable without Docker) -----------------

# Interpreted languages: runtime + flag for inline code (-c / -e / -r).
_INTERPRETED: dict[str, tuple[str, str]] = {
    "python": ("python3", "-c"),
    "python3": ("python3", "-c"),
    "py": ("python3", "-c"),
    "javascript": ("node", "-e"),
    "js": ("node", "-e"),
    "typescript": ("node", "-e"),
    "ts": ("node", "-e"),
    "shell": ("bash", "-c"),
    "bash": ("bash", "-c"),
    "sh": ("sh", "-c"),
    "ruby": ("ruby", "-e"),
    "rb": ("ruby", "-e"),
    "perl": ("perl", "-e"),
    "php": ("php", "-r"),
    "elixir": ("elixir", "-e"),
    "ex": ("elixir", "-e"),
    "r": ("Rscript", "-e"),
}

# Compiled languages: file extension + a function that builds the run command
# from a temp filename inside the container.
_COMPILED: dict[str, tuple[str, object]] = {
    # go: `go run FILE` handles compilation internally
    "go": (".go", lambda f: f"go run {shlex.quote(f)}"),
    # rust: compile to a binary, then run it
    "rust": (".rs", lambda f: f"rustc -O {shlex.quote(f)} -o {shlex.quote(f + '.bin')} && {shlex.quote(f + '.bin')}"),
    "rs": (".rs", lambda f: f"rustc -O {shlex.quote(f)} -o {shlex.quote(f + '.bin')} && {shlex.quote(f + '.bin')}"),
    # csharp: `dotnet script FILE` (requires dotnet-sdk + dotnet-script global tool)
    "csharp": (".csx", lambda f: f"dotnet script {shlex.quote(f)}"),
    "cs": (".csx", lambda f: f"dotnet script {shlex.quote(f)}"),
}

# Languages that support file reading for ctx_execute_file. Maps language → a
# snippet that reads the file into a FILE_CONTENT variable in that language.
_FILE_READERS: dict[str, str] = {
    "python": 'import pathlib; FILE_CONTENT = pathlib.Path({path_r}).read_text()',
    "py": 'import pathlib; FILE_CONTENT = pathlib.Path({path_r}).read_text()',
    "python3": 'import pathlib; FILE_CONTENT = pathlib.Path({path_r}).read_text()',
    "javascript": 'const FILE_CONTENT = require("fs").readFileSync({path_j}, "utf8");',
    "js": 'const FILE_CONTENT = require("fs").readFileSync({path_j}, "utf8");',
    "typescript": 'const FILE_CONTENT = require("fs").readFileSync({path_j}, "utf8");',
    "ts": 'const FILE_CONTENT = require("fs").readFileSync({path_j}, "utf8");',
    "shell": 'FILE_CONTENT=$(cat {path_q})',
    "bash": 'FILE_CONTENT=$(cat {path_q})',
    "sh": 'FILE_CONTENT=$(cat {path_q})',
    "ruby": 'FILE_CONTENT = File.read({path_r})',
    "rb": 'FILE_CONTENT = File.read({path_r})',
    "perl": 'open(my $fh, "<", {path_q}); local $/; $FILE_CONTENT = <$fh>;',
    "php": '$FILE_CONTENT = file_get_contents({path_q});',
}


def _supported_languages() -> list[str]:
    """All recognized language keys (interpreted + compiled), sorted."""
    return sorted(set(_INTERPRETED) | set(_COMPILED))


def _build_exec_command(language: str, code: str) -> str:
    """Build the shell command to run ``code`` in ``language`` inside the
    sandbox container.

    Interpreted languages use ``runtime -c/-e CODE`` (inline). Compiled
    languages write the code to a temp file via a heredoc, then compile + run.
    Raises ``ValueError`` for an unrecognized language (the tool surfaces this
    as a clean error message listing the supported set).
    """
    lang = language.lower().strip()
    if lang in _INTERPRETED:
        runtime, flag = _INTERPRETED[lang]
        return f"{runtime} {flag} {shlex.quote(code)}"
    if lang in _COMPILED:
        ext, runner = _COMPILED[lang]
        fname = f"/tmp/ctx_exec_{uuid.uuid4().hex[:8]}{ext}"
        heredoc = f"cat > {fname} <<'PUX_EOF'\n{code}\nPUX_EOF\n"
        return heredoc + runner(fname)
    raise ValueError(
        f"unsupported language {language!r}. Supported: {', '.join(_supported_languages())}"
    )


def _build_file_command(path: str, language: str, code: str) -> str:
    """Build the command for ctx_execute_file: read ``path`` into FILE_CONTENT,
    then run ``code`` which can reference it. The file bytes never enter the
    agent's context — only stdout does.

    For languages with a known file-reader snippet, the reader is prepended to
    the code. For languages without (elixir, r, go, rust, csharp), the tool
    returns an error rather than silently dropping FILE_CONTENT.
    """
    lang = language.lower().strip()
    reader = _FILE_READERS.get(lang)
    if reader is None:
        raise ValueError(
            f"ctx_execute_file does not support {language!r} (no FILE_CONTENT "
            f"reader). Supported: {', '.join(sorted(_FILE_READERS))}"
        )
    # Format the reader with the path in the language's native quoting style.
    reader_code = reader.format(
        path_r=repr(path),       # Python / Ruby string literal
        path_j=json.dumps(path),  # JS/TS string literal
        path_q=shlex.quote(path), # shell literal
    )
    full_code = reader_code + "\n" + code
    return _build_exec_command(lang, full_code)


# -- HTML → text converter (stdlib-only, runs inside the container) -----------

_HTML_TO_TEXT_CODE = textwrap.dedent("""\
    import sys, re, html.parser
    class _S(html.parser.HTMLParser):
        def __init__(self):
            super().__init__(); self.parts = []; self._skip = False
        def handle_starttag(self, tag, attrs):
            if tag in ('script', 'style', 'nav', 'svg', 'noscript', 'iframe'):
                self._skip = True
            if tag in ('p', 'div', 'br', 'li', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                self.parts.append('\\n')
        def handle_endtag(self, tag):
            if tag in ('script', 'style', 'nav', 'svg', 'noscript', 'iframe'):
                self._skip = False
            if tag in ('p', 'div', 'li', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                self.parts.append('\\n')
        def handle_data(self, data):
            if not self._skip:
                self.parts.append(data)
    p = _S(); p.feed(sys.stdin.read())
    text = ''.join(p.parts)
    text = re.sub(r'\\n{3,}', '\\n\\n', text)
    text = re.sub(r'[ \\t]+', ' ', text)
    sys.stdout.write(text.strip())
""")


def _build_fetch_command(url: str) -> str:
    """curl the URL, pipe through the HTML-to-text converter. The converter is
    a no-op for plain text / JSON (html.parser leaves non-HTML input unchanged).
    Runs as one exec call so the network round-trip stays inside the container."""
    return f"curl -sL {shlex.quote(url)} | python3 -c {shlex.quote(_HTML_TO_TEXT_CODE)}"


# -- TTL cache default -------------------------------------------------------

_DEFAULT_TTL_S = 86_400  # 24 hours in seconds


# -- pydantic schemas --------------------------------------------------------

class _ExecuteArgs(BaseModel):
    language: str = Field(..., description=(
        "One of: " + ", ".join(_supported_languages()) + "."
    ))
    code: str = Field(..., description="Source code to execute. Print output is captured and returned.")


class _ExecuteFileArgs(BaseModel):
    path: str = Field(..., description="Absolute path to the file to read into FILE_CONTENT.")
    language: str = Field(..., description=(
        "Language with FILE_CONTENT support: " + ", ".join(sorted(_FILE_READERS)) + "."
    ))
    code: str = Field(..., description=(
        "Code to run over FILE_CONTENT (the file's text). Only stdout enters context."
    ))


class _BatchCommand(BaseModel):
    command: str = Field(..., description="Shell command to run inside the sandbox.")
    label: str = Field("", description="Optional label for the output section header.")


class _BatchArgs(BaseModel):
    commands: list[_BatchCommand] = Field(..., description="Shell commands to run sequentially.")
    queries: list[str] = Field(
        [], description="Optional BM25 queries to run over indexed outputs (matches returned inline)."
    )


class _FetchArgs(BaseModel):
    url: str = Field(..., description="URL to fetch.")
    source: str = Field("", description="Label for the indexed content (surfaces in ctx_search hits).")
    ttl: int = Field(_DEFAULT_TTL_S, description="Cache freshness window in seconds (default 86400 = 24h).")
    force: bool = Field(False, description="Skip the TTL cache and re-fetch.")


# -- tool builder ------------------------------------------------------------

def _timeout_envelope(tool: str, exc: ExecTimeout, cmd: str) -> str:
    """Render an ExecTimeout as a clean tool-result string the agent can act on.

    Without this, ExecTimeout walks out of the StructuredTool, through
    langgraph's ToolNode (whose default ``_handle_tool_errors`` re-raises
    anything that isn't a ``ToolInvocationError``), up to the model node —
    where ``retry_on_stream_stall`` matches the words "timed out" / "timeout"
    in the exception message and triggers 4 useless retries of the SAME tool
    call (each hitting the same 120s wall-clock budget) before surfacing the
    misleading "⚠️ model stream stalled" banner. The describe_image tool
    already does this envelope conversion at ``sandbox/tools/_media.py:230``;
    the ctx_* tools were missing it.

    The envelope tells the agent (a) it was a timeout, not a crash, (b) the
    budget that was hit, and (c) the workaround for long-running work — split
    into shorter steps and poll across calls instead of waiting in one exec.
    """
    preview = cmd if len(cmd) <= 140 else cmd[:137] + "..."
    return (
        f"[{tool}] timeout: {exc}.\n"
        f"Command preview: {preview}\n"
        f"The sandbox caps each exec call at a hard wall-clock budget. "
        f"For long-running work, write progress to a marker file and poll "
        f"across separate tool calls — do NOT wrap the wait in one exec."
    )

def build_exec_tools(store: EventStore, exec_client: object) -> list[StructuredTool]:
    """The 4 exec-dependent context tools, bound to ``store`` + ``exec_client``.

    Only built when ``build_context_tools`` receives a non-None exec_client —
    tests that pass exec_client=None get the 6 base tools only (no Docker
    dependency). The exec_client must have an ``exec(command: str) ->
    tuple[str, int]`` method (the ``DockerExecClient`` contract).
    """

    def _execute(language: str, code: str) -> str:
        try:
            cmd = _build_exec_command(language, code)
        except ValueError as e:
            return str(e)
        try:
            out, exit_code = exec_client.exec(cmd)
        except ExecTimeout as exc:
            return _timeout_envelope("ctx_execute", exc, cmd)
        if exit_code != 0:
            return f"[ctx_execute] exit {exit_code}\n{out}"
        return out

    def _execute_file(path: str, language: str, code: str) -> str:
        try:
            cmd = _build_file_command(path, language, code)
        except ValueError as e:
            return str(e)
        try:
            out, exit_code = exec_client.exec(cmd)
        except ExecTimeout as exc:
            return _timeout_envelope("ctx_execute_file", exc, cmd)
        if exit_code != 0:
            return f"[ctx_execute_file] exit {exit_code}\n{out}"
        return out

    def _batch_execute(commands: list[dict], queries: list[str] | None = None) -> str:
        qs = queries or []
        sections: list[str] = []
        for entry in commands:
            # LangChain validates against the pydantic schema, so entries arrive
            # as _BatchCommand instances in production; tests may pass raw dicts.
            if isinstance(entry, dict):
                command = entry.get("command", "")
                label = entry.get("label", "") or command[:60]
            else:
                command = getattr(entry, "command", "")
                label = getattr(entry, "label", "") or command[:60]
            if not command:
                continue
            # Per-command try/except: a timeout on one command is reported in
            # its section, and the rest of the batch continues. Aborting the
            # whole loop on a single timeout would lose partial progress and
            # hide which commands actually completed.
            try:
                out, exit_code = exec_client.exec(command)
            except ExecTimeout as exc:
                sections.append(f"- {label}: TIMEOUT — {exc}")
                continue
            stash = store.stash_blob(out, tool=f"ctx_batch:{label}")
            sections.append(
                f"- {label}: exit={exit_code}, {len(out)} chars, handle {stash.handle}"
            )
        store.flush()

        lines = [f"Executed {len(sections)} command(s). Outputs indexed:"]
        lines.extend(sections)

        if qs:
            lines.append(f"\nQuery results ({len(qs)} query/q):")
            for q in qs:
                hits = store.search_context(q, limit=5)
                if hits:
                    lines.append(f"Q: {q}")
                    for h in hits:
                        tag = f"[{h.kind}]"
                        handle = f" {h.handle}" if h.handle else ""
                        lines.append(f"  {tag}{handle}: {h.snippet}")
                else:
                    lines.append(f"Q: {q} — no matches yet")

        return "\n".join(lines)

    def _fetch_and_index(url: str, source: str = "", ttl: int = _DEFAULT_TTL_S, force: bool = False) -> str:
        if not url:
            return "no URL provided."
        cache_tag = f"ctx_fetch:{url}"
        # TTL cache: return the fresh blob's handle without re-fetching.
        if not force:
            cached = store.get_blob_by_tool(cache_tag, max_age_s=ttl if ttl > 0 else None)
            if cached:
                return (
                    f"cached (fresh < {ttl}s): {cached['chars']} chars from {url}. "
                    f"Full text via ctx_recall({cached['handle']!r}); "
                    f"search via ctx_search(<phrase>)."
                )
        # Fetch + convert HTML→text inside the container (one exec call).
        cmd = _build_fetch_command(url)
        try:
            out, exit_code = exec_client.exec(cmd)
        except ExecTimeout as exc:
            return _timeout_envelope("ctx_fetch_and_index", exc, cmd)
        if exit_code != 0:
            return f"[ctx_fetch_and_index] fetch failed (exit {exit_code}):\n{out[:300]}"
        if not out.strip():
            return f"[ctx_fetch_and_index] empty response from {url}."
        tag = f"ctx_fetch:{source or url}" if source else cache_tag
        stash = store.stash_blob(out, tool=tag)
        store.flush()
        preview = out[:200].replace("\n", " ")
        return (
            f"Fetched {len(out)} chars from {url}. Indexed under {stash.handle}. "
            f"Preview: {preview}... "
            f"Search via ctx_search(<phrase>); full text via ctx_recall({stash.handle!r})."
        )

    execute = StructuredTool.from_function(
        _execute,
        name="ctx_execute",
        description=(
            "Run code in the sandbox (NOT your working context). Languages: "
            + ", ".join(_supported_languages()) + ". "
            "Print output is captured and returned; the code itself never enters "
            "context. Use to compute, transform, or inspect without spending tokens."
        ),
        args_schema=_ExecuteArgs,
    )
    execute_file = StructuredTool.from_function(
        _execute_file,
        name="ctx_execute_file",
        description=(
            "Read a file into FILE_CONTENT inside the sandbox and run code over "
            "it. The file's bytes never enter your context — only stdout. Use to "
            "process large files (grep, parse, summarize) without loading them."
        ),
        args_schema=_ExecuteFileArgs,
    )
    batch_execute = StructuredTool.from_function(
        _batch_execute,
        name="ctx_batch_execute",
        description=(
            "Run multiple shell commands in ONE call. Every command's output is "
            "auto-indexed into the knowledge base. Optionally pass queries to "
            "surface matching sections inline (no follow-up ctx_search needed)."
        ),
        args_schema=_BatchArgs,
    )
    fetch_and_index = StructuredTool.from_function(
        _fetch_and_index,
        name="ctx_fetch_and_index",
        description=(
            "Fetch a URL, convert HTML to text, and index it into the persistent "
            "knowledge base. The raw page bytes never enter your context — only "
            "a short preview. 24h TTL cache by default; set force=true to re-fetch."
        ),
        args_schema=_FetchArgs,
    )
    return [execute, execute_file, batch_execute, fetch_and_index]
