"""Web search + fetch tools with swappable backends.

Two agent-callable tools:

    search  — query the web, get titles + URLs + snippets
    fetch   — read a web page, get cleaned text content

Backends chosen at build time:

    Search: DuckDuckGo (zero-config default) or SearXNG (self-hosted).
    Fetch:  httpx + trafilatura (lightweight), stdlib HTML-strip fallback.

Install the optional web deps::

    pip install 'deepagents-context[web]'
"""
from __future__ import annotations

import asyncio
import re

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# -- optional deps (lazy: module imports fine without them) ------------------

try:
    from duckduckgo_search import DDGS
    _HAS_DDG = True
except ImportError:
    _HAS_DDG = False

try:
    from trafilatura import extract as _trafilatura_extract
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

_NEED_WEB = (
    "Web tools require the [web] extra:\n"
    "    pip install 'deepagents-context[web]'"
)

_UA = "Mozilla/5.0 (compatible; deepagents-context/0.1)"


# -- arg schemas -------------------------------------------------------------

class _SearchArgs(BaseModel):
    query: str = Field(..., description="What to search for.")
    max_results: int = Field(5, description="Max results to return (default 5).")


class _FetchArgs(BaseModel):
    url: str = Field(..., description="The URL to read.")
    max_chars: int = Field(4000, description="Max characters to return (default 4000).")


# -- search backends ---------------------------------------------------------

def _ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
    if not _HAS_DDG:
        raise ImportError(_NEED_WEB)
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))
    return [
        {"title": h.get("title", ""), "url": h.get("href", ""),
         "snippet": h.get("body", "")}
        for h in hits
    ]


def _searxng_search(base_url: str, query: str, max_results: int) -> list[dict[str, str]]:
    resp = httpx.get(
        f"{base_url.rstrip('/')}/search",
        params={"q": query, "format": "json"},
        timeout=10.0,
        headers={"User-Agent": _UA},
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])[:max_results]
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""),
         "snippet": r.get("content", "")}
        for r in results
    ]


def _format_hits(hits: list[dict[str, str]]) -> str:
    if not hits:
        return "No results found."
    lines = [f"{len(hits)} result(s):"]
    for h in hits:
        lines.append(f"- {h['title']}\n  {h['url']}\n  {h['snippet']}")
    return "\n".join(lines)


# -- fetch backend -----------------------------------------------------------

def _strip_html(html: str) -> str:
    """Minimal HTML-to-text fallback when trafilatura is absent."""
    html = re.sub(
        r"<(script|style|nav|footer|header)[^>]*>.*?</\1>",
        "", html, flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _extract(html: str, max_chars: int) -> str:
    if _HAS_TRAFILATURA:
        text = _trafilatura_extract(html)
        if text:
            return text[:max_chars]
    return _strip_html(html)[:max_chars]


def _fetch_sync(url: str, max_chars: int) -> str:
    with httpx.Client(
        timeout=15.0, follow_redirects=True, headers={"User-Agent": _UA},
    ) as client:
        resp = client.get(url)
    ctype = resp.headers.get("content-type", "")
    if "text/" not in ctype and "html" not in ctype:
        return f"Unsupported content type: {ctype}"
    return _extract(resp.text, max_chars)


async def _fetch_async(url: str, max_chars: int) -> str:
    async with httpx.AsyncClient(
        timeout=15.0, follow_redirects=True, headers={"User-Agent": _UA},
    ) as client:
        resp = await client.get(url)
    ctype = resp.headers.get("content-type", "")
    if "text/" not in ctype and "html" not in ctype:
        return f"Unsupported content type: {ctype}"
    return _extract(resp.text, max_chars)


# -- builder -----------------------------------------------------------------

def build_web_tools(
    *,
    searxng_url: str | None = None,
) -> list[StructuredTool]:
    """Return ``[search, fetch]`` — two agent-callable web tools.

    Args:
        searxng_url: If set, search uses SearXNG (self-hosted, VPN-proof,
            best quality via engine aggregation). If ``None``, defaults to
            DuckDuckGo (zero-config, no API key, no infrastructure).

    Requires the ``[web]`` extra::

        pip install 'deepagents-context[web]'
    """
    use_searxng = searxng_url is not None

    # -- search --
    def _search(query: str, max_results: int = 5) -> str:
        if use_searxng:
            hits = _searxng_search(searxng_url, query, max_results)
        else:
            hits = _ddg_search(query, max_results)
        return _format_hits(hits)

    async def _asearch(query: str, max_results: int = 5) -> str:
        if use_searxng:
            hits = await asyncio.to_thread(
                _searxng_search, searxng_url, query, max_results,
            )
        else:
            hits = await asyncio.to_thread(_ddg_search, query, max_results)
        return _format_hits(hits)

    # -- fetch --
    def _fetch(url: str, max_chars: int = 4000) -> str:
        try:
            return _fetch_sync(url, max_chars)
        except httpx.HTTPError as exc:
            return f"Fetch failed: {exc}"

    async def _afetch(url: str, max_chars: int = 4000) -> str:
        try:
            return await _fetch_async(url, max_chars)
        except httpx.HTTPError as exc:
            return f"Fetch failed: {exc}"

    search = StructuredTool.from_function(
        _search,
        coroutine=_asearch,
        name="search",
        description="Search the web. Returns titles, URLs, and snippets.",
        args_schema=_SearchArgs,
    )
    fetch = StructuredTool.from_function(
        _fetch,
        coroutine=_afetch,
        name="fetch",
        description="Fetch and read a web page. Returns the content as text.",
        args_schema=_FetchArgs,
    )
    return [search, fetch]
