"""Tool-server resolver — the ``github:`` release-bootstrap block (Phase A).

These are the pure-data (offline, no network) tests for the generic
``github:`` source descriptor on ``ToolServerSpec``: the schema parses, a
malformed block fails LOUD at parse time, the block survives
``_substitute_spec`` (the explicit-rebuild spec that ``mcp_client.py`` consumes),
and a malformed catalog entry surfaces through ``load_catalog`` — the seam that
feeds the offline contract (``validate_tool_servers`` → ``check_org``).

The download itself (``mcp_bootstrap.ensure_server``) + live handshake are
proven in ``test_mcp_bootstrap.py`` + the live Phase-E run; this file proves the
contract surface the catalog ships against.
"""
from __future__ import annotations

import pytest

from pux_harness.agent import tool_servers as TS
from pux_harness.agent.tool_servers import (
    ToolServerSpec,
    _parse_github_block,
    _parse_spec,
    _substitute_spec,
    load_catalog,
)


_GOOD_GITHUB = {
    "repo": "github/github-mcp-server",
    "asset": "github-mcp-server_*{os}*{arch}*.tar.gz",
    "binary": "github-mcp-server",
    "version": "latest",
}


def _stdio_entry(**over) -> dict:
    d = {
        "kind": "mcp",
        "transport": "stdio",
        "command": "github-mcp-server",
        "args": ["stdio"],
        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"},
        "github": dict(_GOOD_GITHUB),
    }
    d.update(over)
    return d


# --- parses cleanly ----------------------------------------------------------

def test_parse_spec_carries_github_block():
    spec = _parse_spec("github", _stdio_entry())
    assert spec.transport == "stdio"
    assert spec.github == _GOOD_GITHUB


def test_parse_spec_without_github_yields_none():
    spec = _parse_spec("web_research", {"kind": "mcp", "transport": "http",
                                        "url": "${URL}"})
    assert spec.github is None


def test_parse_github_block_none_when_absent():
    assert _parse_github_block("x", None, "stdio") is None


# --- malformed blocks fail LOUD at parse time (the contract) -----------------

def test_parse_spec_rejects_github_on_non_stdio_transport():
    # http transport with a github block — bootstrap is meaningless (no binary).
    d = _stdio_entry()
    d["transport"] = "http"
    d["url"] = "https://example.com"
    with pytest.raises(ValueError, match="only meaningful for stdio"):
        _parse_spec("github", d)


def test_parse_spec_rejects_github_missing_key():
    for key in ("repo", "asset", "binary", "version"):
        d = _stdio_entry()
        d["github"] = dict(_GOOD_GITHUB)
        del d["github"][key]
        with pytest.raises(ValueError, match=rf"github\.{key}.*required"):
            _parse_spec("github", d)


def test_parse_spec_rejects_github_bad_repo_shape():
    d = _stdio_entry()
    d["github"] = {**_GOOD_GITHUB, "repo": "not-a-slash-path"}
    with pytest.raises(ValueError, match="must be 'owner/name'"):
        _parse_spec("github", d)


def test_parse_spec_rejects_github_asset_without_platform_tokens():
    # An asset with {arch} but missing {os} → names {os}; vice-versa.
    d = _stdio_entry()
    d["github"] = {**_GOOD_GITHUB, "asset": "foo_{arch}.tar.gz"}
    with pytest.raises(ValueError, match="must contain \\{os\\}"):
        _parse_spec("github", d)
    d = _stdio_entry()
    d["github"] = {**_GOOD_GITHUB, "asset": "foo_{os}.tar.gz"}
    with pytest.raises(ValueError, match="must contain \\{arch\\}"):
        _parse_spec("github", d)
    # missing BOTH → the first check ({os}) fires.
    d = _stdio_entry()
    d["github"] = {**_GOOD_GITHUB, "asset": "no-tokens.tar.gz"}
    with pytest.raises(ValueError, match="must contain \\{os\\}"):
        _parse_spec("github", d)


def test_parse_spec_rejects_github_not_a_mapping():
    d = _stdio_entry()
    d["github"] = ["not", "a", "mapping"]
    with pytest.raises(ValueError, match="'github' must be a mapping"):
        _parse_spec("github", d)


# --- the block survives _substitute_spec (what mcp_client consumes) ----------

def test_substitute_spec_preserves_github_block():
    """``_substitute_spec`` rebuilds the spec with an explicit constructor —
    the ``github`` field MUST be propagated, else the resolved spec the session
    manager consumes loses the bootstrap source. (Env on OTHER fields is still
    resolved.)"""
    spec = _parse_spec("github", _stdio_entry())
    resolved = _substitute_spec(
        spec, env={"GITHUB_TOKEN": "tok"}, permissive=False,
    )
    assert resolved.github == _GOOD_GITHUB
    # and the env placeholder on another field resolved independently
    assert resolved.env["GITHUB_PERSONAL_ACCESS_TOKEN"] == "tok"


def test_substitute_spec_github_none_unchanged():
    spec = _parse_spec("web_research", {"kind": "mcp", "transport": "http",
                                        "url": "https://x"})
    assert _substitute_spec(spec, env={}).github is None


# --- command/args ${VAR} expansion (sandbox-browser MCP wiring) ---------------

def test_substitute_spec_expands_command_placeholder():
    """``command`` field expands ``${VAR}`` from env — same rule as ``url``.
    Required so a catalog entry can name a path resolved at runtime."""
    spec = _parse_spec("custom", {
        "kind": "mcp", "transport": "stdio",
        "command": "${HOME}/.local/bin/my-server",
    })
    resolved = _substitute_spec(spec, env={"HOME": "/root"})
    assert resolved.command == "/root/.local/bin/my-server"


def test_substitute_spec_expands_args_placeholders():
    """Every ``args`` entry is a separate expansion scope — each ``${VAR}`` is
    resolved against the env independently. This is what lets the
    sandbox-browser entry use::

        args: [exec, -i, "${PUX_SANDBOX_CONTAINER}", mc_browser.py]

    where ``PUX_SANDBOX_CONTAINER`` is the dynamic container name
    (``orchestrator-sandbox-p<hash>``)."""
    spec = _parse_spec("sandbox_browser", {
        "kind": "mcp", "transport": "stdio",
        "command": "docker",
        "args": ["exec", "-i", "${PUX_SANDBOX_CONTAINER}", "mc_browser.py"],
    })
    resolved = _substitute_spec(
        spec, env={"PUX_SANDBOX_CONTAINER": "orchestrator-sandbox-pabc123"},
    )
    assert resolved.command == "docker"
    assert resolved.args == [
        "exec", "-i", "orchestrator-sandbox-pabc123", "mc_browser.py",
    ]


def test_substitute_spec_permissive_leaves_args_placeholder_as_is():
    """In permissive (offline contract) mode, unresolved ``${VAR}`` in args is
    LEFT AS-IS — same behavior as url/headers/env. Lets the catalog ship a
    git-safe structural placeholder that fails loud at load time if the
    operator forgot to set the env var."""
    spec = _parse_spec("sandbox_browser", {
        "kind": "mcp", "transport": "stdio",
        "command": "docker",
        "args": ["exec", "-i", "${PUX_SANDBOX_CONTAINER}", "mc_browser.py"],
    })
    resolved = _substitute_spec(spec, env={}, permissive=True)
    assert resolved.args == [
        "exec", "-i", "${PUX_SANDBOX_CONTAINER}", "mc_browser.py",
    ]


def test_substitute_spec_raises_on_unresolved_args_placeholder():
    """Non-permissive mode raises on any unresolved ``${VAR}`` in args —
    fail-loud at load time rather than spawning ``docker exec -i`` with a
    literal ``${VAR}`` argument."""
    spec = _parse_spec("sandbox_browser", {
        "kind": "mcp", "transport": "stdio",
        "command": "docker",
        "args": ["exec", "-i", "${PUX_SANDBOX_CONTAINER}", "mc_browser.py"],
    })
    with pytest.raises(ValueError, match="PUX_SANDBOX_CONTAINER"):
        _substitute_spec(spec, env={})


def test_substitute_spec_args_with_no_placeholder_unchanged():
    """Args without placeholders pass through verbatim (no behavior change
    for existing catalog entries like the github-mcp-server one)."""
    spec = _parse_spec("github", _stdio_entry())
    resolved = _substitute_spec(spec, env={"GITHUB_TOKEN": "tok"})
    # The github entry uses args: ["stdio"] — no placeholder, unchanged
    assert resolved.args == ["stdio"]


def test_catalog_ref_copy_preserves_github_block():
    """The catalog-ref copy path (``ToolServerSpec(**{**ref.__dict__})``) must
    carry github forward — proven by constructing via the same spread the
    resolver uses."""
    ref = _parse_spec("github", _stdio_entry())
    copy = ToolServerSpec(**{**ref.__dict__})
    assert copy.github == _GOOD_GITHUB


# --- the contract seam: load_catalog raises on a malformed github block -------

def test_load_catalog_raises_on_malformed_github_entry(monkeypatch, tmp_path):
    """A malformed ``github:`` block in the shared catalog surfaces as a
    ValueError from ``load_catalog`` — the seam ``validate_tool_servers`` catches
    (via ``resolve_tool_servers``). The org need not reference the entry; the
    whole catalog is parsed for any org with a tool_servers list."""
    catalog_file = tmp_path / "tool_servers.yaml"
    catalog_file.write_text(
        "github:\n"
        "  kind: mcp\n"
        "  transport: stdio\n"
        "  command: github-mcp-server\n"
        "  github:\n"
        "    repo: github/github-mcp-server\n"
        "    asset: 'no-tokens.tar.gz'\n"   # missing {os}/{arch}
        "    binary: github-mcp-server\n"
        "    version: latest\n",
    )
    monkeypatch.setattr(TS, "_catalog_path", lambda: catalog_file)
    monkeypatch.setattr(TS, "_catalog_cache", None)
    with pytest.raises(ValueError, match="must contain"):
        load_catalog()


def test_load_catalog_parses_good_github_entry(monkeypatch, tmp_path):
    catalog_file = tmp_path / "tool_servers.yaml"
    catalog_file.write_text(
        "github:\n"
        "  kind: mcp\n"
        "  transport: stdio\n"
        "  command: github-mcp-server\n"
        "  args: [stdio]\n"
        "  github:\n"
        "    repo: github/github-mcp-server\n"
        "    asset: 'github-mcp-server_*{os}*{arch}*.tar.gz'\n"
        "    binary: github-mcp-server\n"
        "    version: latest\n",
    )
    monkeypatch.setattr(TS, "_catalog_path", lambda: catalog_file)
    monkeypatch.setattr(TS, "_catalog_cache", None)
    cat = load_catalog()
    assert "github" in cat
    assert cat["github"].github["binary"] == "github-mcp-server"
