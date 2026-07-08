"""E2E: the langgraph-api ``assistants`` surface — driven through the
langgraph_sdk's OWN ``AssistantsClient`` (the exact path Studio's agent picker +
the SDK ``assistants.search/get/get_schemas`` hit).

Closes the LOW-priority "agent schemas" checklist item (the SDK has no
``/agents/{id}/schemas`` — the real surface is ``/assistants/{id}/schemas``).
An org maps 1:1 to an assistant (assistant_id == graph_id == org). Read-only:
search + get + get_schemas; the mutating paths are deliberately absent (orgs are
declarative on disk).

get_schemas is proven against a REAL compiled ``create_agent`` graph (not a
stub), so the returned state/input/output/config are genuine JSON Schemas — a
stub can't exercise ``model_json_schema``. The HTTP/SSE machinery is the real
server; only the LLM is scripted (the graph + its schemas are real).
"""

from __future__ import annotations

import asyncio

import pytest
import httpx
from httpx import ASGITransport
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph_sdk.client import LangGraphClient
from langgraph_sdk.errors import NotFoundError

from pux_harness.server import app
from pux_harness.threads import open_thread_store

ORG = "general"


class _PlainModel(BaseChatModel):
    """Always-answers scripted model — only the graph + its schemas matter here."""

    def _llm_type(self) -> str:
        return "plain-scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001,ANN002
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


def _install_real_graph(monkeypatch) -> None:
    """Point ``server._get_graph`` at a REAL compiled agent graph bound to the
    app's shared sqlite checkpointer, so ``get_schemas`` reads genuine JSON
    Schemas. Narrow ``discover_orgs`` to the one org this suite drives."""
    cache: dict[str, object] = {}

    def fake(org: str):  # noqa: ANN202
        if org not in cache:
            cache[org] = create_agent(_PlainModel(), [], checkpointer=app.state.saver)
        return cache[org]

    monkeypatch.setattr("pux_harness.server._get_graph", fake)
    monkeypatch.setattr("pux_harness.server.discover_orgs", lambda: [ORG])
    # orgs live in the PARENT repo; the slug list is read from disk by
    # ``org_agent_slugs`` (a separate resolution path from discover_orgs), so
    # patch it too to keep the submodule test filesystem-independent.
    monkeypatch.setattr("pux_harness.server.org_agent_slugs", lambda org: [ORG])


async def _drive(db_path, fn):
    async with open_thread_store(db_path=db_path) as store:
        app.state.saver = store.saver
        app.state.db = store.db
        app.state.store = store
        app.state.graphs = {}
        app.state.runs = {}
        app.state.run_meta = {}
        app.state.mcp = {}
        app.state.started_at = "2026-07-08T00:00:00+00:00"
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await fn(LangGraphClient(client))


# --- search / get ------------------------------------------------------------


def test_assistants_search_lists_org_as_assistant(monkeypatch, tmp_path):
    _install_real_graph(monkeypatch)

    async def body(c):
        return await c.assistants.search()

    found = asyncio.run(_drive(tmp_path / "t.sqlite", body))
    assert len(found) == 1, found
    a = found[0]
    assert a["assistant_id"] == ORG, a
    assert a["graph_id"] == ORG, a  # org == graph_id (one graph per org)
    assert a["name"] == ORG, a
    assert a["version"] == 1, a
    assert a["created_at"], a  # present + ISO-ish (Studio displays it)


def test_assistants_search_graph_id_filter(monkeypatch, tmp_path):
    _install_real_graph(monkeypatch)

    async def body(c):
        return await c.assistants.search(graph_id="nope")

    assert asyncio.run(_drive(tmp_path / "t.sqlite", body)) == []


def test_assistant_get_returns_descriptor(monkeypatch, tmp_path):
    _install_real_graph(monkeypatch)

    async def body(c):
        return await c.assistants.get(ORG)

    a = asyncio.run(_drive(tmp_path / "t.sqlite", body))
    assert a["assistant_id"] == ORG, a


def test_assistant_get_unknown_raises_not_found(monkeypatch, tmp_path):
    _install_real_graph(monkeypatch)

    async def body(c):
        await c.assistants.get("nope")

    with pytest.raises(NotFoundError):
        asyncio.run(_drive(tmp_path / "t.sqlite", body))


# --- get_schemas (the "agent schemas" item) -----------------------------------


def test_assistant_schemas_returns_json_schemas(monkeypatch, tmp_path):
    """get_schemas returns a JSON Schema per channel, derived LIVE from the real
    compiled graph: input/output/config are valid object schemas with properties
    (a create_agent graph's input carries the messages field)."""
    _install_real_graph(monkeypatch)

    async def body(c):
        return await c.assistants.get_schemas(ORG)

    schemas = asyncio.run(_drive(tmp_path / "t.sqlite", body))
    assert schemas["graph_id"] == ORG, schemas
    for key in ("state_schema", "input", "output", "config_schema", "context_schema"):
        assert key in schemas, f"missing {key}: {list(schemas)}"
    # output is a genuine JSON Schema from the REAL compiled graph (type=object +
    # properties); input uses $ref+$defs (also real). Proves we read a REAL graph,
    # not the {"type":"object"} placeholder a stub would yield.
    out = schemas["output"]
    assert out.get("type") == "object" and "properties" in out, out
    assert "$ref" in schemas["input"] or "properties" in schemas["input"], schemas["input"]
