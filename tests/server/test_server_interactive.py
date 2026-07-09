"""E2E: the server's interactive REST lane — SSE streaming + interrupt-correct
polled runs + resume.

Proves three things that were broken / missing in ``server.py``:

* **The #3 fix** — an ``interrupt()`` (e.g. ``ask_user``) was reported as
  ``status="success"`` with a stale tool-call message (``ainvoke`` RETURNS on
  interrupt, it does not hang; the old ``_final_answer`` read the trailing
  tool-call AIMessage). Now the polled endpoints return ``status="interrupted"``
  + the interrupt payload, and the client resumes via ``command={"resume": ...}``.
* **The HIGH-priority SSE gap** — ``/runs/stream`` + ``/threads/{id}/runs/stream``
  emit the langgraph stream (``messages`` token/tool-call chunks + ``updates`` +
  ``values``) as SSE.
* **Resume** — ``command={"resume": ...}`` drives ``ainvoke(Command(resume=...))``
  on the same thread and the run completes.

The server's HTTP/SSE/resume/interrupt machinery is exercised against a REAL
langgraph react graph whose tool body calls ``interrupt()`` — a stand-in for the
pux ``ask_user`` tool (no shipped org opts into ``ask_user``, so it can't drive
this test). Only the LLM is scripted; the graph, the shared sqlite checkpointer,
the SSE framing, and the resume path are all the real server code. The SSE bytes
are round-tripped through the ``langgraph_sdk``'s OWN ``SSEDecoder`` — if the wire
format drifts from what a ``RunClient`` consumes, parsing fails.

The harness opens the real thread store on a temp DB and patches
``_get_graph`` + ``discover_orgs`` — it does NOT run the full server lifespan
(which would also spin up MCP tool servers + the AG-UI mounts; out of scope for
the interactive-REST surface and environment-fragile in CI).
"""
from __future__ import annotations

import asyncio

import httpx
from httpx import ASGITransport
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.types import interrupt

from langgraph_sdk.sse import BytesLineDecoder, SSEDecoder  # the SDK's own parser

from pux_harness.run_events import EventBus
from pux_harness.server import app
from pux_harness.threads import open_thread_store

ORG = "general"  # accepted by the patched discover_orgs; the graph is swapped below.


# --- scripted models + stand-in graph ----------------------------------------

class _AskModel(BaseChatModel):
    """Calls the interrupting ``ask`` tool on a fresh turn; emits a final
    answer once the tool's ToolMessage is present (i.e. after resume). Stateless
    per call — behaviour keys off the message history, so graph caching is
    irrelevant and resume works across separate HTTP requests."""

    def _llm_type(self) -> str:
        return "ask-scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001,ANN002
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        answered = any(isinstance(m, ToolMessage) for m in messages)
        if answered:
            m = AIMessage(content="done after resume")
        else:
            m = AIMessage(
                content="",
                tool_calls=[{"name": "ask", "args": {"question": "continue?"},
                             "id": "c1", "type": "tool_call"}],
            )
        return ChatResult(generations=[ChatGeneration(message=m)])


class _PlainModel(BaseChatModel):
    """Always answers — the happy-path (no interrupt) model."""

    def _llm_type(self) -> str:
        return "plain-scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001,ANN002
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="plain final answer"))])


def _ask_tool_factory():
    @tool
    def ask(question: str) -> str:
        """Ask the human a question; pauses the graph until they reply."""
        return interrupt({"question": question})
    return ask


def _build_graph(saver, *, interrupt_graph: bool):
    model = _AskModel() if interrupt_graph else _PlainModel()
    tools = [_ask_tool_factory()] if interrupt_graph else []
    return create_agent(model, tools, checkpointer=saver)


def _install_fakes(monkeypatch, *, interrupt_graph: bool) -> None:
    """Replace ``server._get_graph`` with a real react graph bound to the app's
    shared sqlite checkpointer (so cross-request resume persists by thread_id),
    and narrow ``discover_orgs`` to the one org this suite drives. The graph is
    built lazily — ``app.state.saver`` only exists inside the test harness."""
    cache: dict[str, object] = {}

    def fake(org: str):  # noqa: ANN202
        if org not in cache:
            cache[org] = _build_graph(app.state.saver, interrupt_graph=interrupt_graph)
        return cache[org]

    monkeypatch.setattr("pux_harness.server._get_graph", fake)
    monkeypatch.setattr("pux_harness.server.discover_orgs", lambda: [ORG])


def _parse_sse(raw: bytes) -> list:
    """Parse raw SSE bytes with the langgraph_sdk's OWN decoder — the exact code
    path a ``RunClient`` uses. Returns the ``StreamPart``s (``.event``/``.data``)."""
    bld = BytesLineDecoder()
    sse = SSEDecoder()
    parts = []
    for line in bld.decode(raw):
        p = sse.decode(line)
        if p is not None:
            parts.append(p)
    for line in bld.flush():
        p = sse.decode(line)
        if p is not None:
            parts.append(p)
    return parts


async def _drive(db_path, fn):
    """Open the real thread store on a temp DB, wire the minimal app.state the
    endpoints need (NO MCP/AG-UI lifespan), and run ``fn(client)``."""
    async with open_thread_store(db_path=db_path) as store:
        app.state.saver = store.saver
        app.state.db = store.db
        app.state.store = store
        app.state.graphs = {}
        app.state.runs = {}
        app.state.run_meta = {}
        app.state.mcp = {}
        # The lifespan mounts an EventBus on app.state.events; _run_task publishes
        # run completions to it. Tests bypass the lifespan (no MCP/AG-UI), so
        # mirror it here (in-memory) or _run_task raises AttributeError.
        app.state.events = EventBus()
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await fn(client)


# --- tests --------------------------------------------------------------------

def test_polled_ephemeral_run_reports_interrupted_not_success(monkeypatch, tmp_path):
    """The #3 fix: an interrupting ephemeral run returns ``status="interrupted"``
    + the interrupt payload — never ``status="success"`` with a stale message."""
    _install_fakes(monkeypatch, interrupt_graph=True)

    async def body(c):
        return (await c.post("/runs/wait", json={"agent_id": ORG, "input": "go"})).json()

    out = asyncio.run(_drive(tmp_path / "t.sqlite", body))
    assert out["status"] == "interrupted", out
    assert out["interrupts"], f"interrupt payload missing: {out}"
    # The stale-success bug would have set status=success + a non-empty 'output'
    # scraped from the trailing tool-call AIMessage — guard against regression.
    assert out["output"] == ""


def test_resume_over_polled_thread_run_completes(monkeypatch, tmp_path):
    """Resume drives ``ainvoke(Command(resume=...))`` on the same thread: the
    interrupting run pauses, then a second run carrying ``command`` completes."""
    _install_fakes(monkeypatch, interrupt_graph=True)

    async def body(c):
        thread_id = (await c.post("/threads", json={"agent_id": ORG})).json()["thread_id"]
        r1 = await c.post(f"/threads/{thread_id}/runs", json={"input": "go"})
        wait1 = (await c.get(f"/runs/{r1.json()['run_id']}/wait")).json()
        r2 = await c.post(f"/threads/{thread_id}/runs",
                          json={"command": {"resume": "yes"}})
        wait2 = (await c.get(f"/runs/{r2.json()['run_id']}/wait")).json()
        return wait1, wait2

    wait1, wait2 = asyncio.run(_drive(tmp_path / "t.sqlite", body))
    assert wait1["status"] == "interrupted", wait1
    assert wait2["status"] == "success", wait2
    assert wait2["output"] == "done after resume", wait2


def test_stream_emits_sdk_parseable_events_with_interrupt(monkeypatch, tmp_path):
    """``/runs/stream`` emits SSE the SDK can parse: a leading ``metadata``
    event, then ``messages``/``updates``/``values`` — and the interrupt surfaces
    as a ``__interrupt__`` key in a ``values`` event."""
    _install_fakes(monkeypatch, interrupt_graph=True)

    async def body(c):
        raw = b""
        async with c.stream("POST", "/runs/stream",
                            json={"agent_id": ORG, "input": "go"}) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            async for chunk in resp.aiter_bytes():
                raw += chunk
        return _parse_sse(raw)

    parts = asyncio.run(_drive(tmp_path / "t.sqlite", body))
    assert parts, "no SSE frames parsed"
    assert parts[0].event == "metadata", parts[0]          # canonical stream opener
    assert "run_id" in parts[0].data, parts[0].data
    events = {p.event for p in parts}
    assert {"metadata", "values"} <= events, events
    assert any(p.event == "values" and "__interrupt__" in p.data for p in parts), \
        "interrupt did not surface in any values event"


def test_stream_resume_completes_with_final(monkeypatch, tmp_path):
    """Thread-scoped stream: the first stream interrupts, the second (with
    ``command``) completes — no ``__interrupt__`` in the resumed stream's values."""
    _install_fakes(monkeypatch, interrupt_graph=True)

    async def body(c):
        thread_id = (await c.post("/threads", json={"agent_id": ORG})).json()["thread_id"]

        async def stream(payload):
            raw = b""
            async with c.stream("POST", f"/threads/{thread_id}/runs/stream",
                                json=payload) as resp:
                assert resp.status_code == 200
                async for chunk in resp.aiter_bytes():
                    raw += chunk
            return _parse_sse(raw)

        first = await stream({"input": "go"})
        resumed = await stream({"command": {"resume": "yes"}})
        return first, resumed

    first, resumed = asyncio.run(_drive(tmp_path / "t.sqlite", body))
    assert any(p.event == "values" and "__interrupt__" in p.data for p in first)
    assert not any(p.event == "values" and "__interrupt__" in p.data for p in resumed), \
        "resumed stream still interrupted"
    last_values = [p for p in resumed if p.event == "values"][-1]
    contents = [m.get("content") for m in last_values.data.get("messages", [])
                if isinstance(m, dict)]
    assert "done after resume" in contents, contents


def test_stream_happy_path_emits_messages_without_interrupt(monkeypatch, tmp_path):
    """A non-interrupting run streams message chunks + a final values event with
    the answer, and never carries ``__interrupt__`` (the SSE happy path)."""
    _install_fakes(monkeypatch, interrupt_graph=False)

    async def body(c):
        raw = b""
        async with c.stream("POST", "/runs/stream",
                            json={"agent_id": ORG, "input": "hi"}) as resp:
            async for chunk in resp.aiter_bytes():
                raw += chunk
        return _parse_sse(raw)

    parts = asyncio.run(_drive(tmp_path / "t.sqlite", body))
    events = {p.event for p in parts}
    assert {"metadata", "values"} <= events, events
    assert not any(p.event == "values" and "__interrupt__" in p.data for p in parts)
    last_values = [p for p in parts if p.event == "values"][-1]
    contents = [m.get("content") for m in last_values.data.get("messages", [])
                if isinstance(m, dict)]
    assert "plain final answer" in contents, contents


def test_unknown_agent_404(monkeypatch, tmp_path):
    """Guard: the interrupt-aware code paths still reject an unknown agent_id
    before touching the graph (the agent_id check is unchanged)."""
    _install_fakes(monkeypatch, interrupt_graph=True)

    async def body(c):
        r = await c.post("/runs/wait", json={"agent_id": "nope", "input": "x"})
        return r.status_code

    assert asyncio.run(_drive(tmp_path / "t.sqlite", body)) == 404
