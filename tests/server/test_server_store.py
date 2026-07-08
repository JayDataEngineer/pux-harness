"""E2E: the Store/Memory REST surface — driven through the langgraph_sdk's OWN
``StoreClient`` (the exact code path Studio + any langgraph-api client uses).

Closes the MEDIUM-priority Agent Protocol gap from the checklist:
``/store/items`` (PUT/GET/DELETE), ``/store/items/search``, ``/store/namespaces``.
The SDK's ``HttpClient`` raises on 4xx (``_araise_for_status_typed``) and parses
the JSON body (``_adecode_json``), so if the route shapes, query params,
payloads, or response envelopes drift from what the SDK sends/expects, these
calls raise. Round-tripping through the SDK is the contract proof — a hand-rolled
httpx assertion would only prove our own belief about the shape.

Also pins the SEAM: the store the REST surface writes IS the shared
``app.state.base_store`` that ``_get_graph`` hands to ``build_graph`` — so a
memory written over REST is visible to a graph's memory tools on the next run
(one backend across the seam, not a per-org private store).
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from httpx import ASGITransport
from langgraph.store.memory import InMemoryStore
from langgraph_sdk.client import LangGraphClient
from langgraph_sdk.errors import NotFoundError

from pux_harness.server import app


async def _drive(fn):
    """Point the SDK client at the ASGI app over an in-memory transport. The
    store is a fresh InMemoryStore on app.state — the same attribute ``_get_graph``
    reads, so the seam test is meaningful."""
    app.state.base_store = InMemoryStore()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as httpx_client:
        client = LangGraphClient(httpx_client)
        return await fn(client)


def test_put_then_get_roundtrips_via_sdk():
    """put_item then get_item returns the value + ISO timestamps — the canonical
    SDK usage example from StoreClient.put_item/get_item."""
    async def body(c):
        await c.store.put_item(["docs", "u1"], "k1", {"title": "Hi", "n": 1})
        item = await c.store.get_item(["docs", "u1"], "k1")
        return item

    item = asyncio.run(_drive(body))
    assert item["namespace"] == ["docs", "u1"], item
    assert item["key"] == "k1", item
    assert item["value"] == {"title": "Hi", "n": 1}, item
    assert item["created_at"], f"missing created_at: {item}"
    assert item["updated_at"], f"missing updated_at: {item}"


def test_get_missing_raises_not_found():
    """An absent item → 404 → the SDK raises NotFoundError (the contract a client
    catches)."""
    async def body(c):
        await c.store.get_item(["docs", "missing"], "nope")

    with pytest.raises(NotFoundError):
        asyncio.run(_drive(body))


def test_search_by_namespace_prefix():
    """search_items under a prefix returns every item in that subtree, across
    sibling leaf namespaces."""
    async def body(c):
        await c.store.put_item(["docs", "u1"], "a", {"v": 1})
        await c.store.put_item(["docs", "u2"], "b", {"v": 2})
        await c.store.put_item(["notes", "u1"], "c", {"v": 3})  # other root
        res = await c.store.search_items(["docs"])
        return res

    res = asyncio.run(_drive(body))
    keys = {i["key"] for i in res["items"]}
    assert keys == {"a", "b"}, res


def test_search_with_filter():
    """search_items filter narrows by a value field (InMemoryStore key/value
    equality match)."""
    async def body(c):
        await c.store.put_item(["docs"], "a", {"author": "John", "v": 1})
        await c.store.put_item(["docs"], "b", {"author": "Jane", "v": 2})
        res = await c.store.search_items(["docs"], filter={"author": "John"})
        return res

    res = asyncio.run(_drive(body))
    keys = {i["key"] for i in res["items"]}
    assert keys == {"a"}, res


def test_delete_then_get_missing():
    """delete_item removes the item; a subsequent get raises NotFound."""
    async def body(c):
        await c.store.put_item(["docs", "u1"], "k", {"v": 1})
        await c.store.delete_item(["docs", "u1"], "k")
        await c.store.get_item(["docs", "u1"], "k")

    with pytest.raises(NotFoundError):
        asyncio.run(_drive(body))


def test_list_namespaces_under_prefix():
    """list_namespaces returns the namespace paths under a prefix, respecting
    max_depth."""
    async def body(c):
        await c.store.put_item(["docs", "u1", "reports"], "a", {"v": 1})
        await c.store.put_item(["docs", "u2", "invoices"], "b", {"v": 2})
        await c.store.put_item(["notes"], "c", {"v": 3})  # other root
        return await c.store.list_namespaces(prefix=["docs"], max_depth=3)

    ns = asyncio.run(_drive(body))
    assert ["docs", "u1", "reports"] in ns, ns
    assert ["docs", "u2", "invoices"] in ns, ns
    assert all(n[0] == "docs" for n in ns), ns  # prefix respected


def test_put_rejects_dot_in_namespace_label():
    """The server enforces the no-dots-in-namespace rule (the SDK joins on '.'
    for GET) — 422, not a silent mangle. The SDK blocks this client-side, so
    drive the route with raw httpx to prove the server's own guard."""
    async def put_raw():
        app.state.base_store = InMemoryStore()
        async with httpx.AsyncClient(transport=ASGITransport(app=app),
                                     base_url="http://test") as raw:
            resp = await raw.put(
                "/store/items",
                json={"namespace": ["bad.label"], "key": "k", "value": {}},
            )
            return resp.status_code

    assert asyncio.run(put_raw()) == 422


def test_rest_store_is_the_graph_store_seam():
    """A memory written over REST lands on the SHARED ``app.state.base_store``
    — the same object ``_get_graph`` passes to ``build_graph``. So the graph's
    memory tools see REST-written memories (and vice versa): one backend."""
    async def body(c):
        await c.store.put_item(["mem", "u1"], "k", {"fact": "shared"})
        # Read it back via the SAME store the graph would use — NOT via REST.
        item = await app.state.base_store.aget(("mem", "u1"), "k")
        return item

    item = asyncio.run(_drive(body))
    assert item is not None, "REST put did not land on the shared graph store"
    assert item.value == {"fact": "shared"}, item.value
