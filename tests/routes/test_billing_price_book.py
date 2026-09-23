"""Price-book CRUD end to end (WP2).

The service-level invariants have their own tests; these drive the HTTP surface
against a real database and a real ASGI app, which is the only way to catch what
sits between them: the request schemas rejecting a malformed price with a 422,
the route calling the overlap check *before* the write, the version bump landing
in the same update as the price, and the cache being invalidated so the next
read sees what the previous write did.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import register_handlers
from gpustack.routes import billing as billing_routes
from gpustack.schemas.billing import (
    SKU_GPU_HOUR_PREFIX,
    SKU_TOKEN_PROMPT,
    PriceBookEntry,
    UNIT_GPU_HOURS,
    UNIT_TOKENS,
)
from gpustack.server.billing_pricing import invalidate_price_cache
from gpustack.server.deps import get_session

T0 = datetime(2026, 9, 1, 0, 0, 0)
T1 = datetime(2026, 10, 1, 0, 0, 0)
PREFIX = "/billing/price-book"


def _payload(**overrides):
    body = {
        "sku": SKU_TOKEN_PROMPT,
        "model_name": "Qwen3-8B",
        "unit": UNIT_TOKENS,
        "price": "0.002",
        "per_quantity": "1000",
        "effective_from": T0.isoformat(),
    }
    body.update(overrides)
    return body


@pytest_asyncio.fixture
async def app_and_engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(PriceBookEntry.__table__.create)

    app = FastAPI()
    register_handlers(app)
    app.include_router(billing_routes.router, prefix=PREFIX)

    async def _session_dep():
        async with AsyncSession(engine) as session:
            yield session

    app.dependency_overrides[get_session] = _session_dep

    @asynccontextmanager
    async def _list_session():
        # Stands in for the app-wide session factory the list endpoint uses,
        # which is only wired up once the server has initialized the database.
        async with AsyncSession(engine) as session:
            yield session

    invalidate_price_cache()
    with patch.object(billing_routes, "async_session", _list_session):
        yield app, engine
    invalidate_price_cache()
    await engine.dispose()


@pytest_asyncio.fixture
async def client(app_and_engine):
    app, _ = app_and_engine
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_price(client):
    r = await client.post(PREFIX, json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] is not None
    assert body["version"] == 1
    assert body["sku"] == SKU_TOKEN_PROMPT
    assert float(body["price"]) == 0.002


@pytest.mark.asyncio
async def test_create_rejects_overlapping_window(client):
    assert (await client.post(PREFIX, json=_payload())).status_code == 200

    # Same timeline, window inside the open-ended one just created.
    r = await client.post(PREFIX, json=_payload(effective_from=T1.isoformat()))
    assert r.status_code == 422
    assert "overlaps existing price" in r.json()["message"]


@pytest.mark.asyncio
async def test_create_allows_window_after_closing_the_previous(client):
    first = await client.post(PREFIX, json=_payload(effective_to=T1.isoformat()))
    assert first.status_code == 200

    second = await client.post(PREFIX, json=_payload(effective_from=T1.isoformat()))
    assert second.status_code == 200, second.text


@pytest.mark.asyncio
async def test_create_rejects_unit_sku_mismatch(client):
    """A per-token price on a gpu-hour SKU would bill 3600x off."""
    r = await client.post(
        PREFIX, json=_payload(sku=SKU_GPU_HOUR_PREFIX + "910b", unit=UNIT_TOKENS)
    )
    assert r.status_code == 422
    assert "must be 'gpu_hours'" in r.text


@pytest.mark.asyncio
async def test_create_accepts_gpu_hour_sku(client):
    """The resource side of the price book — NPU card-hours, no model scope.

    Resource SKUs carry the accelerator in the sku itself and price per one
    unit, so a model_name would be wrong here and per_quantity stays 1.
    """
    r = await client.post(
        PREFIX,
        json=_payload(
            sku=SKU_GPU_HOUR_PREFIX + "910b",
            model_name=None,
            unit=UNIT_GPU_HOURS,
            price="12.5",
            per_quantity="1",
        ),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sku"] == SKU_GPU_HOUR_PREFIX + "910b"
    assert body["model_name"] is None

    # And it resolves for a resource rating, which passes no model.
    resolved = await client.post(
        f"{PREFIX}/resolve",
        json={"sku": SKU_GPU_HOUR_PREFIX + "910b", "quantity": 1.5},
    )
    assert resolved.json()["priced"] is True
    assert resolved.json()["amount"] == pytest.approx(18.75)


@pytest.mark.asyncio
async def test_create_rejects_unknown_sku(client):
    r = await client.post(PREFIX, json=_payload(sku="model.token.mystery"))
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_rejects_inverted_window(client):
    r = await client.post(PREFIX, json=_payload(effective_to=T0.isoformat()))
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_by_id_and_404(client):
    created = (await client.post(PREFIX, json=_payload())).json()

    r = await client.get(f"{PREFIX}/{created['id']}")
    assert r.status_code == 200
    assert r.json()["model_name"] == "Qwen3-8B"

    assert (await client.get(f"{PREFIX}/999999")).status_code == 404


@pytest.mark.asyncio
async def test_list_is_paginated(client):
    for i in range(3):
        await client.post(
            PREFIX,
            json=_payload(
                model_name=f"model-{i}",
                effective_from=(T0 + timedelta(days=i)).isoformat(),
            ),
        )

    r = await client.get(PREFIX)
    assert r.status_code == 200
    body = r.json()
    assert body["pagination"]["total"] == 3
    assert len(body["items"]) == 3


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_price_bumps_version(client):
    """Also the regression test for the naive/aware identity comparison.

    The stored ``effective_from`` comes back from the database aware while the
    same instant in this request body is naive; before normalization the
    identity guard read that as a change and rejected every update with 422.
    """
    created = (await client.post(PREFIX, json=_payload())).json()

    r = await client.put(f"{PREFIX}/{created['id']}", json=_payload(price="0.0015"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert float(body["price"]) == 0.0015
    assert body["version"] == 2, "a price change must produce a new version"


@pytest.mark.asyncio
async def test_update_without_price_change_keeps_version(client):
    created = (await client.post(PREFIX, json=_payload())).json()

    r = await client.put(f"{PREFIX}/{created['id']}", json=_payload())
    assert r.status_code == 200
    assert r.json()["version"] == 1, "a no-op PUT must not orphan ledger snapshots"


@pytest.mark.asyncio
async def test_update_rejects_identity_change(client):
    created = (await client.post(PREFIX, json=_payload())).json()

    r = await client.put(
        f"{PREFIX}/{created['id']}", json=_payload(model_name="Qwen3-30B")
    )
    assert r.status_code == 422
    assert "Cannot change model_name" in r.json()["message"]


@pytest.mark.asyncio
async def test_update_can_close_a_window(client):
    created = (await client.post(PREFIX, json=_payload())).json()

    r = await client.put(
        f"{PREFIX}/{created['id']}", json=_payload(effective_to=T1.isoformat())
    )
    assert r.status_code == 200, r.text

    # The now-closed timeline leaves room for a successor.
    successor = await client.post(PREFIX, json=_payload(effective_from=T1.isoformat()))
    assert successor.status_code == 200, successor.text


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_then_gone(client):
    created = (await client.post(PREFIX, json=_payload())).json()

    assert (await client.delete(f"{PREFIX}/{created['id']}")).status_code == 200
    assert (await client.get(f"{PREFIX}/{created['id']}")).status_code == 404


@pytest.mark.asyncio
async def test_delete_frees_the_window(client):
    created = (await client.post(PREFIX, json=_payload())).json()
    await client.delete(f"{PREFIX}/{created['id']}")

    again = await client.post(PREFIX, json=_payload())
    assert again.status_code == 200, again.text


# ---------------------------------------------------------------------------
# Resolve preview
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_returns_rate_and_amount(client):
    await client.post(PREFIX, json=_payload())

    r = await client.post(
        f"{PREFIX}/resolve",
        json={"sku": SKU_TOKEN_PROMPT, "model_name": "Qwen3-8B", "quantity": 1234},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["priced"] is True
    # 1234 tokens at 0.002 per 1000.
    assert body["amount"] == pytest.approx(0.002468)


@pytest.mark.asyncio
async def test_resolve_reports_unpriced_rather_than_zero(client):
    r = await client.post(
        f"{PREFIX}/resolve",
        json={"sku": SKU_TOKEN_PROMPT, "model_name": "Never-Priced"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["priced"] is False
    assert body["amount"] is None
    assert "no active price covers" in body["reason"]


@pytest.mark.asyncio
async def test_resolve_sees_a_price_created_moments_ago(client):
    """The write path must invalidate the resolver's cache in-process."""
    first = await client.post(
        f"{PREFIX}/resolve", json={"sku": SKU_TOKEN_PROMPT, "model_name": "Qwen3-8B"}
    )
    assert first.json()["priced"] is False

    await client.post(PREFIX, json=_payload())

    second = await client.post(
        f"{PREFIX}/resolve", json={"sku": SKU_TOKEN_PROMPT, "model_name": "Qwen3-8B"}
    )
    assert second.json()["priced"] is True
