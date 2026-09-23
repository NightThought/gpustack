"""Quota CRUD and the /check preview, over HTTP (WP5.4).

The gate's arithmetic has its own tests; these drive the admin surface against a
real ASGI app and a real database, which is where the things that only exist
between the two show up: the request schema rejecting a row whose subject
columns disagree with its scope, the duplicate guard running *before* the write
(the constraint cannot catch an all-models ceiling, because NULLs are distinct),
identity being immutable on update so a counter never ends up describing a
different subject, and ``used`` being unreachable from the API — a tenant that
could zero its own counter has no counter.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import register_handlers
from gpustack.routes import billing as billing_routes
from gpustack.schemas.billing import (
    SKU_TOKEN_PROMPT,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    Quota,
    QuotaLimitType,
    QuotaScope,
    SettleMode,
    UNIT_TOKENS,
)
from gpustack.server.billing_quota import invalidate_quota_cache
from gpustack.server.deps import get_session

NOW = datetime(2026, 9, 23, 14, 30, 0, tzinfo=timezone.utc)
DAY_START = datetime(2026, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
PREFIX = "/billing/quotas"

ORG = 990101
USER = 990201
KEY = 58
MODEL = "Qwen3-8B"


def _payload(**overrides):
    body = {
        "scope": QuotaScope.API_KEY.value,
        "api_key_id": KEY,
        "model_name": None,
        "limit_type": QuotaLimitType.DAILY_TOKENS.value,
        "limit_value": "1000",
        "enabled": True,
    }
    body.update(overrides)
    return body


@pytest_asyncio.fixture
async def app_and_engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        for model in (Quota, LedgerEntry):
            await conn.run_sync(model.__table__.create)

    app = FastAPI()
    register_handlers(app)
    app.include_router(billing_routes.quota_router, prefix=PREFIX)

    async def _session_dep():
        async with AsyncSession(engine) as session:
            yield session

    app.dependency_overrides[get_session] = _session_dep

    @asynccontextmanager
    async def _list_session():
        # Stands in for the app-wide factory the list endpoint and the gate's
        # rollover write use, which is only wired up once the server has
        # initialized the database.
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session

    invalidate_quota_cache()
    with patch.object(billing_routes, "async_session", _list_session):
        with patch("gpustack.server.db.async_session", _list_session):
            yield app, engine
    invalidate_quota_cache()
    await engine.dispose()


@pytest_asyncio.fixture
async def client(app_and_engine):
    app, _ = app_and_engine
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


async def _rows(engine):
    async with AsyncSession(engine) as s:
        return list((await s.exec(select(Quota))).all())


async def _seed_usage(engine, *, quantity="400", amount="0.80", occurred_at=NOW):
    async with AsyncSession(engine) as s:
        s.add(
            LedgerEntry(
                source_table="model_usage_details",
                source_id=1,
                api_key_id=KEY,
                user_id=USER,
                principal_id=ORG,
                model_name=MODEL,
                sku=SKU_TOKEN_PROMPT,
                quantity=Decimal(quantity),
                unit=UNIT_TOKENS,
                unit_price=Decimal("0.002"),
                amount=Decimal(amount),
                currency="CNY",
                direction=LedgerDirection.DEBIT,
                settle_mode=SettleMode.REALTIME,
                status=LedgerStatus.PENDING,
                occurred_at=occurred_at,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        await s.commit()


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_a_key_scoped_daily_token_ceiling(client, app_and_engine):
    _, engine = app_and_engine
    response = await client.post(PREFIX, json=_payload())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scope"] == QuotaScope.API_KEY.value
    assert body["api_key_id"] == KEY
    assert body["limit_type"] == QuotaLimitType.DAILY_TOKENS.value
    assert Decimal(body["limit_value"]) == Decimal("1000")
    # A fresh counter: nothing used, no window established yet.
    assert Decimal(body["used"]) == Decimal("0")
    assert body["window_start"] is None
    assert len(await _rows(engine)) == 1


@pytest.mark.asyncio
async def test_create_rejects_a_subject_that_disagrees_with_its_scope(client):
    """An api_key-scoped row carrying only a user_id would match nobody."""
    response = await client.post(
        PREFIX,
        json=_payload(scope=QuotaScope.API_KEY.value, api_key_id=None, user_id=USER),
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_create_rejects_extra_subject_columns(client):
    response = await client.post(
        PREFIX, json=_payload(scope=QuotaScope.USER.value, user_id=USER, api_key_id=KEY)
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_create_rejects_a_non_positive_ceiling(client):
    response = await client.post(PREFIX, json=_payload(limit_value="0"))
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_create_rejects_a_duplicate_all_models_ceiling(client, app_and_engine):
    """The unique constraint cannot catch this one, so the route must."""
    _, engine = app_and_engine
    first = await client.post(PREFIX, json=_payload())
    assert first.status_code == 200, first.text

    second = await client.post(PREFIX, json=_payload(limit_value="5000"))

    assert second.status_code == 422, second.text
    assert "already exists" in second.json()["message"]
    assert len(await _rows(engine)) == 1


@pytest.mark.asyncio
async def test_create_allows_the_same_subject_under_another_limit_type(client):
    assert (await client.post(PREFIX, json=_payload())).status_code == 200
    other = await client.post(
        PREFIX,
        json=_payload(limit_type=QuotaLimitType.DAILY_AMOUNT.value, limit_value="50"),
    )
    assert other.status_code == 200, other.text


@pytest.mark.asyncio
async def test_create_allows_the_same_limit_type_on_another_model(client):
    assert (await client.post(PREFIX, json=_payload())).status_code == 200
    other = await client.post(PREFIX, json=_payload(model_name="Qwen3-30B-A3B"))
    assert other.status_code == 200, other.text


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_is_paginated_and_filterable(client):
    await client.post(PREFIX, json=_payload())
    await client.post(
        PREFIX,
        json=_payload(
            scope=QuotaScope.ORGANIZATION.value,
            api_key_id=None,
            principal_id=ORG,
            limit_type=QuotaLimitType.MONTHLY_AMOUNT.value,
            limit_value="500",
            enabled=False,
        ),
    )

    everything = await client.get(PREFIX)
    assert everything.status_code == 200, everything.text
    body = everything.json()
    assert body["pagination"]["total"] == 2
    assert len(body["items"]) == 2

    only_keys = await client.get(PREFIX, params={"scope": QuotaScope.API_KEY.value})
    assert only_keys.json()["pagination"]["total"] == 1

    only_enabled = await client.get(PREFIX, params={"enabled": True})
    assert only_enabled.json()["pagination"]["total"] == 1

    by_org = await client.get(PREFIX, params={"principal_id": ORG})
    assert by_org.json()["pagination"]["total"] == 1


@pytest.mark.asyncio
async def test_get_one_and_404(client):
    created = (await client.post(PREFIX, json=_payload())).json()

    found = await client.get(f"{PREFIX}/{created['id']}")
    assert found.status_code == 200
    assert found.json()["id"] == created["id"]

    missing = await client.get(f"{PREFIX}/{created['id'] + 999}")
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_changes_the_ceiling_and_can_disable_it(client, app_and_engine):
    _, engine = app_and_engine
    created = (await client.post(PREFIX, json=_payload())).json()

    raised = await client.put(
        f"{PREFIX}/{created['id']}", json=_payload(limit_value="5000", enabled=False)
    )

    assert raised.status_code == 200, raised.text
    body = raised.json()
    assert Decimal(body["limit_value"]) == Decimal("5000")
    assert body["enabled"] is False
    assert Decimal((await _rows(engine))[0].limit_value) == Decimal("5000")


@pytest.mark.asyncio
async def test_update_rejects_moving_a_ceiling_to_another_subject(client):
    """The counter would keep a total that describes somebody else."""
    created = (await client.post(PREFIX, json=_payload())).json()

    moved = await client.put(
        f"{PREFIX}/{created['id']}", json=_payload(api_key_id=KEY + 1)
    )

    assert moved.status_code == 422, moved.text
    assert "cannot change" in moved.json()["message"]


@pytest.mark.asyncio
async def test_update_rejects_repointing_a_ceiling_at_another_model(client):
    created = (await client.post(PREFIX, json=_payload(model_name=MODEL))).json()

    moved = await client.put(
        f"{PREFIX}/{created['id']}", json=_payload(model_name="Qwen3-30B-A3B")
    )

    assert moved.status_code == 422, moved.text


@pytest.mark.asyncio
async def test_update_cannot_reset_the_counter(client, app_and_engine):
    """``used`` is absent from the editable surface, so sending it does nothing."""
    _, engine = app_and_engine
    created = (await client.post(PREFIX, json=_payload())).json()
    await _seed_usage(engine, quantity="400")
    # Let the gate establish and fill the window's counter.
    await client.post(PREFIX + "/check", json={"api_key_id": KEY, "model_name": MODEL})
    before = Decimal((await _rows(engine))[0].used)
    assert before == Decimal("400")

    payload = _payload(limit_value="2000")
    payload["used"] = "0"
    payload["window_start"] = None
    response = await client.put(f"{PREFIX}/{created['id']}", json=payload)

    assert response.status_code == 200, response.text
    row = (await _rows(engine))[0]
    assert Decimal(row.used) == Decimal("400")
    assert row.window_start is not None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_removes_it_from_the_admin_surface(client, app_and_engine):
    _, engine = app_and_engine
    created = (await client.post(PREFIX, json=_payload())).json()

    deleted = await client.delete(f"{PREFIX}/{created['id']}")
    assert deleted.status_code == 200, deleted.text

    assert (await client.get(f"{PREFIX}/{created['id']}")).status_code == 404
    listed = await client.get(PREFIX)
    assert listed.json()["pagination"]["total"] == 0


# ---------------------------------------------------------------------------
# The /check preview
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_reports_every_ceiling_and_admits_under_them(client):
    await client.post(PREFIX, json=_payload(limit_value="1000"))
    await client.post(
        PREFIX,
        json=_payload(limit_type=QuotaLimitType.DAILY_AMOUNT.value, limit_value="50"),
    )

    response = await client.post(
        PREFIX + "/check", json={"api_key_id": KEY, "model_name": MODEL}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["admitted"] is True
    assert body["refused_by"] is None
    assert len(body["evaluated"]) == 2
    by_type = {v["limit_type"]: v for v in body["evaluated"]}
    assert Decimal(by_type["daily_tokens"]["remaining"]) == Decimal("1000")
    assert by_type["daily_tokens"]["window_start"] is not None


@pytest.mark.asyncio
async def test_check_names_the_ceiling_that_refuses(client, app_and_engine):
    _, engine = app_and_engine
    await client.post(PREFIX, json=_payload(limit_value="100"))
    await _seed_usage(engine, quantity="400")

    response = await client.post(
        PREFIX + "/check", json={"api_key_id": KEY, "model_name": MODEL}
    )

    body = response.json()
    assert body["admitted"] is False
    assert body["refused_by"]["limit_type"] == QuotaLimitType.DAILY_TOKENS.value
    assert Decimal(body["refused_by"]["used"]) == Decimal("400")
    assert Decimal(body["refused_by"]["remaining"]) == Decimal("0")
    assert body["refused_by"]["exhausted"] is True


@pytest.mark.asyncio
async def test_check_sees_a_ceiling_created_moments_earlier(client):
    """Writes invalidate the cache; a stale one would hide the new limit."""
    assert (await client.post(PREFIX + "/check", json={"api_key_id": KEY})).json()[
        "evaluated"
    ] == []

    await client.post(PREFIX, json=_payload(limit_value="1000"))

    body = (await client.post(PREFIX + "/check", json={"api_key_id": KEY})).json()
    assert len(body["evaluated"]) == 1


@pytest.mark.asyncio
async def test_check_for_an_unlimited_caller_evaluates_nothing(client):
    await client.post(PREFIX, json=_payload(limit_value="1000"))

    response = await client.post(
        PREFIX + "/check", json={"api_key_id": KEY + 1, "model_name": MODEL}
    )

    body = response.json()
    assert body["admitted"] is True
    assert body["evaluated"] == []


@pytest.mark.asyncio
async def test_check_counts_only_the_model_a_ceiling_names(client, app_and_engine):
    _, engine = app_and_engine
    await client.post(PREFIX, json=_payload(model_name=MODEL, limit_value="100"))
    await _seed_usage(engine, quantity="400")

    other_model = await client.post(
        PREFIX + "/check", json={"api_key_id": KEY, "model_name": "Qwen3-30B-A3B"}
    )

    assert other_model.json()["admitted"] is True
    assert other_model.json()["evaluated"] == []


@pytest.mark.asyncio
async def test_quota_cache_is_dropped_after_a_delete(client, app_and_engine):
    _, engine = app_and_engine
    created = (await client.post(PREFIX, json=_payload(limit_value="1"))).json()
    await _seed_usage(engine, quantity="400")
    assert (await client.post(PREFIX + "/check", json={"api_key_id": KEY})).json()[
        "admitted"
    ] is False

    await client.delete(f"{PREFIX}/{created['id']}")

    assert (await client.post(PREFIX + "/check", json={"api_key_id": KEY})).json()[
        "admitted"
    ] is True
