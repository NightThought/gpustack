"""Wallet, ledger, redemption and adjustment endpoints (WP7.1 / WP4.5).

The money-moving surface, so these tests are about who is allowed to say what:

* a tenant reads *its own* wallet, resolved from the tenant context — never from
  a parameter it supplies, because an endpoint that takes a principal id from the
  caller is an endpoint that serves every wallet;
* an adjustment's author is the authenticated caller, not a field in the body;
* one idempotency key moves money once, and reusing a key for different
  parameters is refused rather than silently ignored;
* the ledger can be asked the questions a reconciliation asks — including an
  exact ``request_id``, which is the handle a dispute arrives with.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import get_tenant_context
from gpustack.routes import billing as billing_routes
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.schemas.billing import (
    SKU_GPU_HOUR_PREFIX,
    SKU_TOKEN_PROMPT,
    SKU_WALLET_ADJUSTMENT,
    SKU_WALLET_TOPUP,
    UNIT_GPU_HOURS,
    UNIT_TOKENS,
    Adjustment,
    Invoice,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    Redemption,
    RedemptionStatus,
    SettleMode,
    Wallet,
)
from gpustack.server import billing_settlement
from gpustack.server.billing_alerts import billing_alerts
from gpustack.server.deps import get_session

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
ORG = 990101
OTHER_ORG = 990102
OPERATOR = 7
SKU_910B = SKU_GPU_HOUR_PREFIX + "910b"


@pytest_asyncio.fixture
async def app_and_engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        for model in (
            Wallet,
            LedgerEntry,
            Adjustment,
            Invoice,
            Redemption,
            ApiKey,
            # An adjustment snapshots the subject's name, so the table has to
            # exist for the lookup to run.
            Principal,
        ):
            await conn.run_sync(model.__table__.create)

    async with AsyncSession(engine) as s:
        for principal_id, name in ((ORG, "acme"), (OTHER_ORG, "globex")):
            s.add(
                Principal(
                    id=principal_id,
                    kind=PrincipalType.ORG,
                    name=name,
                    display_name=name,
                    source="local",
                    is_admin=False,
                    is_active=True,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        await s.commit()

    app = FastAPI()
    register_handlers(app)
    app.include_router(billing_routes.wallet_router, prefix="/billing/wallets")
    app.include_router(billing_routes.ledger_router, prefix="/billing/ledger")
    app.include_router(billing_routes.adjustment_router, prefix="/billing/adjustments")
    app.include_router(billing_routes.tenant_router, prefix="/billing")

    async def _session_dep():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session

    app.dependency_overrides[get_session] = _session_dep
    # The tenant endpoints take the caller from here, not from the request body.
    app.dependency_overrides[get_tenant_context] = lambda: SimpleNamespace(
        user=SimpleNamespace(id=OPERATOR, name="admin"),
        is_platform_admin=True,
        current_principal_id=ORG,
        org_role=None,
    )

    @asynccontextmanager
    async def _list_session():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session

    billing_alerts.clear()
    with patch.object(billing_routes, "async_session", _list_session):
        with patch.object(billing_settlement, "async_session", _list_session):
            yield app, engine
    billing_alerts.clear()
    await engine.dispose()


@pytest_asyncio.fixture
async def client(app_and_engine):
    app, _ = app_and_engine
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


def _wallet(principal_id=ORG, balance="100.00", frozen="0", suspended=False,
            name="acme"):
    return Wallet(
        id=principal_id - 990000,
        principal_id=principal_id,
        principal_name=name,
        balance=Decimal(balance),
        frozen=Decimal(frozen),
        currency="CNY",
        suspended=suspended,
        suspended_at=NOW if suspended else None,
        created_at=NOW,
        updated_at=NOW,
    )


def _entry(id_, *, sku=SKU_TOKEN_PROMPT, unit=UNIT_TOKENS, amount="1.00",
           principal_id=ORG, settle_mode=SettleMode.REALTIME,
           status=LedgerStatus.PENDING, request_id=None,
           occurred_at=NOW - timedelta(days=1), direction=LedgerDirection.DEBIT):
    return LedgerEntry(
        id=id_,
        source_table="model_usage_details",
        source_id=id_,
        principal_id=principal_id,
        principal_name="acme",
        model_name="Qwen3-8B" if sku == SKU_TOKEN_PROMPT else None,
        request_id=request_id,
        sku=sku,
        quantity=Decimal(1000),
        unit=unit,
        unit_price=Decimal("0.001"),
        amount=Decimal(amount),
        currency="CNY",
        direction=direction,
        settle_mode=settle_mode,
        status=status,
        occurred_at=occurred_at,
        created_at=NOW,
        updated_at=NOW,
    )


def _redemption(code="CODE-100", amount="250", status=RedemptionStatus.ENABLED):
    return Redemption(
        code=code,
        amount=Decimal(amount),
        currency="CNY",
        status=status,
        batch="test",
        created_at=NOW,
        updated_at=NOW,
    )


async def _seed(engine, *rows):
    async with AsyncSession(engine) as s:
        for row in rows:
            s.add(row)
        await s.commit()


async def _wallet_of(engine, principal_id=ORG):
    async with AsyncSession(engine) as s:
        return (
            await s.exec(select(Wallet).where(Wallet.principal_id == principal_id))
        ).first()


# ---------------------------------------------------------------------------
# Admin wallet views
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_wallets_carries_available_and_what_is_owed(client, app_and_engine):
    _, engine = app_and_engine
    await _seed(
        engine,
        _wallet(balance="100.00", frozen="20.00"),
        _entry(1, amount="5.00", settle_mode=SettleMode.REALTIME),
        _entry(2, amount="7.00", settle_mode=SettleMode.DEFERRED,
               sku=SKU_910B, unit=UNIT_GPU_HOURS),
        Invoice(
            id=1, principal_id=ORG, period_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 9, 1, tzinfo=timezone.utc), amount=Decimal("30"),
            currency="CNY", status="issued", issued_at=NOW, created_at=NOW,
            updated_at=NOW,
        ),
    )

    response = await client.get("/billing/wallets")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pagination"]["total"] == 1
    wallet = body["items"][0]
    # Derived server-side: a client that computes this itself will eventually
    # compute it differently.
    assert Decimal(wallet["available"]) == Decimal("80.00")
    # Split the way the two pipelines collect it, because "how much do I top up"
    # and "when does service come back" have different answers.
    assert Decimal(wallet["outstanding_realtime"]) == Decimal("5.00")
    assert Decimal(wallet["outstanding_invoices"]) == Decimal("30")
    assert Decimal(wallet["outstanding_total"]) == Decimal("35.00")


@pytest.mark.asyncio
async def test_list_wallets_filters_by_principal_and_suspension(
    client, app_and_engine
):
    _, engine = app_and_engine
    await _seed(
        engine,
        _wallet(ORG, balance="10"),
        _wallet(OTHER_ORG, balance="0", suspended=True, name="globex"),
    )

    one = await client.get("/billing/wallets", params={"principal_id": OTHER_ORG})
    assert one.json()["pagination"]["total"] == 1
    assert one.json()["items"][0]["principal_name"] == "globex"

    suspended = await client.get("/billing/wallets", params={"suspended": True})
    assert suspended.json()["pagination"]["total"] == 1

    searched = await client.get("/billing/wallets", params={"search": "glob"})
    assert searched.json()["pagination"]["total"] == 1


@pytest.mark.asyncio
async def test_get_one_wallet_and_404(client, app_and_engine):
    _, engine = app_and_engine
    await _seed(engine, _wallet(ORG, balance="42.50"))

    found = await client.get(f"/billing/wallets/{ORG}")
    assert found.status_code == 200, found.text
    assert Decimal(found.json()["balance"]) == Decimal("42.50")

    missing = await client.get(f"/billing/wallets/{ORG + 5}")
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# Tenant's own wallet
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_tenant_reads_the_wallet_the_context_names(client, app_and_engine):
    """No principal id in the query: whose wallet this is comes from the caller."""
    _, engine = app_and_engine
    await _seed(
        engine, _wallet(ORG, balance="77.00"), _wallet(OTHER_ORG, balance="1.00")
    )

    response = await client.get("/billing/wallet")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["principal_id"] == ORG
    assert Decimal(body["balance"]) == Decimal("77.00")


@pytest.mark.asyncio
async def test_a_principal_with_no_billing_yet_gets_a_zero_state_not_a_404(client):
    """A wallet is created on first use, so "no billing yet" is a state to show
    and not an error to raise."""
    response = await client.get("/billing/wallet")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["principal_id"] == ORG
    assert Decimal(body["balance"]) == Decimal(0)
    assert body["suspended"] is False


# ---------------------------------------------------------------------------
# Redemption
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redeeming_a_code_credits_the_callers_wallet(client, app_and_engine):
    _, engine = app_and_engine
    await _seed(engine, _wallet(ORG, balance="10.00"), _redemption(amount="250"))

    response = await client.post("/billing/redeem", json={"code": "CODE-100"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert Decimal(body["amount"]) == Decimal("250")
    # The effect of the code, in the same response: a tenant should not have to
    # ask a second question to learn whether it worked.
    assert Decimal(body["balance"]) == Decimal("260.00")
    assert body["suspended"] is False
    assert Decimal((await _wallet_of(engine)).balance) == Decimal("260.00")


@pytest.mark.asyncio
async def test_a_code_can_only_be_spent_once(client, app_and_engine):
    _, engine = app_and_engine
    await _seed(engine, _wallet(ORG, balance="10.00"), _redemption(amount="250"))

    first = await client.post("/billing/redeem", json={"code": "CODE-100"})
    second = await client.post("/billing/redeem", json={"code": "CODE-100"})

    assert first.status_code == 200
    assert second.status_code >= 400
    assert Decimal((await _wallet_of(engine)).balance) == Decimal("260.00")


@pytest.mark.asyncio
async def test_an_unknown_code_is_refused(client):
    response = await client.post("/billing/redeem", json={"code": "NOPE"})
    assert response.status_code >= 400


@pytest.mark.asyncio
async def test_redeeming_resumes_a_suspended_wallet(client, app_and_engine):
    """The point of the endpoint from the tenant's side: service comes back."""
    _, engine = app_and_engine
    await _seed(
        engine,
        _wallet(ORG, balance="0", suspended=True),
        _redemption(amount="500"),
        ApiKey(
            id=1, name="k", access_key="ak-1", hashed_secret_key="argon2-1",
            scope=[], user_id=ORG, owner_principal_id=ORG, is_custom=False,
            suspended=True, suspension_reason="billing:wallet balance exhausted",
            created_at=NOW, updated_at=NOW,
        ),
    )

    response = await client.post("/billing/redeem", json={"code": "CODE-100"})

    assert response.status_code == 200, response.text
    assert response.json()["suspended"] is False
    async with AsyncSession(engine) as s:
        key = (await s.exec(select(ApiKey))).first()
    assert key.suspended is False


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_ledger_answers_the_questions_a_reconciliation_asks(
    client, app_and_engine
):
    _, engine = app_and_engine
    await _seed(
        engine,
        _entry(1, request_id="req-aaa", amount="1.00"),
        _entry(2, request_id="req-bbb", amount="2.00",
               settle_mode=SettleMode.DEFERRED, sku=SKU_910B, unit=UNIT_GPU_HOURS),
        _entry(3, request_id="req-ccc", amount="3.00", principal_id=OTHER_ORG),
        _entry(4, request_id="req-ddd", amount="4.00",
               status=LedgerStatus.SETTLED),
    )

    everything = await client.get("/billing/ledger")
    assert everything.json()["pagination"]["total"] == 4

    by_org = await client.get("/billing/ledger", params={"principal_id": OTHER_ORG})
    assert by_org.json()["pagination"]["total"] == 1

    by_sku = await client.get("/billing/ledger", params={"sku": SKU_910B})
    assert by_sku.json()["pagination"]["total"] == 1
    assert by_sku.json()["items"][0]["unit"] == UNIT_GPU_HOURS

    by_mode = await client.get(
        "/billing/ledger", params={"settle_mode": SettleMode.DEFERRED.value}
    )
    assert by_mode.json()["pagination"]["total"] == 1

    by_status = await client.get(
        "/billing/ledger", params={"status": LedgerStatus.SETTLED.value}
    )
    assert by_status.json()["pagination"]["total"] == 1


@pytest.mark.asyncio
async def test_a_request_id_is_matched_exactly(client, app_and_engine):
    """It is the handle a dispute arrives with; a partial match would return rows
    from somebody else's request."""
    _, engine = app_and_engine
    await _seed(
        engine,
        _entry(1, request_id="req-aaa"),
        _entry(2, request_id="req-aaa-2"),
    )

    response = await client.get("/billing/ledger", params={"request_id": "req-aaa"})

    assert response.json()["pagination"]["total"] == 1
    assert response.json()["items"][0]["request_id"] == "req-aaa"


@pytest.mark.asyncio
async def test_the_time_range_is_half_open(client, app_and_engine):
    """So paging a period in chunks neither drops nor duplicates a boundary row."""
    _, engine = app_and_engine
    boundary = datetime(2026, 9, 20, 0, 0, 0)
    await _seed(
        engine,
        _entry(1, occurred_at=boundary - timedelta(seconds=1)),
        _entry(2, occurred_at=boundary),
        _entry(3, occurred_at=boundary + timedelta(seconds=1)),
    )

    response = await client.get(
        "/billing/ledger",
        params={
            "occurred_after": boundary.isoformat(),
            "occurred_before": (boundary + timedelta(seconds=1)).isoformat(),
        },
    )

    ids = [item["id"] for item in response.json()["items"]]
    assert ids == [2]


@pytest.mark.asyncio
async def test_ledger_rows_carry_the_price_snapshot_and_source(client, app_and_engine):
    """What makes a charge reproducible after its price has been superseded."""
    _, engine = app_and_engine
    await _seed(engine, _entry(1, request_id="req-aaa"))

    item = (await client.get("/billing/ledger")).json()["items"][0]

    assert item["source_table"] == "model_usage_details"
    assert item["source_id"] == 1
    assert item["request_id"] == "req-aaa"
    assert Decimal(item["unit_price"]) == Decimal("0.001")
    assert item["direction"] == LedgerDirection.DEBIT.value


# ---------------------------------------------------------------------------
# Adjustments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_adjustment_credits_the_wallet_and_records_its_author(
    client, app_and_engine
):
    _, engine = app_and_engine
    await _seed(engine, _wallet(ORG, balance="100.00"))

    response = await client.post(
        "/billing/adjustments",
        json={
            "principal_id": ORG,
            "amount": "50.00",
            "reason": "billing error corrected",
            "idempotency_key": "TICKET-1",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert Decimal(body["amount"]) == Decimal("50.00")
    assert body["idempotency_key"] == "TICKET-1"
    assert body["ledger_entry_id"] is not None
    assert Decimal((await _wallet_of(engine)).balance) == Decimal("150.00")


@pytest.mark.asyncio
async def test_the_author_is_the_caller_and_not_a_field_in_the_body(
    client, app_and_engine
):
    """An adjustment whose operator could be named by whoever sent the request
    would be an anonymous money movement with a trail that proves nothing."""
    _, engine = app_and_engine
    await _seed(engine, _wallet(ORG, balance="100.00"))

    response = await client.post(
        "/billing/adjustments",
        json={
            "principal_id": ORG,
            "amount": "10.00",
            "reason": "x",
            "idempotency_key": "TICKET-2",
            "operator_id": 999,
            "operator_name": "somebody-else",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["operator_id"] == OPERATOR
    assert response.json()["operator_name"] == "admin"


@pytest.mark.asyncio
async def test_a_retried_adjustment_moves_money_once(client, app_and_engine):
    _, engine = app_and_engine
    await _seed(engine, _wallet(ORG, balance="100.00"))
    payload = {
        "principal_id": ORG,
        "amount": "50.00",
        "reason": "billing error corrected",
        "idempotency_key": "TICKET-3",
    }

    first = await client.post("/billing/adjustments", json=payload)
    second = await client.post("/billing/adjustments", json=payload)

    assert first.status_code == second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert Decimal((await _wallet_of(engine)).balance) == Decimal("150.00")
    async with AsyncSession(engine) as s:
        assert len(list((await s.exec(select(Adjustment))).all())) == 1


@pytest.mark.asyncio
async def test_a_key_reused_for_a_different_amount_is_refused(client, app_and_engine):
    """Not a retry — a mistake, and the loud kind is the safe kind."""
    _, engine = app_and_engine
    await _seed(engine, _wallet(ORG, balance="1000.00"))

    first = await client.post(
        "/billing/adjustments",
        json={
            "principal_id": ORG, "amount": "50.00", "reason": "x",
            "idempotency_key": "TICKET-4",
        },
    )
    second = await client.post(
        "/billing/adjustments",
        json={
            "principal_id": ORG, "amount": "5000.00", "reason": "x",
            "idempotency_key": "TICKET-4",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 422, second.text
    assert "already used" in second.json()["message"]
    assert Decimal((await _wallet_of(engine)).balance) == Decimal("1050.00")


@pytest.mark.asyncio
async def test_a_debit_beyond_the_balance_is_refused(client, app_and_engine):
    _, engine = app_and_engine
    await _seed(engine, _wallet(ORG, balance="10.00"))

    response = await client.post(
        "/billing/adjustments",
        json={
            "principal_id": ORG, "amount": "-500.00", "reason": "clawback",
            "idempotency_key": "TICKET-5",
        },
    )

    assert response.status_code == 422, response.text
    assert Decimal((await _wallet_of(engine)).balance) == Decimal("10.00")


@pytest.mark.asyncio
async def test_an_adjustment_needs_a_key_and_a_reason(client):
    missing_key = await client.post(
        "/billing/adjustments",
        json={"principal_id": ORG, "amount": "10.00", "reason": "x"},
    )
    blank_reason = await client.post(
        "/billing/adjustments",
        json={
            "principal_id": ORG, "amount": "10.00", "reason": "",
            "idempotency_key": "TICKET-6",
        },
    )

    assert missing_key.status_code == 422
    assert blank_reason.status_code == 422


@pytest.mark.asyncio
async def test_adjustments_are_listed_and_filterable(client, app_and_engine):
    _, engine = app_and_engine
    await _seed(engine, _wallet(ORG, balance="1000.00"))
    for index, key in enumerate(("TICKET-7", "TICKET-8")):
        await client.post(
            "/billing/adjustments",
            json={
                "principal_id": ORG if index == 0 else OTHER_ORG,
                "amount": "10.00",
                "reason": "goodwill credit",
                "idempotency_key": key,
            },
        )

    everything = await client.get("/billing/adjustments")
    assert everything.json()["pagination"]["total"] == 2

    by_principal = await client.get(
        "/billing/adjustments", params={"principal_id": OTHER_ORG}
    )
    assert by_principal.json()["pagination"]["total"] == 1

    by_key = await client.get(
        "/billing/adjustments", params={"idempotency_key": "TICKET-7"}
    )
    assert by_key.json()["pagination"]["total"] == 1

    searched = await client.get("/billing/adjustments", params={"search": "goodwill"})
    assert searched.json()["pagination"]["total"] == 2


@pytest.mark.asyncio
async def test_an_adjustment_writes_a_ledger_row_the_audit_can_follow(
    client, app_and_engine
):
    _, engine = app_and_engine
    await _seed(engine, _wallet(ORG, balance="100.00"))

    created = await client.post(
        "/billing/adjustments",
        json={
            "principal_id": ORG, "amount": "25.00", "reason": "ticket 4711",
            "idempotency_key": "TICKET-9",
        },
    )
    adjustment_id = created.json()["id"]

    rows = (
        await client.get(
            "/billing/ledger",
            params={"sku": SKU_WALLET_ADJUSTMENT},
        )
    ).json()["items"]

    assert len(rows) == 1
    assert rows[0]["id"] == created.json()["ledger_entry_id"]
    assert rows[0]["source_table"] == "billing_adjustment"
    assert rows[0]["source_id"] == adjustment_id
    assert rows[0]["direction"] == LedgerDirection.CREDIT.value
    # Settled at once, so the realtime settler never tries to take it again.
    assert rows[0]["status"] == LedgerStatus.SETTLED.value


@pytest.mark.asyncio
async def test_a_top_up_lands_in_the_ledger_as_a_credit(client, app_and_engine):
    """And, per the quota rules, must never count towards a spend ceiling."""
    _, engine = app_and_engine
    await _seed(engine, _wallet(ORG, balance="0"), _redemption(amount="100"))

    await client.post("/billing/redeem", json={"code": "CODE-100"})

    rows = (await client.get("/billing/ledger", params={"sku": SKU_WALLET_TOPUP})).json()[
        "items"
    ]
    assert len(rows) == 1
    assert rows[0]["direction"] == LedgerDirection.CREDIT.value
