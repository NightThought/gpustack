"""Manual wallet corrections (WP4.5).

This is the only way money enters or leaves a wallet that a human initiates, so
the properties under test are the ones that stop a mistake from becoming a
silent one:

* one idempotency key moves money once — a retried request returns the original
  adjustment rather than crediting again, and a key reused for *different*
  parameters is refused rather than quietly ignored, because the second case is a
  mistake and not a retry;
* a correction always produces a ledger row, linked both ways, so a wallet
  movement can be traced to the entry that explains it and to the operator who
  made it;
* a debit the balance cannot cover is refused rather than recorded — a prepaid
  wallet does not go negative because an operator typed the wrong sign;
* a credit that clears the arrears resumes the org in the same call, so the
  tenant sees service return rather than a balance that moved.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import InvalidException
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.billing import (
    SKU_WALLET_ADJUSTMENT,
    Adjustment,
    Invoice,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    UNIT_CURRENCY,
    Wallet,
)
from gpustack.server import billing_settlement
from gpustack.server.billing_alerts import billing_alerts
from gpustack.server.billing_settlement import SOURCE_ADJUSTMENT, apply_adjustment

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
ORG = 990101
OPERATOR = 7


@pytest_asyncio.fixture
async def engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        for model in (Wallet, LedgerEntry, Adjustment, Invoice, ApiKey):
            await conn.run_sync(model.__table__.create)
    billing_alerts.clear()
    yield engine
    billing_alerts.clear()
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    @asynccontextmanager
    async def _factory():
        async with AsyncSession(engine, expire_on_commit=False) as s:
            yield s

    with patch.object(billing_settlement, "async_session", _factory):
        yield _factory


async def _seed(factory, *rows):
    async with factory() as s:
        for row in rows:
            s.add(row)
        await s.commit()


def _wallet(balance="100", suspended=False):
    return Wallet(
        principal_id=ORG,
        balance=Decimal(balance),
        suspended=suspended,
        suspended_at=NOW if suspended else None,
        created_at=NOW,
        updated_at=NOW,
    )


async def _wallet_of(engine):
    async with AsyncSession(engine) as s:
        return (await s.exec(select(Wallet))).first()


async def _adjustments(engine):
    async with AsyncSession(engine) as s:
        return list((await s.exec(select(Adjustment))).all())


async def _entries(engine):
    async with AsyncSession(engine) as s:
        return list((await s.exec(select(LedgerEntry))).all())


async def _adjust(session, amount="50", key="TICKET-1", **kwargs):
    kwargs.setdefault("reason", "billing error corrected")
    kwargs.setdefault("operator_id", OPERATOR)
    kwargs.setdefault("operator_name", "admin")
    kwargs.setdefault("principal_name", "acme")
    return await apply_adjustment(
        session,
        principal_id=kwargs.pop("principal_id", ORG),
        amount=Decimal(amount),
        idempotency_key=key,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The money moves, and the trail explains it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_credit_moves_the_wallet_and_writes_its_ledger_row(
    engine, session_factory
):
    await _seed(session_factory, _wallet(balance="100"))

    async with session_factory() as s:
        adjustment = await _adjust(s, amount="50")

    assert Decimal(adjustment.amount) == Decimal("50")
    assert adjustment.wallet_id is not None
    assert adjustment.ledger_entry_id is not None
    assert (await _wallet_of(engine)).balance == Decimal("150")

    entries = await _entries(engine)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.id == adjustment.ledger_entry_id
    # Both directions of the link, because reconciliation walks it both ways:
    # from the correction to the money, and from the money to its explanation.
    assert entry.source_table == SOURCE_ADJUSTMENT
    assert entry.source_id == adjustment.id
    assert entry.sku == SKU_WALLET_ADJUSTMENT
    assert entry.unit == UNIT_CURRENCY
    assert entry.direction == LedgerDirection.CREDIT.value
    assert Decimal(entry.amount) == Decimal("50")
    # Settled at once: a correction is not a charge awaiting collection, and a
    # PENDING one would be picked up by the realtime settler and taken twice.
    assert entry.status == LedgerStatus.SETTLED.value
    assert entry.user_id == OPERATOR


@pytest.mark.asyncio
async def test_a_debit_takes_from_the_wallet_and_records_the_sign(
    engine, session_factory
):
    await _seed(session_factory, _wallet(balance="100"))

    async with session_factory() as s:
        adjustment = await _adjust(s, amount="-30", key="TICKET-2")

    assert Decimal(adjustment.amount) == Decimal("-30")
    assert (await _wallet_of(engine)).balance == Decimal("70")
    entry = (await _entries(engine))[0]
    assert entry.direction == LedgerDirection.DEBIT.value
    # The ledger stores a magnitude and lets direction carry the sign, as every
    # other entry does; a negative amount here would make sums come out doubled.
    assert Decimal(entry.amount) == Decimal("30")


@pytest.mark.asyncio
async def test_a_debit_beyond_the_balance_is_refused_not_recorded(
    engine, session_factory
):
    """A prepaid wallet does not go negative because an operator mistyped."""
    await _seed(session_factory, _wallet(balance="10"))

    with pytest.raises(InvalidException) as exc_info:
        async with session_factory() as s:
            await _adjust(s, amount="-500", key="TICKET-3")

    assert "cannot be overdrawn" in exc_info.value.message
    assert (await _wallet_of(engine)).balance == Decimal("10")
    # Nothing was left behind to explain a movement that never happened.
    assert await _adjustments(engine) == []
    assert await _entries(engine) == []


@pytest.mark.asyncio
async def test_a_wallet_is_created_for_a_principal_that_has_none(
    engine, session_factory
):
    """A first correction is often the first money an org ever had."""
    async with session_factory() as s:
        await _adjust(s, amount="25", key="TICKET-4")

    wallet = await _wallet_of(engine)
    assert wallet is not None
    assert wallet.principal_id == ORG
    assert wallet.balance == Decimal("25")


@pytest.mark.asyncio
async def test_the_operator_and_reason_are_recorded(engine, session_factory):
    await _seed(session_factory, _wallet())

    async with session_factory() as s:
        await _adjust(s, amount="10", key="TICKET-5", reason="goodwill credit for outage")

    adjustment = (await _adjustments(engine))[0]
    assert adjustment.operator_id == OPERATOR
    assert adjustment.operator_name == "admin"
    assert adjustment.principal_name == "acme"
    assert adjustment.reason == "goodwill credit for outage"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replaying_the_same_request_moves_money_once(engine, session_factory):
    """The case a client retry actually produces: same key, same parameters."""
    await _seed(session_factory, _wallet(balance="100"))

    async with session_factory() as s:
        first = await _adjust(s, amount="50", key="TICKET-6")
    async with session_factory() as s:
        second = await _adjust(s, amount="50", key="TICKET-6")

    assert second.id == first.id
    assert (await _wallet_of(engine)).balance == Decimal("150")
    assert len(await _adjustments(engine)) == 1
    assert len(await _entries(engine)) == 1


@pytest.mark.asyncio
async def test_a_key_reused_for_a_different_amount_is_refused(
    engine, session_factory
):
    """Not a retry — a mistake, and the loud kind is the safe kind.

    Silently returning the original would leave an operator believing a 500
    correction had been applied when 50 had been.
    """
    await _seed(session_factory, _wallet(balance="1000"))

    async with session_factory() as s:
        await _adjust(s, amount="50", key="TICKET-7")

    with pytest.raises(InvalidException) as exc_info:
        async with session_factory() as s:
            await _adjust(s, amount="500", key="TICKET-7")

    assert "already used" in exc_info.value.message
    assert (await _wallet_of(engine)).balance == Decimal("1050")
    assert len(await _adjustments(engine)) == 1


@pytest.mark.asyncio
async def test_a_key_reused_for_another_principal_is_refused(
    engine, session_factory
):
    await _seed(session_factory, _wallet(balance="1000"))

    async with session_factory() as s:
        await _adjust(s, amount="50", key="TICKET-8")

    with pytest.raises(InvalidException):
        async with session_factory() as s:
            await _adjust(s, amount="50", key="TICKET-8", principal_id=ORG + 1)

    assert len(await _adjustments(engine)) == 1


@pytest.mark.asyncio
async def test_different_keys_are_different_corrections(engine, session_factory):
    await _seed(session_factory, _wallet(balance="100"))

    async with session_factory() as s:
        await _adjust(s, amount="10", key="TICKET-9")
    async with session_factory() as s:
        await _adjust(s, amount="10", key="TICKET-10")

    assert (await _wallet_of(engine)).balance == Decimal("120")
    assert len(await _adjustments(engine)) == 2


@pytest.mark.asyncio
async def test_a_zero_amount_is_refused(engine, session_factory):
    """It would write a ledger row that explains nothing."""
    await _seed(session_factory, _wallet(balance="100"))

    with pytest.raises(InvalidException) as exc_info:
        async with session_factory() as s:
            await _adjust(s, amount="0", key="TICKET-11")

    assert "must not be zero" in exc_info.value.message
    assert await _entries(engine) == []


@pytest.mark.asyncio
async def test_a_correction_without_a_reason_is_refused(engine, session_factory):
    await _seed(session_factory, _wallet(balance="100"))

    with pytest.raises(InvalidException) as exc_info:
        async with session_factory() as s:
            await _adjust(s, amount="10", key="TICKET-12", reason="   ")

    assert "reason is required" in exc_info.value.message


@pytest.mark.asyncio
async def test_a_correction_without_a_key_is_refused(engine, session_factory):
    await _seed(session_factory, _wallet(balance="100"))

    with pytest.raises(InvalidException):
        async with session_factory() as s:
            await _adjust(s, amount="10", key="  ")

    assert await _adjustments(engine) == []


# ---------------------------------------------------------------------------
# Interaction with suspension
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_credit_that_clears_the_arrears_resumes_the_org(
    engine, session_factory
):
    """The tenant sees service return, not just a balance that moved."""
    await _seed(
        session_factory,
        _wallet(balance="0", suspended=True),
        ApiKey(
            id=1,
            name="key1",
            access_key="ak-1",
            hashed_secret_key="argon2-1",
            scope=[],
            user_id=ORG,
            owner_principal_id=ORG,
            is_custom=False,
            suspended=True,
            suspension_reason="billing:wallet balance exhausted",
            created_at=NOW,
            updated_at=NOW,
        ),
    )

    async with session_factory() as s:
        await _adjust(s, amount="100", key="TICKET-13")

    wallet = await _wallet_of(engine)
    assert wallet.suspended is False
    async with AsyncSession(engine) as s:
        key = (await s.exec(select(ApiKey))).first()
    assert key.suspended is False


@pytest.mark.asyncio
async def test_a_credit_that_does_not_cover_an_invoice_leaves_the_org_suspended(
    engine, session_factory
):
    """Arrears are both halves; a top-up short of the statement resumes nobody."""
    await _seed(
        session_factory,
        _wallet(balance="0", suspended=True),
        Invoice(
            id=1,
            principal_id=ORG,
            period_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            amount=Decimal("500"),
            currency="CNY",
            status="issued",
            issued_at=NOW,
            created_at=NOW,
            updated_at=NOW,
        ),
    )

    async with session_factory() as s:
        await _adjust(s, amount="100", key="TICKET-14")

    assert (await _wallet_of(engine)).suspended is True


@pytest.mark.asyncio
async def test_a_correction_logs_at_warning(engine, session_factory, caplog):
    """Money moved by hand is worth a line in the log at a level people read."""
    await _seed(session_factory, _wallet(balance="100"))

    with caplog.at_level("WARNING", logger="gpustack.server.billing_settlement"):
        async with session_factory() as s:
            await _adjust(s, amount="50", key="TICKET-15")

    assert "manual adjustment" in caplog.text
    assert "TICKET-15" not in caplog.text or "admin" in caplog.text
