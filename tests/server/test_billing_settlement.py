"""Settlement tests (WP4).

The properties a money-moving module has to prove, against a real database:

* a debit is one conditional statement — the balance check and the deduction
  cannot be interleaved, so a wallet never goes negative;
* settlement is idempotent: a repeated sweep collects nothing twice;
* an insufficient balance settles the oldest charges, suspends the wallet, and
  leaves the rest PENDING so a top-up collects the arrears;
* deferred (resource) charges are untouched — they belong to the invoice;
* a redemption code credits exactly once however many times it is presented;
* a refund reverses a session once.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
import pytest_asyncio
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from unittest.mock import patch

from gpustack import envs
from gpustack.api.exceptions import AlreadyExistsException, InvalidException
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.billing import (
    Invoice,
    InvoiceStatus,
    SKU_TOKEN_COMPLETION,
    SKU_TOKEN_PROMPT,
    SKU_WALLET_TOPUP,
    UNIT_CURRENCY,
    BillingSession,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    Redemption,
    RedemptionStatus,
    SettleMode,
    Wallet,
)
from gpustack.server import billing_settlement
from gpustack.server.billing_rater import BillingMode
from gpustack.server.billing_settlement import (
    in_enforce_scope,
    enforce_scope,
    outstanding_charges,
    outstanding_invoices,
    BillingSettler,
    credit_wallet,
    debit_wallet,
    get_or_create_wallet,
    redeem_code,
    resume_wallet_if_funded,
    suspend_wallet,
)

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
ORG = 990101
ORG_B = 990102
OTHER_ORG = 990102


def _charge(
    id_,
    principal_id=ORG,
    amount="1.00",
    sku=SKU_TOKEN_PROMPT,
    settle_mode=SettleMode.REALTIME,
    occurred_at=NOW,
    status=LedgerStatus.PENDING,
    source_id=None,
):
    return LedgerEntry(
        id=id_,
        source_table="model_usage_details",
        source_id=source_id if source_id is not None else id_,
        principal_id=principal_id,
        sku=sku,
        quantity=Decimal(1000),
        unit="tokens",
        unit_price=Decimal("0.002"),
        amount=Decimal(amount),
        currency="CNY",
        direction=LedgerDirection.DEBIT,
        settle_mode=settle_mode,
        status=status,
        occurred_at=occurred_at,
        created_at=NOW,
        updated_at=NOW,
    )


def _wallet(principal_id=ORG, balance="100", suspended=False):
    return Wallet(
        principal_id=principal_id,
        balance=Decimal(balance),
        suspended=suspended,
        created_at=NOW,
        updated_at=NOW,
    )


def _code(id_=1, code="A" * 32, amount="50", status=RedemptionStatus.ENABLED,
          expires_at=None):
    return Redemption(
        id=id_,
        code=code,
        amount=Decimal(amount),
        currency="CNY",
        status=status,
        expires_at=expires_at,
        created_at=NOW,
        updated_at=NOW,
    )


@pytest_asyncio.fixture
async def engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        # ApiKey is here because suspending a wallet propagates to the keys that
        # spend from it — settlement is not complete without that half.
        for model in (
            Wallet,
            LedgerEntry,
            BillingSession,
            Redemption,
            ApiKey,
            # Resuming a suspension now weighs unpaid invoices as well as
            # realtime arrears, so the table has to exist for the check to run.
            Invoice,
        ):
            await conn.run_sync(model.__table__.create)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    @asynccontextmanager
    async def _factory():
        async with AsyncSession(engine, expire_on_commit=False) as s:
            yield s

    with patch.object(billing_settlement, "async_session", _factory):
        yield _factory


@pytest_asyncio.fixture
async def session(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s


async def _seed(factory, *rows):
    async with factory() as s:
        for row in rows:
            s.add(row)
        await s.commit()


async def _wallets(engine):
    async with AsyncSession(engine) as s:
        return list((await s.exec(select(Wallet))).all())


async def _wallets_of(session):
    """Same read, on a session the test already holds."""
    return list((await session.exec(select(Wallet))).all())


async def _ledger(engine, **filters):
    async with AsyncSession(engine) as s:
        rows = list((await s.exec(select(LedgerEntry))).all())
    for key, value in filters.items():
        rows = [r for r in rows if getattr(r, key) == value]
    return rows


async def _sessions(engine):
    async with AsyncSession(engine) as s:
        return list((await s.exec(select(BillingSession))).all())


# ---------------------------------------------------------------------------
# Wallet primitives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debit_takes_the_amount_when_covered(session):
    session.add(_wallet(balance="100"))
    await session.commit()

    assert await debit_wallet(session, ORG, Decimal("30")) is True
    await session.commit()
    assert (await _wallets_of(session))[0].balance == Decimal("70")


@pytest.mark.asyncio
async def test_debit_refuses_to_overdraw(session):
    session.add(_wallet(balance="10"))
    await session.commit()

    assert await debit_wallet(session, ORG, Decimal("30")) is False
    await session.commit()
    assert (await _wallets_of(session))[0].balance == Decimal("10")


@pytest.mark.asyncio
async def test_debit_rejects_a_non_positive_amount(session):
    """A negative debit would be a credit, and must never slip through."""
    session.add(_wallet(balance="10"))
    await session.commit()

    with pytest.raises(ValueError, match="must be positive"):
        await debit_wallet(session, ORG, Decimal("0"))
    with pytest.raises(ValueError, match="must be positive"):
        await debit_wallet(session, ORG, Decimal("-5"))


@pytest.mark.asyncio
async def test_repeated_debits_stop_at_the_balance(session):
    """The conditional UPDATE is what stands between concurrency and overdraft.

    Ten attempts of 10 against a balance of 70 must produce exactly seven
    successes and a final balance of zero — never negative, never eight.
    """
    session.add(_wallet(balance="70"))
    await session.commit()

    successes = 0
    for _ in range(10):
        if await debit_wallet(session, ORG, Decimal("10")):
            successes += 1
        await session.commit()

    assert successes == 7
    balance = (await _wallets_of(session))[0].balance
    assert balance == Decimal("0")


@pytest.mark.asyncio
async def test_credit_creates_the_wallet_on_first_use(session):
    await credit_wallet(session, ORG, Decimal("25"))
    await session.commit()

    wallets = await _wallets_of(session)
    assert len(wallets) == 1
    assert wallets[0].balance == Decimal("25")


@pytest.mark.asyncio
async def test_get_or_create_wallet_is_idempotent(session):
    first = await get_or_create_wallet(session, ORG)
    second = await get_or_create_wallet(session, ORG)
    await session.commit()

    assert first.id == second.id
    assert len(await _wallets_of(session)) == 1


# ---------------------------------------------------------------------------
# Settlement sweep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settle_collects_pending_realtime_charges(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(balance="100"),
        _charge(1, amount="1.50"),
        _charge(2, amount="2.25", sku=SKU_TOKEN_COMPLETION),
    )

    report = await BillingSettler(mode=BillingMode.ENFORCE).settle_once()

    assert (report.principals, report.sessions) == (1, 1)
    assert report.entries_settled == 2
    assert report.amount_settled == Decimal("3.75")
    assert (await _wallets(engine))[0].balance == Decimal("96.25")

    entries = await _ledger(engine)
    assert all(e.status == LedgerStatus.SETTLED.value for e in entries)
    assert all(e.settled_at is not None for e in entries)
    assert {e.billing_session_id for e in entries} == {
        (await _sessions(engine))[0].id
    }
    settled_session = (await _sessions(engine))[0]
    assert settled_session.settled is True
    assert settled_session.actual == Decimal("3.75")
    assert settled_session.estimate == Decimal("3.75")


@pytest.mark.asyncio
async def test_settle_is_idempotent(engine, session_factory):
    await _seed(
        session_factory, _wallet(balance="100"), _charge(1, amount="1.50")
    )
    settler = BillingSettler(mode=BillingMode.ENFORCE)

    first = await settler.settle_once()
    second = await settler.settle_once()

    assert first.entries_settled == 1
    assert (second.entries_settled, second.principals) == (0, 0)
    assert (await _wallets(engine))[0].balance == Decimal("98.50")


@pytest.mark.asyncio
async def test_shadow_mode_never_debits(engine, session_factory):
    """What shadow reconciles is what enforce would charge — but not yet."""
    await _seed(
        session_factory, _wallet(balance="100"), _charge(1, amount="1.50")
    )

    report = await BillingSettler(mode=BillingMode.SHADOW).settle_once()

    assert report.entries_settled == 0
    assert (await _wallets(engine))[0].balance == Decimal("100")
    assert (await _ledger(engine))[0].status == LedgerStatus.PENDING.value


@pytest.mark.asyncio
async def test_deferred_charges_wait_for_the_invoice(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(balance="100"),
        _charge(1, amount="1.50"),
        _charge(
            2,
            amount="100.00",
            sku="gpu.hour.910b",
            settle_mode=SettleMode.DEFERRED,
        ),
    )

    report = await BillingSettler(mode=BillingMode.ENFORCE).settle_once()

    assert report.entries_settled == 1
    assert report.deferred_skipped == 1
    assert (await _wallets(engine))[0].balance == Decimal("98.50")
    deferred = (await _ledger(engine, sku="gpu.hour.910b"))[0]
    assert deferred.status == LedgerStatus.PENDING.value


@pytest.mark.asyncio
async def test_insufficient_balance_settles_oldest_first_and_suspends(
    engine, session_factory
):
    await _seed(
        session_factory,
        _wallet(balance="2.50"),
        _charge(1, amount="1.00", occurred_at=NOW - timedelta(hours=3)),
        _charge(2, amount="1.00", occurred_at=NOW - timedelta(hours=2)),
        _charge(3, amount="1.00", occurred_at=NOW - timedelta(hours=1)),
    )

    report = await BillingSettler(mode=BillingMode.ENFORCE).settle_once()

    # 2.50 covers the two oldest; the third is arrears.
    assert report.entries_settled == 2
    assert report.entries_unpaid == 1
    assert report.amount_unpaid == Decimal("1.00")
    assert report.suspended == [ORG]
    assert (await _wallets(engine))[0].balance == Decimal("0.50")

    settled = {e.source_id for e in await _ledger(engine, status=LedgerStatus.SETTLED.value)}
    assert settled == {1, 2}
    pending = await _ledger(engine, status=LedgerStatus.PENDING.value)
    assert [e.source_id for e in pending] == [3]
    assert (await _wallets(engine))[0].suspended is True


@pytest.mark.asyncio
async def test_a_top_up_clears_the_suspension_and_collects_the_arrears(
    engine, session_factory
):
    await _seed(
        session_factory,
        _wallet(balance="0.50", suspended=True),
        _charge(3, amount="1.00"),
    )
    settler = BillingSettler(mode=BillingMode.ENFORCE)

    # Still short: nothing settles.
    first = await settler.settle_once()
    assert first.entries_settled == 0

    async with session_factory() as s:
        await credit_wallet(s, ORG, Decimal("10"))
        assert await resume_wallet_if_funded(s, ORG)
        await s.commit()

    second = await settler.settle_once()
    assert second.entries_settled == 1
    wallet = (await _wallets(engine))[0]
    assert wallet.suspended is False
    assert wallet.balance == Decimal("9.50")


@pytest.mark.asyncio
async def test_one_principal_failing_does_not_block_another(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(principal_id=ORG, balance="0.10"),
        _wallet(principal_id=OTHER_ORG, balance="50"),
        _charge(1, principal_id=ORG, amount="5.00"),
        _charge(2, principal_id=OTHER_ORG, amount="2.00"),
    )

    report = await BillingSettler(mode=BillingMode.ENFORCE).settle_once()

    assert report.principals == 2
    assert report.entries_settled == 1
    assert report.suspended == [ORG]
    wallets = {w.principal_id: w.balance for w in await _wallets(engine)}
    assert wallets[ORG] == Decimal("0.10")  # untouched: could not cover it
    assert wallets[OTHER_ORG] == Decimal("48.00")


@pytest.mark.asyncio
async def test_charge_with_no_payer_is_reported_not_settled(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(balance="100"),
        _charge(1, principal_id=None, amount="1.00"),
    )

    report = await BillingSettler(mode=BillingMode.ENFORCE).settle_once()

    assert report.entries_unpaid == 1
    assert report.entries_settled == 0
    assert (await _wallets(engine))[0].balance == Decimal("100")


@pytest.mark.asyncio
async def test_wallet_is_created_when_a_charge_arrives_first(engine, session_factory):
    """Rating can precede any top-up; settlement must not fail on a missing wallet."""
    await _seed(session_factory, _charge(1, amount="1.00"))

    report = await BillingSettler(mode=BillingMode.ENFORCE).settle_once()

    # Created with a zero balance, so the charge is arrears and the wallet is
    # suspended rather than the sweep erroring out.
    assert report.entries_settled == 0
    assert report.suspended == [ORG]
    wallets = await _wallets(engine)
    assert len(wallets) == 1
    assert wallets[0].balance == Decimal("0")


# ---------------------------------------------------------------------------
# Redemption
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redeem_credits_wallet_and_records_the_topup(engine, session_factory):
    await _seed(session_factory, _code(amount="50"))

    async with session_factory() as s:
        redemption = await redeem_code(s, code="A" * 32, principal_id=ORG, user_id=7)

    assert redemption.status == RedemptionStatus.USED.value
    assert redemption.used_by_principal_id == ORG
    assert (await _wallets(engine))[0].balance == Decimal("50")

    credits = await _ledger(engine, direction=LedgerDirection.CREDIT.value)
    assert len(credits) == 1
    assert credits[0].sku == SKU_WALLET_TOPUP
    assert credits[0].amount == Decimal("50")
    assert credits[0].status == LedgerStatus.SETTLED.value
    assert credits[0].unit == UNIT_CURRENCY
    assert credits[0].source_table == "billing_redemption"


@pytest.mark.asyncio
async def test_redeeming_twice_credits_once(engine, session_factory):
    await _seed(session_factory, _code(amount="50"))

    async with session_factory() as s:
        await redeem_code(s, code="A" * 32, principal_id=ORG)
    async with session_factory() as s:
        with pytest.raises(AlreadyExistsException) as exc_info:
            await redeem_code(s, code="A" * 32, principal_id=ORG)
    assert "already used" in exc_info.value.message

    assert (await _wallets(engine))[0].balance == Decimal("50")
    assert len(await _ledger(engine)) == 1


@pytest.mark.asyncio
async def test_redeem_rejects_unknown_and_expired_codes(engine, session_factory):
    await _seed(
        session_factory,
        _code(id_=2, code="B" * 32, expires_at=NOW - timedelta(days=1)),
    )

    async with session_factory() as s:
        with pytest.raises(InvalidException) as unknown:
            await redeem_code(s, code="nope", principal_id=ORG)
    assert "not found" in unknown.value.message

    async with session_factory() as s:
        with pytest.raises(InvalidException) as expired:
            await redeem_code(s, code="B" * 32, principal_id=ORG)
    assert "expired" in expired.value.message

    assert await _wallets(engine) == []


@pytest.mark.asyncio
async def test_redeem_clears_a_suspension(engine, session_factory):
    await _seed(
        session_factory, _wallet(balance="0", suspended=True), _code(amount="50")
    )

    async with session_factory() as s:
        await redeem_code(s, code="A" * 32, principal_id=ORG)

    wallet = (await _wallets(engine))[0]
    assert wallet.balance == Decimal("50")
    assert wallet.suspended is False
    assert wallet.suspended_at is None


# ---------------------------------------------------------------------------
# Refund
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refund_reverses_a_session_once(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(balance="100"),
        _charge(1, amount="1.50"),
        _charge(2, amount="2.00", sku=SKU_TOKEN_COMPLETION),
    )
    settler = BillingSettler(mode=BillingMode.ENFORCE)
    await settler.settle_once()
    session_id = (await _sessions(engine))[0].id
    assert (await _wallets(engine))[0].balance == Decimal("96.50")

    async with session_factory() as s:
        refunded = await settler.refund_session(s, session_id)
    assert refunded == Decimal("3.50")
    assert (await _wallets(engine))[0].balance == Decimal("100")
    assert all(
        e.status == LedgerStatus.PENDING.value for e in await _ledger(engine)
    )

    # A second refund must not pay out twice.
    async with session_factory() as s:
        assert await settler.refund_session(s, session_id) == Decimal(0)
    assert (await _wallets(engine))[0].balance == Decimal("100")


@pytest.mark.asyncio
async def test_refunding_an_unknown_session_is_rejected(engine, session_factory):
    settler = BillingSettler(mode=BillingMode.ENFORCE)
    async with session_factory() as s:
        with pytest.raises(InvalidException, ):
            await settler.refund_session(s, 424242)


# ---------------------------------------------------------------------------
# Suspension helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suspend_is_idempotent_and_keeps_the_first_timestamp(session):
    session.add(_wallet(balance="0"))
    await session.commit()

    await suspend_wallet(session, ORG)
    await session.commit()
    first_at = (await _wallets_of(session))[0].suspended_at

    await suspend_wallet(session, ORG)
    await session.commit()
    wallet = (await _wallets_of(session))[0]
    assert wallet.suspended is True
    assert wallet.suspended_at == first_at


@pytest.mark.asyncio
async def test_resume_refuses_while_the_balance_is_still_empty(session):
    session.add(_wallet(balance="0", suspended=True))
    await session.commit()

    assert await resume_wallet_if_funded(session, ORG) is False
    await session.commit()
    assert (await _wallets_of(session))[0].suspended is True


@pytest.mark.asyncio
async def test_an_unpaid_invoice_blocks_a_resume(engine, session_factory):
    """Arrears are both halves, not just the realtime one.

    Resource usage is collected by a statement (``billing_invoice``) rather than
    as it accrues, so an org can owe a bill with nothing pending on its realtime
    ledger. Resuming on the realtime figure alone would hand service back to an
    org that has not paid its statement, and nothing would notice until the next
    invoicing pass.
    """
    await _seed(
        session_factory,
        _wallet(balance="50", suspended=True),
        Invoice(
            id=1,
            principal_id=ORG,
            period_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            amount=Decimal("100.00"),
            currency="CNY",
            status=InvoiceStatus.ISSUED,
            issued_at=NOW,
            created_at=NOW,
            updated_at=NOW,
        ),
    )

    async with session_factory() as s:
        assert await outstanding_invoices(s, ORG) == Decimal("100.00")
        assert await outstanding_charges(s, ORG) == Decimal("100.00")
        # 50 on the wallet, 100 owed: still in arrears.
        assert await resume_wallet_if_funded(s, ORG) is False
        await s.commit()

    async with session_factory() as s:
        wallet = (await s.exec(select(Wallet))).first()
        assert wallet.suspended is True

    # Once the statement is paid, the same balance resumes the org.
    async with session_factory() as s:
        invoice = (await s.exec(select(Invoice))).first()
        invoice.status = InvoiceStatus.SETTLED
        s.add(invoice)
        await s.commit()

    async with session_factory() as s:
        assert await outstanding_invoices(s, ORG) == Decimal(0)
        assert await resume_wallet_if_funded(s, ORG) is True
        await s.commit()

    assert (await _wallets(engine))[0].suspended is False


@pytest.mark.asyncio
async def test_a_settled_invoice_is_not_counted_as_owed(engine, session_factory):
    """Only ISSUED statements are arrears; counting paid ones would keep an org
    suspended forever, which is the failure a tenant escalates immediately."""
    await _seed(
        session_factory,
        _wallet(balance="500"),
        Invoice(
            id=1,
            principal_id=ORG,
            period_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            amount=Decimal("100.00"),
            currency="CNY",
            status=InvoiceStatus.SETTLED,
            issued_at=NOW,
            settled_at=NOW,
            created_at=NOW,
            updated_at=NOW,
        ),
    )

    async with session_factory() as s:
        assert await outstanding_invoices(s, ORG) == Decimal(0)
        assert await outstanding_charges(s, ORG) == Decimal(0)


# ---------------------------------------------------------------------------
# The rollout whitelist: rating for everyone, charging a named few
# ---------------------------------------------------------------------------


def test_enforce_scope_parsing():
    assert enforce_scope("") is None  # empty means everybody
    assert enforce_scope("   ") is None
    assert enforce_scope("all") is None
    assert enforce_scope("*") is None
    assert enforce_scope("1, 2 ,3") == frozenset({1, 2, 3})
    # An unparseable list must not fall back to "everybody": that would be the
    # difference between charging nobody and charging everyone, decided by a typo.
    with pytest.raises(ValueError) as exc_info:
        enforce_scope("1,acme")
    assert "GPUSTACK_BILLING_ENFORCE_PRINCIPALS" in str(exc_info.value)


def test_in_enforce_scope():
    assert in_enforce_scope(7, None) is True
    assert in_enforce_scope(7, frozenset({7})) is True
    assert in_enforce_scope(8, frozenset({7})) is False
    assert in_enforce_scope(None, None) is False


@pytest.mark.asyncio
async def test_an_org_outside_the_whitelist_is_rated_but_not_charged(
    engine, session_factory, monkeypatch
):
    """The gradual-rollout property: its charges wait rather than vanish."""
    monkeypatch.setattr(envs, "BILLING_ENFORCE_PRINCIPALS", str(ORG))
    await _seed(
        session_factory,
        _wallet(ORG, balance="100"),
        _wallet(ORG_B, balance="100"),
        _charge(1, principal_id=ORG, amount="5.00"),
        _charge(2, principal_id=ORG_B, amount="7.00"),
    )

    report = await BillingSettler(mode=BillingMode.ENFORCE).settle_once()

    assert report.entries_settled == 1
    assert report.entries_out_of_scope == 1
    assert report.suspended == []
    wallets = {w.principal_id: w for w in await _wallets(engine)}
    assert wallets[ORG].balance == Decimal("95.00")
    # Untouched: not debited, not suspended.
    assert wallets[ORG_B].balance == Decimal("100")
    async with AsyncSession(engine) as s:
        rows = {e.id: e for e in (await s.exec(select(LedgerEntry))).all()}
    assert rows[1].status == LedgerStatus.SETTLED.value
    assert rows[2].status == LedgerStatus.PENDING.value


@pytest.mark.asyncio
async def test_widening_the_whitelist_collects_what_waited(
    engine, session_factory, monkeypatch
):
    """Turning the list into "everyone" is a switch, not a backfill."""
    monkeypatch.setattr(envs, "BILLING_ENFORCE_PRINCIPALS", str(ORG))
    await _seed(
        session_factory,
        _wallet(ORG_B, balance="100"),
        _charge(2, principal_id=ORG_B, amount="7.00"),
    )
    first = await BillingSettler(mode=BillingMode.ENFORCE).settle_once()
    assert first.entries_settled == 0

    monkeypatch.setattr(envs, "BILLING_ENFORCE_PRINCIPALS", "")
    second = await BillingSettler(mode=BillingMode.ENFORCE).settle_once()

    assert second.entries_settled == 1
    assert (await _wallets(engine))[0].balance == Decimal("93.00")
