"""Deferred billing: period invoicing and collection (WP6).

The behaviours that would be expensive to get wrong are the ones tested here:

* an invoice is the moment deferred money moves — issuing debits the wallet, and
  an org that cannot cover it is suspended exactly as one that cannot cover a
  realtime charge is;
* only *closed* periods are invoiced, and one statement per (org, period), so a
  pass that runs twice, or a server that restarts across a period boundary,
  neither double-charges nor skips;
* realtime (token) entries never reach an invoice — the two halves of the hybrid
  model stay disjoint, and a bug here would bill a request twice;
* usage rated after its period was invoiced is collected by the next statement
  rather than lost, and never back-edited onto one already issued;
* a period boundary is a half-open interval, so a charge lands in exactly one
  statement.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.billing import (
    SKU_GPU_HOUR_PREFIX,
    SKU_STORAGE_GB_HOUR,
    SKU_TOKEN_PROMPT,
    UNIT_GB_HOURS,
    UNIT_GPU_HOURS,
    UNIT_TOKENS,
    Invoice,
    InvoiceItem,
    InvoiceStatus,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    SettleMode,
    Wallet,
)
from gpustack.server import billing_invoice
from gpustack.server.billing_invoice import (
    BillingInvoicer,
    BillingPeriod,
    billing_period,
    closed_periods,
    period_bounds,
    shift_period,
)
from gpustack.server.billing_rater import BillingMode

# Naive UTC, the shape the rater writes and UTCDateTime stores.
NOW = datetime(2026, 9, 23, 14, 0, 0)
SEPTEMBER = datetime(2026, 9, 1)
OCTOBER = datetime(2026, 10, 1)
AUGUST = datetime(2026, 8, 1)

ORG_A = 990101
ORG_B = 990102
SKU_910B = SKU_GPU_HOUR_PREFIX + "910b"


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        for model in (LedgerEntry, Invoice, InvoiceItem, Wallet, ApiKey):
            await conn.run_sync(model.__table__.create)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    @asynccontextmanager
    async def _factory():
        async with AsyncSession(engine, expire_on_commit=False) as s:
            yield s

    with patch.object(billing_invoice, "async_session", _factory):
        yield _factory


@pytest_asyncio.fixture
async def session(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s


def _entry(
    id_,
    *,
    sku=SKU_910B,
    unit=UNIT_GPU_HOURS,
    quantity="12",
    amount="150.00",
    occurred_at=NOW - timedelta(days=1),
    principal_id=ORG_A,
    settle_mode=SettleMode.DEFERRED,
    status=LedgerStatus.PENDING,
    model_name=None,
    resource_name="worker-1",
    request_id=None,
    invoice_id=None,
):
    return LedgerEntry(
        id=id_,
        source_table="metered_usage",
        source_id=id_,
        principal_id=principal_id,
        principal_name=f"org-{principal_id}",
        model_name=model_name,
        resource_name=resource_name,
        request_id=request_id,
        sku=sku,
        quantity=Decimal(str(quantity)),
        unit=unit,
        unit_price=Decimal("12.5"),
        amount=Decimal(str(amount)),
        currency="CNY",
        direction=LedgerDirection.DEBIT,
        settle_mode=settle_mode,
        status=status,
        occurred_at=occurred_at,
        created_at=NOW,
        updated_at=NOW,
        invoice_id=invoice_id,
    )


def _wallet(principal_id=ORG_A, balance="1000", suspended=False):
    return Wallet(
        principal_id=principal_id,
        balance=Decimal(balance),
        suspended=suspended,
        created_at=NOW,
        updated_at=NOW,
    )


async def _seed(factory, *rows):
    async with factory() as s:
        for row in rows:
            s.add(row)
        await s.commit()


async def _invoices(engine):
    async with AsyncSession(engine) as s:
        return list((await s.exec(select(Invoice))).all())


async def _items(engine, invoice_id=None):
    async with AsyncSession(engine) as s:
        rows = (await s.exec(select(InvoiceItem))).all()
    return [r for r in rows if invoice_id is None or r.invoice_id == invoice_id]


async def _entries(engine):
    async with AsyncSession(engine) as s:
        return {e.id: e for e in (await s.exec(select(LedgerEntry))).all()}


async def _wallets(engine):
    async with AsyncSession(engine) as s:
        return {w.principal_id: w for w in (await s.exec(select(Wallet))).all()}


def _utc(moment: datetime) -> datetime:
    """The shape the period helpers return: aware UTC.

    They normalise their input with ``as_utc``, so a naive expectation would
    compare unequal to a correct answer.
    """
    return moment.replace(tzinfo=timezone.utc)


def _invoicer(**kwargs) -> BillingInvoicer:
    kwargs.setdefault("mode", BillingMode.ENFORCE)
    kwargs.setdefault("period", BillingPeriod.MONTHLY)
    kwargs.setdefault("cron", "0 3 1 * *")
    return BillingInvoicer(**kwargs)


# ---------------------------------------------------------------------------
# Period arithmetic
# ---------------------------------------------------------------------------


def test_period_comes_from_the_environment_and_refuses_a_typo():
    assert billing_period("daily") is BillingPeriod.DAILY
    assert billing_period("MONTHLY") is BillingPeriod.MONTHLY
    with pytest.raises(ValueError) as exc_info:
        billing_period("weekly")
    assert "GPUSTACK_BILLING_INVOICE_PERIOD" in str(exc_info.value)


def test_monthly_bounds_are_the_calendar_month():
    start, end = period_bounds(BillingPeriod.MONTHLY, datetime(2026, 9, 23, 14, 0))
    assert (start, end) == (_utc(SEPTEMBER), _utc(OCTOBER))


def test_daily_bounds_are_the_utc_day():
    start, end = period_bounds(BillingPeriod.DAILY, datetime(2026, 9, 23, 14, 0))
    assert (start, end) == (_utc(datetime(2026, 9, 23)), _utc(datetime(2026, 9, 24)))


def test_an_aware_instant_is_normalised_before_bounds_are_taken():
    aware = datetime(2026, 9, 23, 14, 0, tzinfo=timezone.utc)
    assert period_bounds(BillingPeriod.MONTHLY, aware) == (
        _utc(SEPTEMBER),
        _utc(OCTOBER),
    )


def test_the_in_flight_period_is_never_closed():
    """Issuing it would bill partial usage and — being idempotent per period —
    never bill the rest."""
    periods = closed_periods(
        BillingPeriod.MONTHLY, now=datetime(2026, 9, 23), lookback=3
    )
    assert [start for start, _ in periods] == [
        _utc(datetime(2026, 6, 1)),
        _utc(datetime(2026, 7, 1)),
        _utc(AUGUST),
    ]
    assert all(end <= _utc(SEPTEMBER) for _, end in periods)


def test_closed_periods_are_oldest_first():
    """So a pass after downtime collects old usage into the oldest statement it
    issues, rather than leaving it behind."""
    periods = closed_periods(BillingPeriod.DAILY, now=datetime(2026, 9, 23), lookback=3)
    assert [start for start, _ in periods] == [
        _utc(datetime(2026, 9, 20)),
        _utc(datetime(2026, 9, 21)),
        _utc(datetime(2026, 9, 22)),
    ]


def test_shifting_months_crosses_a_year_boundary():
    assert shift_period(BillingPeriod.MONTHLY, datetime(2026, 12, 1), 1) == (
        datetime(2027, 1, 1)
    )
    assert shift_period(BillingPeriod.MONTHLY, datetime(2027, 1, 1), -1) == (
        datetime(2026, 12, 1)
    )


def test_a_non_positive_lookback_is_refused():
    with pytest.raises(ValueError):
        closed_periods(BillingPeriod.MONTHLY, now=NOW, lookback=0)


def test_an_unparsable_cron_fails_at_construction():
    """Not at the first fire, which for a monthly cron is up to a month away."""
    with pytest.raises(ValueError) as exc_info:
        BillingInvoicer(mode=BillingMode.ENFORCE, cron="not a cron")
    assert "cron" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Issuing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_closed_period_becomes_one_paid_statement(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(balance="1000"),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=1)
        ),
        _entry(
            2, quantity="8", amount="100.00", occurred_at=AUGUST + timedelta(hours=2)
        ),
    )

    report = await _invoicer().invoice_once(now=NOW)

    assert report.invoices_issued == 1
    assert report.entries_invoiced == 2
    assert report.amount_invoiced == Decimal("250.00")
    assert report.amount_collected == Decimal("250.00")

    invoices = await _invoices(engine)
    assert len(invoices) == 1
    invoice = invoices[0]
    assert invoice.status == InvoiceStatus.SETTLED.value
    assert invoice.principal_id == ORG_A
    assert invoice.principal_name == f"org-{ORG_A}"
    assert Decimal(invoice.amount) == Decimal("250.00")
    # The closed period, not the one the pass ran in.
    assert invoice.period_start.replace(tzinfo=None) == AUGUST
    assert invoice.period_end.replace(tzinfo=None) == SEPTEMBER
    assert invoice.settled_at is not None

    wallets = await _wallets(engine)
    assert wallets[ORG_A].balance == Decimal("750.00")
    assert wallets[ORG_A].suspended is False

    entries = await _entries(engine)
    assert all(e.status == LedgerStatus.SETTLED.value for e in entries.values())
    assert all(e.invoice_id == invoice.id for e in entries.values())


@pytest.mark.asyncio
async def test_items_are_aggregated_per_sku_and_model(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(balance="10000"),
        _entry(
            1,
            sku=SKU_910B,
            quantity="10",
            amount="125.00",
            occurred_at=AUGUST + timedelta(hours=1),
        ),
        _entry(
            2,
            sku=SKU_910B,
            quantity="6",
            amount="75.00",
            occurred_at=AUGUST + timedelta(hours=2),
        ),
        _entry(
            3,
            sku=SKU_STORAGE_GB_HOUR,
            unit=UNIT_GB_HOURS,
            quantity="100",
            amount="0.01",
            occurred_at=AUGUST + timedelta(hours=3),
        ),
        # Same SKU, different model: a separate line, because a tenant reading a
        # statement asks which deployment the hours belong to.
        _entry(
            4,
            sku=SKU_910B,
            quantity="2",
            amount="25.00",
            model_name="Qwen3-8B",
            occurred_at=AUGUST + timedelta(hours=4),
        ),
    )

    await _invoicer(period=BillingPeriod.MONTHLY).invoice_once(now=datetime(2026, 9, 5))

    invoice = (await _invoices(engine))[0]
    lines = await _items(engine, invoice.id)
    by_key = {(line.sku, line.model_name): line for line in lines}

    assert len(lines) == 3
    assert Decimal(by_key[(SKU_910B, None)].quantity) == Decimal("16")
    assert Decimal(by_key[(SKU_910B, None)].amount) == Decimal("200.00")
    assert by_key[(SKU_910B, None)].entry_count == 2
    assert by_key[(SKU_910B, None)].unit == UNIT_GPU_HOURS
    assert Decimal(by_key[(SKU_STORAGE_GB_HOUR, None)].amount) == Decimal("0.01")
    assert Decimal(by_key[(SKU_910B, "Qwen3-8B")].amount) == Decimal("25.00")
    # The lines sum to the statement, which is what makes it verifiable.
    assert sum((Decimal(line.amount) for line in lines), Decimal(0)) == Decimal(
        invoice.amount
    )


@pytest.mark.asyncio
async def test_realtime_token_charges_are_never_invoiced(engine, session_factory):
    """The two halves of the hybrid model stay disjoint: a token charge is
    settled in real time, so invoicing it as well would bill one request twice."""
    await _seed(
        session_factory,
        _wallet(balance="10000"),
        _entry(
            1,
            sku=SKU_TOKEN_PROMPT,
            unit=UNIT_TOKENS,
            quantity="1000",
            amount="2.00",
            settle_mode=SettleMode.REALTIME,
            occurred_at=AUGUST + timedelta(hours=1),
        ),
        _entry(
            2, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=2)
        ),
    )

    report = await _invoicer().invoice_once(now=datetime(2026, 9, 5))

    assert report.entries_invoiced == 1
    assert report.amount_invoiced == Decimal("150.00")
    entries = await _entries(engine)
    assert entries[1].status == LedgerStatus.PENDING.value
    assert entries[1].invoice_id is None
    assert entries[2].status == LedgerStatus.SETTLED.value
    assert (await _wallets(engine))[ORG_A].balance == Decimal("9850.00")


@pytest.mark.asyncio
async def test_unpriced_placeholders_are_not_invoiced(engine, session_factory):
    """A VOID row is not consumption; billing it would charge for usage that has
    no price, and would leave nothing to promote when the price appears."""
    await _seed(
        session_factory,
        _wallet(balance="10000"),
        _entry(
            1,
            status=LedgerStatus.VOID,
            quantity="0",
            amount="0",
            occurred_at=AUGUST + timedelta(hours=1),
        ),
        _entry(
            2, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=2)
        ),
    )

    report = await _invoicer().invoice_once(now=datetime(2026, 9, 5))

    assert report.entries_invoiced == 1
    entries = await _entries(engine)
    assert entries[1].status == LedgerStatus.VOID.value
    assert entries[1].invoice_id is None


@pytest.mark.asyncio
async def test_usage_with_no_payer_is_left_for_the_rater_to_explain(
    engine, session_factory
):
    """An invoice with nobody to bill would be a statement nobody can pay."""
    await _seed(
        session_factory,
        _wallet(balance="10000"),
        _entry(
            1,
            principal_id=None,
            quantity="12",
            amount="150.00",
            occurred_at=AUGUST + timedelta(hours=1),
        ),
    )

    report = await _invoicer().invoice_once(now=datetime(2026, 9, 5))

    assert report.invoices_issued == 0
    assert await _invoices(engine) == []


@pytest.mark.asyncio
async def test_usage_in_the_open_period_waits(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(balance="10000"),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=SEPTEMBER + timedelta(days=1)
        ),
    )

    report = await _invoicer().invoice_once(now=NOW)

    assert report.invoices_issued == 0
    assert report.periods_scanned == 3
    assert await _invoices(engine) == []


@pytest.mark.asyncio
async def test_a_period_boundary_is_half_open(engine, session_factory):
    """The instant a period ends belongs to the next one, so a charge lands in
    exactly one statement whichever way the clock falls."""
    await _seed(
        session_factory,
        _wallet(balance="100000"),
        # Last instant of August, and the first of September.
        _entry(
            1,
            quantity="1",
            amount="10.00",
            occurred_at=SEPTEMBER - timedelta(seconds=1),
        ),
        _entry(2, quantity="1", amount="20.00", occurred_at=SEPTEMBER),
    )

    await _invoicer().invoice_once(now=datetime(2026, 9, 5))

    invoices = await _invoices(engine)
    assert len(invoices) == 1
    assert Decimal(invoices[0].amount) == Decimal("10.00")
    entries = await _entries(engine)
    assert entries[1].invoice_id == invoices[0].id
    assert entries[2].invoice_id is None


@pytest.mark.asyncio
async def test_a_second_pass_issues_nothing_and_debits_nothing(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(balance="1000"),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=1)
        ),
    )
    invoicer = _invoicer()

    first = await invoicer.invoice_once(now=datetime(2026, 9, 5))
    second = await invoicer.invoice_once(now=datetime(2026, 9, 5))

    assert first.invoices_issued == 1
    assert second.invoices_issued == 0
    assert second.entries_invoiced == 0
    assert len(await _invoices(engine)) == 1
    assert (await _wallets(engine))[ORG_A].balance == Decimal("850.00")


@pytest.mark.asyncio
async def test_each_org_gets_its_own_statement(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(ORG_A, balance="1000"),
        _wallet(ORG_B, balance="1000"),
        _entry(
            1,
            principal_id=ORG_A,
            quantity="12",
            amount="150.00",
            occurred_at=AUGUST + timedelta(hours=1),
        ),
        _entry(
            2,
            principal_id=ORG_B,
            quantity="4",
            amount="50.00",
            occurred_at=AUGUST + timedelta(hours=2),
        ),
    )

    report = await _invoicer().invoice_once(now=datetime(2026, 9, 5))

    assert report.invoices_issued == 2
    invoices = {i.principal_id: i for i in await _invoices(engine)}
    assert Decimal(invoices[ORG_A].amount) == Decimal("150.00")
    assert Decimal(invoices[ORG_B].amount) == Decimal("50.00")
    wallets = await _wallets(engine)
    assert wallets[ORG_A].balance == Decimal("850.00")
    assert wallets[ORG_B].balance == Decimal("950.00")


@pytest.mark.asyncio
async def test_a_zero_total_statement_is_settled_without_a_debit(
    engine, session_factory
):
    """Free-tier pricing is a real configuration; taking zero money must not
    leave a statement outstanding forever."""
    await _seed(
        session_factory,
        _wallet(balance="10"),
        _entry(1, quantity="12", amount="0", occurred_at=AUGUST + timedelta(hours=1)),
    )

    report = await _invoicer().invoice_once(now=datetime(2026, 9, 5))

    invoices = await _invoices(engine)
    assert len(invoices) == 1
    assert invoices[0].status == InvoiceStatus.SETTLED.value
    assert report.unpaid_invoices == 0
    assert (await _wallets(engine))[ORG_A].balance == Decimal("10")
    assert (await _entries(engine))[1].status == LedgerStatus.SETTLED.value


# ---------------------------------------------------------------------------
# Late usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_late_usage_for_an_unstated_period_gets_its_own_statement(
    engine, session_factory
):
    """A missing price appearing promotes a VOID row into a real charge for a
    period that may not have been stated yet. When it has not, the usage gets a
    statement for the period it actually belongs to — better attribution than
    folding it into a later one, and no reason to bend any rule to do it."""
    await _seed(
        session_factory,
        _wallet(balance="10000"),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=1)
        ),
    )
    invoicer = _invoicer()
    # Only August has usage, so only August is stated; July has no invoice.
    await invoicer.invoice_once(now=datetime(2026, 9, 5))

    await _seed(
        session_factory,
        _entry(2, quantity="4", amount="50.00", occurred_at=datetime(2026, 7, 15)),
    )

    report = await invoicer.invoice_once(now=datetime(2026, 10, 5))

    assert report.carried_entries == 0
    invoices = {i.period_start.month: i for i in await _invoices(engine)}
    assert sorted(invoices) == [7, 8]
    assert Decimal(invoices[7].amount) == Decimal("50.00")
    assert Decimal(invoices[8].amount) == Decimal("150.00")
    assert (await _entries(engine))[2].invoice_id == invoices[7].id


@pytest.mark.asyncio
async def test_late_usage_for_a_stated_period_rides_the_next_statement(
    engine, session_factory
):
    """One statement per period is not bent for late usage.

    Back-editing an issued statement would make the document a tenant was shown
    disagree with the money taken from them; dropping the usage would lose
    revenue. So it waits, then rides the next statement and is reported as a
    carry — the number an operator needs when a statement looks too large.
    """
    await _seed(
        session_factory,
        _wallet(balance="10000"),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=1)
        ),
    )
    invoicer = _invoicer()
    await invoicer.invoice_once(now=datetime(2026, 9, 5))
    august = (await _invoices(engine))[0]

    await _seed(
        session_factory,
        _entry(
            2, quantity="4", amount="50.00", occurred_at=AUGUST + timedelta(hours=5)
        ),
    )

    # August is still the newest closed period and is already stated: the entry
    # has nowhere to go yet.
    waiting = await invoicer.invoice_once(now=datetime(2026, 9, 6))
    assert waiting.invoices_issued == 0
    assert waiting.waiting_entries == 1

    # September closes, and the late August usage rides its statement.
    carried = await invoicer.invoice_once(now=datetime(2026, 10, 5))

    assert carried.carried_entries == 1
    assert carried.carried_amount == Decimal("50.00")
    invoices = {i.period_start.month: i for i in await _invoices(engine)}
    assert sorted(invoices) == [8, 9]
    assert Decimal(invoices[9].amount) == Decimal("50.00")
    assert (await _entries(engine))[2].invoice_id == invoices[9].id
    # The issued statement was not rewritten.
    reloaded = {i.id: i for i in await _invoices(engine)}[august.id]
    assert Decimal(reloaded.amount) == Decimal("150.00")


@pytest.mark.asyncio
async def test_late_usage_waits_when_every_closed_period_is_already_stated(
    engine, session_factory
):
    await _seed(
        session_factory,
        _wallet(balance="10000"),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=1)
        ),
    )
    invoicer = _invoicer()
    await invoicer.invoice_once(now=datetime(2026, 9, 5))

    # August is invoiced and is still the newest closed period.
    await _seed(
        session_factory,
        _entry(
            2, quantity="4", amount="50.00", occurred_at=AUGUST + timedelta(hours=5)
        ),
    )
    report = await invoicer.invoice_once(now=datetime(2026, 9, 6))

    assert report.invoices_issued == 0
    assert report.waiting_entries == 1
    entries = await _entries(engine)
    assert entries[2].invoice_id is None
    assert entries[2].status == LedgerStatus.PENDING.value


# ---------------------------------------------------------------------------
# Collection failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_uncollectable_statement_suspends_the_org(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(balance="100"),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=1)
        ),
    )

    report = await _invoicer().invoice_once(now=datetime(2026, 9, 5))

    invoices = await _invoices(engine)
    assert len(invoices) == 1
    assert invoices[0].status == InvoiceStatus.ISSUED.value
    assert invoices[0].unpaid_reason is not None
    assert report.unpaid_invoices == 1
    assert report.amount_unpaid == Decimal("150.00")
    assert report.suspended == [ORG_A]

    wallets = await _wallets(engine)
    assert wallets[ORG_A].balance == Decimal("100")  # nothing taken
    assert wallets[ORG_A].suspended is True

    # The entries are claimed by the statement but still uncollected, so no
    # later pass can invoice them a second time.
    entries = await _entries(engine)
    assert entries[1].status == LedgerStatus.PENDING.value
    assert entries[1].invoice_id == invoices[0].id


@pytest.mark.asyncio
async def test_a_top_up_clears_an_outstanding_statement_on_the_next_pass(
    engine, session_factory
):
    await _seed(
        session_factory,
        _wallet(balance="100", suspended=True),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=1)
        ),
    )
    invoicer = _invoicer()
    await invoicer.invoice_once(now=datetime(2026, 9, 5))

    async with session_factory() as s:
        wallet = (await s.exec(select(Wallet))).first()
        wallet.balance = Decimal("500")
        s.add(wallet)
        await s.commit()

    report = await invoicer.invoice_once(now=datetime(2026, 9, 6))

    assert report.invoices_collected == 1
    assert report.amount_collected == Decimal("150.00")
    assert report.resumed == [ORG_A]
    invoices = await _invoices(engine)
    assert invoices[0].status == InvoiceStatus.SETTLED.value
    assert invoices[0].unpaid_reason is None
    wallets = await _wallets(engine)
    assert wallets[ORG_A].balance == Decimal("350.00")
    assert wallets[ORG_A].suspended is False
    assert (await _entries(engine))[1].status == LedgerStatus.SETTLED.value


@pytest.mark.asyncio
async def test_an_unpaid_statement_is_retried_and_still_uncollectable(
    engine, session_factory
):
    await _seed(
        session_factory,
        _wallet(balance="10"),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=1)
        ),
    )
    invoicer = _invoicer()
    await invoicer.invoice_once(now=datetime(2026, 9, 5))

    second = await invoicer.invoice_once(now=datetime(2026, 9, 6))

    assert second.invoices_issued == 0
    assert second.unpaid_invoices == 1
    assert len(await _invoices(engine)) == 1
    assert (await _wallets(engine))[ORG_A].balance == Decimal("10")


@pytest.mark.asyncio
async def test_one_org_failing_does_not_block_another(engine, session_factory):
    """Per-statement commits: a lock timeout in one org's collection must not
    roll back another org's payment."""
    await _seed(
        session_factory,
        _wallet(ORG_A, balance="10"),
        _wallet(ORG_B, balance="1000"),
        _entry(
            1,
            principal_id=ORG_A,
            quantity="12",
            amount="150.00",
            occurred_at=AUGUST + timedelta(hours=1),
        ),
        _entry(
            2,
            principal_id=ORG_B,
            quantity="4",
            amount="50.00",
            occurred_at=AUGUST + timedelta(hours=2),
        ),
    )

    report = await _invoicer().invoice_once(now=datetime(2026, 9, 5))

    assert report.unpaid_invoices == 1
    assert report.invoices_collected == 1
    assert report.suspended == [ORG_A]
    wallets = await _wallets(engine)
    assert wallets[ORG_A].balance == Decimal("10")
    assert wallets[ORG_B].balance == Decimal("950.00")
    assert wallets[ORG_B].suspended is False


@pytest.mark.asyncio
async def test_a_suspension_for_an_unpaid_statement_reaches_the_keys(
    engine, session_factory
):
    """WP5's propagation, from the invoicing side: an org that cannot pay its
    statement must not keep serving traffic through the gateway."""
    await _seed(
        session_factory,
        _wallet(balance="10"),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=1)
        ),
        ApiKey(
            id=1,
            name="key1",
            access_key="ak-1",
            hashed_secret_key="argon2-1",
            scope=[],
            user_id=ORG_A,
            owner_principal_id=ORG_A,
            is_custom=False,
            created_at=NOW,
            updated_at=NOW,
        ),
    )

    await _invoicer().invoice_once(now=datetime(2026, 9, 5))

    async with AsyncSession(engine) as s:
        key = (await s.exec(select(ApiKey))).first()
    assert key.suspended is True
    assert key.suspension_reason.startswith("billing:")


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_mode_writes_no_statement_and_takes_no_money(
    engine, session_factory
):
    await _seed(
        session_factory,
        _wallet(balance="1000"),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=1)
        ),
    )

    report = await _invoicer(mode=BillingMode.SHADOW).invoice_once(
        now=datetime(2026, 9, 5)
    )

    assert report.invoices_issued == 0
    assert report.periods_scanned == 0
    assert await _invoices(engine) == []
    assert (await _wallets(engine))[ORG_A].balance == Decimal("1000")
    assert (await _entries(engine))[1].status == LedgerStatus.PENDING.value


@pytest.mark.asyncio
async def test_off_mode_is_also_a_no_op(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(balance="1000"),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=1)
        ),
    )

    report = await _invoicer(mode=BillingMode.OFF).invoice_once(
        now=datetime(2026, 9, 5)
    )

    assert report.invoices_issued == 0
    assert await _invoices(engine) == []


@pytest.mark.asyncio
async def test_a_daily_period_closes_each_day(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(balance="10000"),
        _entry(1, quantity="1", amount="10.00", occurred_at=datetime(2026, 9, 21, 6)),
        _entry(2, quantity="1", amount="20.00", occurred_at=datetime(2026, 9, 22, 6)),
    )

    report = await _invoicer(period=BillingPeriod.DAILY, cron="0 1 * * *").invoice_once(
        now=datetime(2026, 9, 23, 1, 30),
    )

    assert report.invoices_issued == 2
    invoices = sorted(await _invoices(engine), key=lambda i: i.period_start)
    assert [Decimal(i.amount) for i in invoices] == [Decimal("10.00"), Decimal("20.00")]
    assert invoices[0].period_start.replace(tzinfo=None) == datetime(2026, 9, 21)
    assert invoices[1].period_start.replace(tzinfo=None) == datetime(2026, 9, 22)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_summary_names_what_a_pass_did(engine, session_factory):
    await _seed(
        session_factory,
        _wallet(balance="1000"),
        _entry(
            1, quantity="12", amount="150.00", occurred_at=AUGUST + timedelta(hours=1)
        ),
    )

    report = await _invoicer().invoice_once(now=datetime(2026, 9, 5))
    text = report.summary()

    assert "1 issued" in text
    assert "monthly" in text
    assert str(report.amount_invoiced) in text
    assert report.duration_ms >= 0


# ---------------------------------------------------------------------------
# The rollout whitelist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_org_outside_the_whitelist_gets_no_statement(
    engine, session_factory, monkeypatch
):
    """Its deferred charges stay rated, PENDING and uninvoiced — waiting, not
    lost — so bringing it into scope later is a switch rather than a backfill."""
    monkeypatch.setattr(envs, "BILLING_ENFORCE_PRINCIPALS", str(ORG_A))
    await _seed(
        session_factory,
        _wallet(ORG_A, balance="10000"),
        _wallet(ORG_B, balance="10000"),
        _entry(
            1,
            principal_id=ORG_A,
            quantity="12",
            amount="150.00",
            occurred_at=AUGUST + timedelta(hours=1),
        ),
        _entry(
            2,
            principal_id=ORG_B,
            quantity="4",
            amount="50.00",
            occurred_at=AUGUST + timedelta(hours=2),
        ),
    )

    report = await _invoicer().invoice_once(now=datetime(2026, 9, 5))

    assert report.out_of_scope_principals == 1
    assert report.invoices_issued == 1
    invoices = {i.principal_id: i for i in await _invoices(engine)}
    assert list(invoices) == [ORG_A]
    entries = await _entries(engine)
    assert entries[2].invoice_id is None
    assert entries[2].status == LedgerStatus.PENDING.value
    # And no money moved for the org that is not being charged yet.
    wallets = await _wallets(engine)
    assert wallets[ORG_B].balance == Decimal("10000")


@pytest.mark.asyncio
async def test_bringing_an_org_into_scope_bills_what_waited(
    engine, session_factory, monkeypatch
):
    monkeypatch.setattr(envs, "BILLING_ENFORCE_PRINCIPALS", str(ORG_A))
    await _seed(
        session_factory,
        _wallet(ORG_B, balance="10000"),
        _entry(
            2,
            principal_id=ORG_B,
            quantity="4",
            amount="50.00",
            occurred_at=AUGUST + timedelta(hours=2),
        ),
    )
    invoicer = _invoicer()
    assert (await invoicer.invoice_once(now=datetime(2026, 9, 5))).invoices_issued == 0

    monkeypatch.setattr(envs, "BILLING_ENFORCE_PRINCIPALS", "")
    # A month later. The waiting usage gets a statement for the period it
    # actually belongs to — August was never invoiced, so there is nothing to
    # carry it onto and no reason to misattribute it to September.
    report = await invoicer.invoice_once(now=datetime(2026, 10, 5))

    assert report.invoices_issued == 1
    assert report.carried_entries == 0
    invoice = (await _invoices(engine))[0]
    assert invoice.principal_id == ORG_B
    assert invoice.period_start.replace(tzinfo=None) == AUGUST
    assert Decimal(invoice.amount) == Decimal("50.00")
    assert (await _wallets(engine))[ORG_B].balance == Decimal("9950")
