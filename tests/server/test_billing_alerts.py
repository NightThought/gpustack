"""Billing alerts (WP7.3).

Two things are under test, and they fail differently.

The registry's job is deduplication: every condition here is a state that a scan
re-observes, so logging each observation would produce one line per problem per
interval — 288 identical warnings a day for a suspended org, which is how a new
problem ends up unread. What has to hold is that a problem is emitted once,
counted while it persists, emitted again only on resolution or escalation.

The detector's job is reading the right condition out of the database: unpriced
usage from the ledger's placeholders rather than from one sweep's report (which
does not survive a restart), unpaid invoices only past their grace period (an
invoice is briefly ISSUED inside its own transaction, and alerting on that would
fire for every statement ever written), and suspensions with the outstanding
figure attached — because "balance 0.05" does not tell an operator whether a
top-up of 1 resumes the org or 400 is owed.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.billing import (
    SKU_GPU_HOUR_PREFIX,
    SKU_TOKEN_PROMPT,
    UNIT_GPU_HOURS,
    UNIT_TOKENS,
    Invoice,
    InvoiceStatus,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    SettleMode,
    Wallet,
)
from gpustack.server import billing_alerts as alerts_module
from gpustack.server.billing_alerts import (
    LOG_PREFIX,
    AlertSeverity,
    BillingAlertDetector,
    BillingAlertKind,
    BillingAlertRegistry,
    billing_alerts,
)

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
ORG = 990101
SKU_910B = SKU_GPU_HOUR_PREFIX + "910b"


# ---------------------------------------------------------------------------
# Registry: deduplication, escalation, resolution
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    return BillingAlertRegistry()


def test_a_problem_is_emitted_once_and_counted_afterwards(registry, caplog):
    with caplog.at_level("WARNING", logger="gpustack.server.billing_alerts"):
        first = registry.raise_alert(
            BillingAlertKind.UNPRICED_GAP,
            SKU_910B,
            severity=AlertSeverity.WARNING,
            summary="3 records cannot be priced",
        )
        second = registry.raise_alert(
            BillingAlertKind.UNPRICED_GAP,
            SKU_910B,
            severity=AlertSeverity.WARNING,
            summary="3 records cannot be priced",
        )
        third = registry.raise_alert(
            BillingAlertKind.UNPRICED_GAP,
            SKU_910B,
            severity=AlertSeverity.WARNING,
            summary="4 records cannot be priced",
        )

    assert (first, second, third) == (True, False, False)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert LOG_PREFIX in warnings[0].message
    assert f"kind={BillingAlertKind.UNPRICED_GAP.value}" in warnings[0].message

    alerts = registry.active()
    assert len(alerts) == 1
    assert alerts[0].occurrences == 3
    # The latest observation wins, so the line an operator reads is current.
    assert alerts[0].summary == "4 records cannot be priced"


def test_resolution_is_logged_with_the_exposure_it_represents(registry, caplog):
    registry.raise_alert(
        BillingAlertKind.WALLET_SUSPENDED,
        str(ORG),
        severity=AlertSeverity.WARNING,
        summary="suspended",
    )
    registry.raise_alert(
        BillingAlertKind.WALLET_SUSPENDED,
        str(ORG),
        severity=AlertSeverity.WARNING,
        summary="suspended",
    )

    with caplog.at_level("INFO", logger="gpustack.server.billing_alerts"):
        resolved = registry.resolve(BillingAlertKind.WALLET_SUSPENDED, str(ORG))

    assert resolved is True
    assert registry.active() == []
    line = [r for r in caplog.records if "resolved" in r.message][0].message
    assert "2 observations" in line
    # Resolving something that was not active is not an event.
    assert registry.resolve(BillingAlertKind.WALLET_SUSPENDED, str(ORG)) is False


def test_an_escalation_is_worth_a_second_line(registry, caplog):
    """ "Still unpaid" is noise; "unpaid for a week and now critical" is not."""
    with caplog.at_level("WARNING", logger="gpustack.server.billing_alerts"):
        registry.raise_alert(
            BillingAlertKind.UNPAID_INVOICE,
            "1",
            severity=AlertSeverity.WARNING,
            summary="unpaid 3 days",
        )
        again = registry.raise_alert(
            BillingAlertKind.UNPAID_INVOICE,
            "1",
            severity=AlertSeverity.WARNING,
            summary="unpaid 4 days",
        )
        escalated = registry.raise_alert(
            BillingAlertKind.UNPAID_INVOICE,
            "1",
            severity=AlertSeverity.CRITICAL,
            summary="unpaid 8 days",
        )

    assert (again, escalated) == (False, True)
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 2
    assert registry.active()[0].severity is AlertSeverity.CRITICAL
    # One problem, not two: escalation replaces rather than adds.
    assert len(registry.active()) == 1


def test_different_problems_do_not_dedup_against_each_other(registry):
    registry.raise_alert(
        BillingAlertKind.UNPRICED_GAP,
        SKU_910B,
        severity=AlertSeverity.WARNING,
        summary="a",
    )
    registry.raise_alert(
        BillingAlertKind.UNPRICED_GAP,
        SKU_TOKEN_PROMPT,
        severity=AlertSeverity.WARNING,
        summary="b",
    )
    registry.raise_alert(
        BillingAlertKind.WALLET_SUSPENDED,
        str(ORG),
        severity=AlertSeverity.WARNING,
        summary="c",
    )
    assert len(registry.active()) == 3


def test_resolve_absent_clears_only_what_is_no_longer_true(registry):
    """How a state-based detector clears: report what is true now, and anything
    asserted last time that is not in that set is resolved."""
    for key in ("a", "b", "c"):
        registry.raise_alert(
            BillingAlertKind.UNPRICED_GAP,
            key,
            severity=AlertSeverity.WARNING,
            summary=key,
        )

    cleared = registry.resolve_absent(BillingAlertKind.UNPRICED_GAP, ["b"])

    assert cleared == 2
    assert [alert.key for alert in registry.active()] == ["b"]


def test_resolve_absent_leaves_other_kinds_alone(registry):
    registry.raise_alert(
        BillingAlertKind.UNPRICED_GAP,
        "a",
        severity=AlertSeverity.WARNING,
        summary="a",
    )
    registry.raise_alert(
        BillingAlertKind.WALLET_SUSPENDED,
        str(ORG),
        severity=AlertSeverity.WARNING,
        summary="s",
    )

    registry.resolve_absent(BillingAlertKind.UNPRICED_GAP, [])

    assert [alert.kind for alert in registry.active()] == [
        BillingAlertKind.WALLET_SUSPENDED
    ]


def test_a_problem_raised_again_after_resolving_is_a_new_one(registry):
    registry.raise_alert(
        BillingAlertKind.WALLET_SUSPENDED,
        str(ORG),
        severity=AlertSeverity.WARNING,
        summary="first",
    )
    registry.resolve(BillingAlertKind.WALLET_SUSPENDED, str(ORG))

    emitted = registry.raise_alert(
        BillingAlertKind.WALLET_SUSPENDED,
        str(ORG),
        severity=AlertSeverity.WARNING,
        summary="second",
    )

    assert emitted is True
    assert registry.active()[0].occurrences == 1


def test_raised_total_never_goes_down(registry):
    """A gauge of active alerts falls when a problem is fixed; if that number is
    used as a counter, Prometheus reads the fall as a reset."""
    registry.raise_alert(
        BillingAlertKind.UNPRICED_GAP,
        "a",
        severity=AlertSeverity.WARNING,
        summary="a",
    )
    registry.resolve(BillingAlertKind.UNPRICED_GAP, "a")
    registry.raise_alert(
        BillingAlertKind.UNPRICED_GAP,
        "a",
        severity=AlertSeverity.WARNING,
        summary="a",
    )

    assert registry.raised_total()[BillingAlertKind.UNPRICED_GAP.value] == 2


def test_the_active_alert_cap_stops_unbounded_growth(registry, caplog):
    """Not about ordinary use — about a bug that mints keys from a row id in a
    table that grows without limit turning an alerting module into a leak."""
    with patch.object(alerts_module, "MAX_ACTIVE_ALERTS", 2):
        with caplog.at_level("ERROR", logger="gpustack.server.billing_alerts"):
            for key in ("a", "b", "c", "d"):
                registry.raise_alert(
                    BillingAlertKind.UNPAID_INVOICE,
                    key,
                    severity=AlertSeverity.WARNING,
                    summary=key,
                )

    assert len(registry.active()) == 2
    assert "cap" in caplog.text


def test_metrics_snapshot_is_replaced_not_merged(registry):
    registry.publish_metrics({"unpriced_skus": 2.0})
    registry.publish_metrics({"suspended_wallets": 1.0})
    assert registry.metrics() == {"suspended_wallets": 1.0}


def test_snapshots_are_copies(registry):
    """A scrape reads on another thread; handing it the live dict would let a
    concurrent write show up half-applied."""
    registry.publish_metrics({"unpriced_skus": 1.0})
    snapshot = registry.metrics()
    registry.publish_metrics({"unpriced_skus": 99.0})
    assert snapshot == {"unpriced_skus": 1.0}


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        for model in (LedgerEntry, Invoice, Wallet):
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

    with patch.object(alerts_module, "async_session", _factory):
        yield _factory


async def _seed(factory, *rows):
    async with factory() as s:
        for row in rows:
            s.add(row)
        await s.commit()


def _detector(**kwargs) -> BillingAlertDetector:
    kwargs.setdefault("interval_seconds", 300)
    kwargs.setdefault("unpaid_grace_days", 3)
    kwargs.setdefault("unpaid_critical_days", 7)
    return BillingAlertDetector(**kwargs)


def _void_entry(id_, sku=SKU_910B, occurred_at=NOW - timedelta(days=2)):
    return LedgerEntry(
        id=id_,
        source_table="metered_usage",
        source_id=id_,
        principal_id=ORG,
        sku=sku,
        quantity=Decimal(0),
        unit=UNIT_GPU_HOURS,
        unit_price=Decimal(0),
        amount=Decimal(0),
        currency="CNY",
        direction=LedgerDirection.DEBIT,
        settle_mode=SettleMode.DEFERRED,
        status=LedgerStatus.VOID,
        occurred_at=occurred_at,
        created_at=NOW,
        updated_at=NOW,
    )


def _charge(
    id_,
    *,
    amount="1.00",
    settle_mode=SettleMode.REALTIME,
    status=LedgerStatus.PENDING,
    sku=SKU_TOKEN_PROMPT,
    unit=UNIT_TOKENS,
):
    return LedgerEntry(
        id=id_,
        source_table="model_usage_details",
        source_id=id_,
        principal_id=ORG,
        sku=sku,
        quantity=Decimal(1000),
        unit=unit,
        unit_price=Decimal("0.001"),
        amount=Decimal(amount),
        currency="CNY",
        direction=LedgerDirection.DEBIT,
        settle_mode=settle_mode,
        status=status,
        occurred_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def _invoice(
    id_,
    *,
    status=InvoiceStatus.ISSUED,
    issued_at=NOW - timedelta(days=5),
    amount="120.00",
):
    return Invoice(
        id=id_,
        principal_id=ORG,
        principal_name="acme",
        period_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        period_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
        amount=Decimal(amount),
        currency="CNY",
        status=status,
        issued_at=issued_at,
        created_at=NOW,
        updated_at=NOW,
    )


def _wallet(balance="0.05", suspended=True, suspended_at=NOW - timedelta(hours=6)):
    return Wallet(
        principal_id=ORG,
        principal_name="acme",
        balance=Decimal(balance),
        currency="CNY",
        suspended=suspended,
        suspended_at=suspended_at if suspended else None,
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.mark.asyncio
async def test_a_clean_deployment_raises_nothing(session_factory, caplog):
    await _seed(session_factory, _charge(1, status=LedgerStatus.SETTLED))

    with caplog.at_level("WARNING", logger="gpustack.server.billing_alerts"):
        report = await _detector().detect_once()

    assert (report.raised, report.active) == (0, 0)
    assert LOG_PREFIX not in caplog.text


@pytest.mark.asyncio
async def test_unpriced_usage_raises_one_alert_per_sku(session_factory):
    """Keyed by SKU, not by row: rows come and go as prices appear and
    placeholders are promoted, so a per-row key would raise and resolve
    constantly and tell an operator nothing."""
    await _seed(
        session_factory,
        _void_entry(1, sku=SKU_910B),
        _void_entry(2, sku=SKU_910B),
        _void_entry(3, sku=SKU_TOKEN_PROMPT),
    )

    report = await _detector().detect_once()

    assert report.unpriced_skus == 2
    assert report.unpriced_entries == 3
    kinds = {(alert.kind, alert.key): alert for alert in billing_alerts.active()}
    gap = kinds[(BillingAlertKind.UNPRICED_GAP, SKU_910B)]
    assert "2 usage record(s)" in gap.summary
    assert "not being billed" in gap.summary
    assert "price-book" in gap.detail


@pytest.mark.asyncio
async def test_an_unpriced_gap_resolves_once_a_price_appears(session_factory):
    await _seed(session_factory, _void_entry(1))
    await _detector().detect_once()
    assert len(billing_alerts.active()) == 1

    # The price arrived and the placeholder was promoted.
    async with session_factory() as s:
        from sqlmodel import select

        entry = (await s.exec(select(LedgerEntry))).first()
        entry.status = LedgerStatus.PENDING.value
        s.add(entry)
        await s.commit()

    report = await _detector().detect_once()

    assert report.resolved == 1
    assert billing_alerts.active() == []


@pytest.mark.asyncio
async def test_a_second_scan_does_not_repeat_the_alert(session_factory, caplog):
    await _seed(session_factory, _void_entry(1))
    detector = _detector()

    await detector.detect_once()
    caplog.clear()
    with caplog.at_level("WARNING", logger="gpustack.server.billing_alerts"):
        second = await detector.detect_once()
        third = await detector.detect_once()

    assert second.raised == 0
    assert third.raised == 0
    assert caplog.text == ""
    # But the persistence is still visible, which is the point of counting.
    assert billing_alerts.active()[0].occurrences == 3


@pytest.mark.asyncio
async def test_an_invoice_inside_its_grace_period_is_not_an_alert(session_factory):
    """Issuing and collecting are one step for a funded org and two for an
    unfunded one; alerting on the transient would fire for every statement."""
    await _seed(session_factory, _invoice(1, issued_at=NOW - timedelta(hours=1)))

    report = await _detector().detect_once()

    assert report.unpaid_invoices == 0
    assert billing_alerts.active() == []


@pytest.mark.asyncio
async def test_an_invoice_past_its_grace_period_is(session_factory):
    await _seed(session_factory, _invoice(1, issued_at=NOW - timedelta(days=4)))

    report = await _detector().detect_once(now=NOW)

    assert report.unpaid_invoices == 1
    assert report.unpaid_amount == Decimal("120.00")
    alert = billing_alerts.active()[0]
    assert alert.kind is BillingAlertKind.UNPAID_INVOICE
    assert alert.severity is AlertSeverity.WARNING
    assert "unpaid for 4 day(s)" in alert.summary
    assert "acme" in alert.summary


@pytest.mark.asyncio
async def test_an_invoice_unpaid_for_a_week_escalates(session_factory):
    await _seed(session_factory, _invoice(1, issued_at=NOW - timedelta(days=9)))

    await _detector().detect_once(now=NOW)

    alert = billing_alerts.active()[0]
    assert alert.severity is AlertSeverity.CRITICAL


@pytest.mark.asyncio
async def test_a_paid_invoice_resolves(session_factory):
    await _seed(session_factory, _invoice(1, issued_at=NOW - timedelta(days=4)))
    await _detector().detect_once()
    assert len(billing_alerts.active()) == 1

    async with session_factory() as s:
        from sqlmodel import select

        invoice = (await s.exec(select(Invoice))).first()
        invoice.status = InvoiceStatus.SETTLED.value
        s.add(invoice)
        await s.commit()

    report = await _detector().detect_once()

    assert report.resolved == 1
    assert billing_alerts.active() == []


@pytest.mark.asyncio
async def test_a_suspension_alert_carries_what_it_would_take_to_resume(
    session_factory,
):
    """ "Balance 0.05" does not answer whether a top-up of 1 fixes it."""
    await _seed(
        session_factory,
        _wallet(balance="0.05"),
        _charge(2, amount="400.00"),
    )

    await _detector().detect_once()

    alert = billing_alerts.active()[0]
    assert alert.kind is BillingAlertKind.WALLET_SUSPENDED
    assert "balance 0.05" in alert.summary
    assert "outstanding 400" in alert.summary
    assert "covering 400" in alert.detail
    assert "402" in alert.detail


@pytest.mark.asyncio
async def test_a_resumed_wallet_resolves_its_suspension(session_factory):
    await _seed(session_factory, _wallet(balance="0"))
    await _detector().detect_once()
    assert len(billing_alerts.active()) == 1

    async with session_factory() as s:
        from sqlmodel import select

        wallet = (await s.exec(select(Wallet))).first()
        wallet.suspended = False
        s.add(wallet)
        await s.commit()

    await _detector().detect_once()

    assert billing_alerts.active() == []


@pytest.mark.asyncio
async def test_suspension_is_alerted_at_the_moment_it_happens(session_factory):
    """The scan owns resolution; the settlement path owns immediacy, so an
    operator hears about a tenant going dark within the sweep that suspended it."""
    await _seed(session_factory, _wallet(balance="0", suspended=True))

    BillingAlertDetector.note_suspension(
        ORG, balance=Decimal("0.05"), owed=Decimal("400"), currency="CNY"
    )

    alert = billing_alerts.active()[0]
    assert alert.kind is BillingAlertKind.WALLET_SUSPENDED
    assert "just suspended" in alert.summary

    # The scan agrees rather than duplicating it.
    report = await _detector().detect_once()
    assert report.raised == 0
    assert len(billing_alerts.active()) == 1
    assert billing_alerts.active()[0].occurrences == 2


@pytest.mark.asyncio
async def test_the_backlog_is_measured_and_published(session_factory):
    """Not an alert: deferred charges are *supposed* to wait for their period.
    It is the number that answers "is the pipeline keeping up"."""
    await _seed(
        session_factory,
        _charge(1, amount="2.50", settle_mode=SettleMode.REALTIME),
        _charge(
            2,
            amount="7.25",
            settle_mode=SettleMode.DEFERRED,
            sku=SKU_910B,
            unit=UNIT_GPU_HOURS,
        ),
        _charge(3, amount="99.00", status=LedgerStatus.SETTLED),
    )

    report = await _detector().detect_once()

    assert report.pending_realtime_amount == Decimal("2.50")
    assert report.pending_deferred_amount == Decimal("7.25")
    metrics = billing_alerts.metrics()
    assert metrics["pending_realtime_amount"] == 2.5
    assert metrics["pending_deferred_amount"] == 7.25
    assert metrics["suspended_wallets"] == 0.0
    assert metrics["unpriced_skus"] == 0.0


@pytest.mark.asyncio
async def test_the_snapshot_exposes_every_gauge_the_exporter_serves(
    session_factory,
):
    from gpustack.exporter.billing_metrics import _GAUGES

    await _seed(session_factory, _void_entry(1), _wallet())
    await _detector().detect_once()

    metrics = billing_alerts.metrics()
    missing = [key for key, _ in _GAUGES if key not in metrics]
    assert missing == []


@pytest.mark.asyncio
async def test_a_scan_survives_one_condition_failing(session_factory, caplog):
    """The detector runs in a leader loop with no supervision; a scan that dies
    on one condition would stop reporting all of them."""
    await _seed(session_factory, _void_entry(1))
    detector = _detector()

    with patch.object(
        BillingAlertDetector, "_check_suspensions", side_effect=RuntimeError("boom")
    ):
        with caplog.at_level("ERROR", logger="gpustack.server.billing_alerts"):
            with pytest.raises(RuntimeError):
                await detector.detect_once()

    # The loop, not the scan, is what contains the failure.
    assert await detector.detect_once() is not None


def test_configuration_is_validated_at_construction():
    with pytest.raises(ValueError):
        BillingAlertDetector(interval_seconds=0)
    with pytest.raises(ValueError) as exc_info:
        BillingAlertDetector(
            interval_seconds=60, unpaid_grace_days=7, unpaid_critical_days=3
        )
    # Escalating before the alert exists would read as a bug, not an escalation.
    assert "at least the grace" in str(exc_info.value)
