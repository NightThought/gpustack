"""Deferred billing: closing a period into an invoice and collecting it (WP6).

The hybrid decision, stated once because every branch below depends on it
=====================================================================
Token charges are settled in real time by ``billing_settlement``: a request is
rated, and the wallet is debited within one sweep of it. Resource charges —
``gpu.hour.*`` and ``storage.gb.hour``, the entries the rater marks
``settle_mode=deferred`` — are *not* debited as they accrue. They are collected
here, by an invoice, once the period they belong to has closed.

So an invoice is not a summary of money already taken. **Issuing one is the
moment the deferred money moves**: the entries it covers are debited from the
wallet in the same transaction that writes them, and an org that cannot cover
the statement is suspended exactly as one that cannot cover a realtime charge
is. The alternative — invoice as a read-only statement, wallet debited by the
settler — was rejected because it would leave resource usage unpaid until
somebody looked at a bill, which is the one property a prepaid platform cannot
have.

Consequences worth knowing before changing anything here:

* A token (realtime) entry never appears on an invoice as a charge. The two
  pipelines are disjoint by ``settle_mode``, and the settler skips deferred
  entries just as strictly as this module skips realtime ones.
* An unpaid invoice is arrears. ``billing_settlement.outstanding_charges``
  counts issued-but-unsettled invoices alongside pending realtime charges, so a
  top-up that does not cover the statement does not resume the org.

Periods, and why only closed ones are invoiced
==============================================
A period is a half-open UTC interval ``[period_start, period_end)``, daily or
monthly per ``GPUSTACK_BILLING_INVOICE_PERIOD``. Only periods that have already
closed are invoiced. Issuing the in-flight one would bill partial usage and —
because one invoice per (principal, period) is enforced — never bill the rest.

Idempotency is two layers deep, and both are load bearing
=========================================================
The unique constraint on ``(principal_id, period_start)`` stops a second
statement for the same period. The entry filter (``invoice_id IS NULL`` and
still ``PENDING``) stops the same usage appearing on two statements even if the
first layer were bypassed, and makes a pass that finds nothing to do the normal
case rather than a lucky one.

Late usage — rated after its period was invoiced, which happens whenever a
missing price appears and promotes a VOID placeholder — is carried into the next
invoice rather than lost or back-edited onto a statement already issued.
Mutating an issued invoice would make the document a tenant was shown disagree
with the money taken from them.

Scale
=====
Aggregation and the entry claim are both done in SQL: a month of hourly resource
buckets for a large cluster is tens of thousands of rows, and an invoicing pass
that hydrated them into entities would spend its time in Python. The same
predicate builds the aggregate and the UPDATE, so the statement's total and the
entries it claims cannot disagree.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import func, update
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.schemas.billing import (
    Invoice,
    InvoiceItem,
    InvoiceStatus,
    LedgerEntry,
    LedgerStatus,
    SettleMode,
)
from gpustack.server.billing_pricing import as_utc
from gpustack.server.billing_rater import BillingMode, billing_mode
from gpustack.server.billing_settlement import (
    debit_wallet,
    resume_wallet_if_funded,
    suspend_wallet,
)
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)

REASON_UNPAID = "billing:wallet balance insufficient when the invoice was issued"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------


class BillingPeriod(str, Enum):
    DAILY = "daily"
    MONTHLY = "monthly"


def billing_period(raw: Optional[str] = None) -> BillingPeriod:
    """Read ``GPUSTACK_BILLING_INVOICE_PERIOD``, refusing a value we cannot honour.

    An unrecognized period silently falling back to monthly would change what
    tenants are billed and when, so a typo is a startup failure instead.
    """
    value = (raw if raw is not None else envs.BILLING_INVOICE_PERIOD).strip().lower()
    try:
        return BillingPeriod(value)
    except ValueError:
        raise ValueError(
            f"GPUSTACK_BILLING_INVOICE_PERIOD must be one of "
            f"{[p.value for p in BillingPeriod]}, got {value!r}"
        ) from None


def period_bounds(
    period: BillingPeriod, at: Optional[datetime] = None
) -> Tuple[datetime, datetime]:
    """``[start, end)`` of the period containing ``at``, in UTC."""
    moment = as_utc(at) or _utcnow()
    if period is BillingPeriod.DAILY:
        start = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1)
    start = moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return start, _add_months(start, 1)


def shift_period(period: BillingPeriod, start: datetime, count: int) -> datetime:
    """``count`` periods after ``start`` (negative for earlier)."""
    if period is BillingPeriod.DAILY:
        return start + timedelta(days=count)
    return _add_months(start, count)


def _add_months(moment: datetime, months: int) -> datetime:
    """Calendar months forward. ``moment`` is always a first-of-month here, so
    there is no day to clamp — but the arithmetic stays correct if it is not."""
    total = (moment.year * 12 + (moment.month - 1)) + months
    return moment.replace(year=total // 12, month=total % 12 + 1)


def closed_periods(
    period: BillingPeriod,
    *,
    now: Optional[datetime] = None,
    lookback: int = 3,
) -> List[Tuple[datetime, datetime]]:
    """The ``lookback`` most recent closed periods, oldest first.

    Oldest first matters: the first period invoiced in a pass is the one that
    picks up anything carried from before the window, so a pass that runs after
    downtime collects old usage instead of leaving it behind.
    """
    if lookback <= 0:
        raise ValueError("invoice lookback must be at least one period")
    current_start, _ = period_bounds(period, now)
    return [
        (
            shift_period(period, current_start, -offset),
            shift_period(period, current_start, -offset + 1),
        )
        for offset in range(lookback, 0, -1)
    ]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class InvoicingReport:
    """What one pass did — returned for tests and logged for operators."""

    mode: BillingMode = BillingMode.SHADOW
    period: BillingPeriod = BillingPeriod.MONTHLY
    periods_scanned: int = 0
    invoices_issued: int = 0
    invoices_collected: int = 0  # outstanding statements paid this pass
    items: int = 0
    entries_invoiced: int = 0
    amount_invoiced: Decimal = field(default_factory=lambda: Decimal(0))
    amount_collected: Decimal = field(default_factory=lambda: Decimal(0))
    unpaid_invoices: int = 0
    amount_unpaid: Decimal = field(default_factory=lambda: Decimal(0))
    # Entries from before the period they were invoiced with (late rating).
    carried_entries: int = 0
    carried_amount: Decimal = field(default_factory=lambda: Decimal(0))
    # Uninvoiced entries left for a later period, because every closed period
    # that could hold them already has its one invoice.
    waiting_entries: int = 0
    suspended: List[int] = field(default_factory=list)
    resumed: List[int] = field(default_factory=list)
    duration_ms: int = 0

    def summary(self) -> str:
        return (
            f"billing invoicing pass ({self.mode.value}, {self.period.value}) in "
            f"{self.duration_ms}ms: {self.invoices_issued} issued over "
            f"{self.periods_scanned} closed period(s), {self.entries_invoiced} "
            f"entries / {self.amount_invoiced} total, {self.invoices_collected} "
            f"collected / {self.amount_collected}, {self.unpaid_invoices} unpaid / "
            f"{self.amount_unpaid}, suspended={self.suspended} "
            f"resumed={self.resumed}"
            + (
                f", {self.carried_entries} carried from earlier periods"
                if self.carried_entries
                else ""
            )
            + (
                f", {self.waiting_entries} left for a later period"
                if self.waiting_entries
                else ""
            )
        )


# ---------------------------------------------------------------------------
# Invoicer
# ---------------------------------------------------------------------------


class BillingInvoicer:
    """Leader-only pass that closes periods into invoices and collects them."""

    def __init__(
        self,
        *,
        mode: Optional[BillingMode] = None,
        period: Optional[BillingPeriod] = None,
        cron: Optional[str] = None,
        lookback: Optional[int] = None,
    ) -> None:
        self._mode = mode or billing_mode()
        self._period = period or billing_period()
        self._cron = cron if cron is not None else envs.BILLING_INVOICE_CRON
        self._lookback = (
            lookback
            if lookback is not None
            else envs.BILLING_INVOICE_LOOKBACK_PERIODS
        )
        try:
            self._trigger = CronTrigger.from_crontab(self._cron, timezone=timezone.utc)
        except Exception as e:
            raise ValueError(
                f"Invalid billing invoice cron (value={self._cron!r}): {e}"
            ) from e
        if self._lookback <= 0:
            raise ValueError("billing invoice lookback must be at least one period")
        self.last_report: Optional[InvoicingReport] = None

    @property
    def mode(self) -> BillingMode:
        return self._mode

    @property
    def period(self) -> BillingPeriod:
        return self._period

    async def start(self) -> None:
        # Invoicing moves money, so like the settler it idles unless the
        # deployment has been switched to enforce. Rating in shadow writes the
        # ledger this module would invoice, which is exactly what makes shadow a
        # useful rehearsal without charging anyone.
        if self._mode is not BillingMode.ENFORCE:
            logger.info(
                f"Billing invoicer idle (mode={self._mode.value}); deferred "
                "charges accrue in the ledger but no invoice is issued and no "
                "wallet is debited."
            )
            return

        logger.info(
            f"Billing invoicer started (mode=enforce, period={self._period.value}, "
            f"cron={self._cron!r} UTC, lookback={self._lookback})."
        )
        # One pass up front rather than waiting for the first fire: a server
        # that was down across a period boundary would otherwise leave that
        # period uncollected for a whole cycle, and a pass with nothing to do is
        # a few queries.
        await self._run_pass("initial")
        while True:
            seconds = self._seconds_until_next_fire()
            if seconds is None:
                logger.error(
                    "Billing invoicer: cron yielded no future fire time; loop stopping."
                )
                return
            await asyncio.sleep(seconds)
            await self._run_pass("scheduled")

    async def _run_pass(self, label: str) -> None:
        try:
            report = await self.invoice_once()
            self.last_report = report
            if (
                report.invoices_issued
                or report.invoices_collected
                or report.unpaid_invoices
                or report.waiting_entries
            ):
                logger.info(f"{label} invoicing pass: {report.summary()}")
            else:
                logger.debug(f"{label} invoicing pass: nothing to invoice")
        except Exception as e:
            # Contained for the same reason as the rater and settler: the leader
            # task loop has no per-task supervision, so an escaping exception
            # would end invoicing until the next restart.
            logger.error(f"Billing invoicing pass failed: {e}", exc_info=True)

    def _seconds_until_next_fire(self) -> Optional[float]:
        now = _utcnow()
        next_fire = self._trigger.get_next_fire_time(None, now)
        if next_fire is None:
            return None
        return max(0.0, (next_fire - now).total_seconds())

    async def invoice_once(self, *, now: Optional[datetime] = None) -> InvoicingReport:
        """One pass. A no-op unless the mode is ``enforce``."""
        started = time.monotonic()
        report = InvoicingReport(mode=self._mode, period=self._period)
        if self._mode is not BillingMode.ENFORCE:
            report.duration_ms = int((time.monotonic() - started) * 1000)
            return report

        moment = as_utc(now) or _utcnow()
        async with async_session() as session:
            # Older debt before new statements: collecting an outstanding invoice
            # first is what lets a top-up that arrived between passes clear it,
            # and keeps a pass from issuing a second statement to an org that has
            # not paid the first.
            await self._collect_outstanding(session, report)
            await self._issue_closed_periods(session, report, now=moment)

        report.duration_ms = int((time.monotonic() - started) * 1000)
        return report

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    async def _collect_outstanding(
        self, session: AsyncSession, report: InvoicingReport
    ) -> None:
        """Retry every issued-but-unpaid invoice, oldest first."""
        invoices = (
            await session.exec(
                select(Invoice)
                .where(
                    Invoice.status == InvoiceStatus.ISSUED.value,
                    Invoice.deleted_at.is_(None),
                )
                .order_by(Invoice.period_start, Invoice.id)
            )
        ).all()
        for invoice in invoices:
            await self._collect(session, invoice, report)
            # Per invoice, as the settler commits per principal: one org's lock
            # timeout must not roll back another's payment.
            await session.commit()

    async def _collect(
        self, session: AsyncSession, invoice: Invoice, report: InvoicingReport
    ) -> None:
        amount = Decimal(invoice.amount or 0)
        if amount <= 0:
            # Nothing to take. Marking it settled is what keeps it out of every
            # later pass and out of the org's outstanding total.
            await self._mark_paid(session, invoice)
            report.invoices_collected += 1
            return

        if not await debit_wallet(session, invoice.principal_id, amount):
            report.unpaid_invoices += 1
            report.amount_unpaid += amount
            invoice.unpaid_reason = REASON_UNPAID
            session.add(invoice)
            await suspend_wallet(session, invoice.principal_id)
            if invoice.principal_id not in report.suspended:
                report.suspended.append(invoice.principal_id)
            logger.warning(
                f"billing: invoice {invoice.id} for principal "
                f"{invoice.principal_id} ({amount} {invoice.currency}) still "
                "uncollectable — wallet suspended"
            )
            return

        entries = await self._mark_paid(session, invoice)
        report.invoices_collected += 1
        report.amount_collected += amount
        wallet = await self._wallet_of(session, invoice.principal_id)
        if wallet is not None and wallet.suspended:
            if await resume_wallet_if_funded(session, invoice.principal_id):
                if invoice.principal_id not in report.resumed:
                    report.resumed.append(invoice.principal_id)
        logger.info(
            f"billing: collected invoice {invoice.id} for principal "
            f"{invoice.principal_id} — {amount} {invoice.currency} over "
            f"{entries} entries"
        )

    async def _mark_paid(self, session: AsyncSession, invoice: Invoice) -> int:
        """Flip the invoice and the entries it claimed to settled.

        Returns how many entries were settled — a count, not the rows, since the
        claim is one UPDATE and nothing here needs the entities.
        """
        invoice.status = InvoiceStatus.SETTLED.value
        invoice.settled_at = _utcnow()
        invoice.unpaid_reason = None
        session.add(invoice)
        result = await session.exec(
            update(LedgerEntry)
            .where(
                LedgerEntry.invoice_id == invoice.id,
                LedgerEntry.status == LedgerStatus.PENDING.value,
            )
            .values(
                status=LedgerStatus.SETTLED.value,
                settled_at=_utcnow(),
            )
        )
        await session.flush()
        return result.rowcount or 0

    async def _wallet_of(self, session: AsyncSession, principal_id: int):
        from gpustack.schemas.billing import Wallet

        return (
            await session.exec(
                select(Wallet).where(
                    Wallet.principal_id == principal_id, Wallet.deleted_at.is_(None)
                )
            )
        ).first()

    # ------------------------------------------------------------------
    # Issuing
    # ------------------------------------------------------------------

    async def _issue_closed_periods(
        self, session: AsyncSession, report: InvoicingReport, *, now: datetime
    ) -> None:
        for period_start, period_end in closed_periods(
            self._period, now=now, lookback=self._lookback
        ):
            report.periods_scanned += 1
            subjects = await self._subjects_with_uninvoiced_usage(
                session, period_end
            )
            for principal_id, entry_count, total in subjects:
                if await self._invoice_exists(session, principal_id, period_start):
                    # One statement per period is the rule, and it is not bent
                    # for late usage: what arrives after a period was invoiced
                    # rides the next one, where it is visible as a carry.
                    report.waiting_entries += entry_count
                    continue
                await self._issue(
                    session,
                    principal_id=principal_id,
                    period_start=period_start,
                    period_end=period_end,
                    report=report,
                )
                await session.commit()

    async def _subjects_with_uninvoiced_usage(
        self, session: AsyncSession, period_end: datetime
    ) -> Sequence[Tuple[int, int, Decimal]]:
        """Principals owing deferred charges incurred before ``period_end``.

        A NULL payer is excluded rather than reported: the rater already records
        that gap as an unpriced/no-payer row, and an invoice with nobody to bill
        would be a statement nobody can pay.
        """
        rows = (
            await session.exec(
                select(
                    LedgerEntry.principal_id,
                    func.count(LedgerEntry.id),
                    func.sum(LedgerEntry.amount),
                )
                .where(*self._claim_predicate(period_end=period_end))
                .group_by(LedgerEntry.principal_id)
            )
        ).all()
        return [
            (principal_id, int(count), Decimal(total or 0))
            for principal_id, count, total in rows
            if principal_id is not None
        ]

    @staticmethod
    def _claim_predicate(
        *, period_end: datetime, principal_id: Optional[int] = None
    ) -> tuple:
        """The one definition of "deferred usage not yet on any invoice".

        Shared by the aggregate that sizes a statement and the UPDATE that claims
        its entries. Two separately written predicates would eventually disagree,
        and the failure is a statement whose total does not match the entries it
        settled — the kind of discrepancy that is only found by a tenant.
        """
        conditions = [
            LedgerEntry.deleted_at.is_(None),
            LedgerEntry.settle_mode == SettleMode.DEFERRED.value,
            LedgerEntry.status == LedgerStatus.PENDING.value,
            LedgerEntry.invoice_id.is_(None),
            LedgerEntry.occurred_at < period_end,
            LedgerEntry.principal_id.is_not(None),
        ]
        if principal_id is not None:
            conditions.append(LedgerEntry.principal_id == principal_id)
        return tuple(conditions)

    async def _invoice_exists(
        self, session: AsyncSession, principal_id: int, period_start: datetime
    ) -> bool:
        existing = (
            await session.exec(
                select(Invoice.id).where(
                    Invoice.principal_id == principal_id,
                    Invoice.period_start == period_start,
                    Invoice.deleted_at.is_(None),
                )
            )
        ).first()
        return existing is not None

    async def _issue(
        self,
        session: AsyncSession,
        *,
        principal_id: int,
        period_start: datetime,
        period_end: datetime,
        report: InvoicingReport,
    ) -> Optional[Invoice]:
        """Build one statement, claim its entries, and take the money."""
        predicate = self._claim_predicate(
            period_end=period_end, principal_id=principal_id
        )

        lines = (
            await session.exec(
                select(
                    LedgerEntry.sku,
                    LedgerEntry.model_name,
                    LedgerEntry.unit,
                    LedgerEntry.currency,
                    func.count(LedgerEntry.id),
                    func.sum(LedgerEntry.quantity),
                    func.sum(LedgerEntry.amount),
                )
                .where(*predicate)
                .group_by(
                    LedgerEntry.sku,
                    LedgerEntry.model_name,
                    LedgerEntry.unit,
                    LedgerEntry.currency,
                )
                .order_by(LedgerEntry.sku, LedgerEntry.model_name)
            )
        ).all()
        if not lines:
            return None

        total = sum((Decimal(row[6] or 0) for row in lines), Decimal(0))
        currencies = {row[3] for row in lines}
        if len(currencies) > 1:
            # Not a state the rater can produce today (one price book, one
            # currency), so it is logged rather than modelled: a mixed-currency
            # statement would need per-currency totals, not a warning.
            logger.warning(
                f"billing: entries for principal {principal_id} in "
                f"{period_start}..{period_end} span currencies {sorted(currencies)}; "
                "invoicing in the first"
            )
        currency = lines[0][3] or "CNY"

        carried = await self._carried_totals(
            session, principal_id=principal_id, period_start=period_start,
            period_end=period_end,
        )
        principal_name = await self._principal_name(session, principal_id, predicate)

        invoice = Invoice(
            principal_id=principal_id,
            principal_name=principal_name,
            period_start=period_start,
            period_end=period_end,
            amount=total,
            currency=currency,
            # Issued first, settled below only if the wallet covers it. The
            # intermediate state is what an operator sees for an org in arrears.
            status=InvoiceStatus.ISSUED,
            issued_at=_utcnow(),
        )
        session.add(invoice)
        await session.flush()  # the items and the claim need its id

        for sku, model_name, unit, _currency, count, quantity, amount in lines:
            session.add(
                InvoiceItem(
                    invoice_id=invoice.id,
                    sku=sku,
                    model_name=model_name,
                    quantity=Decimal(quantity or 0),
                    unit=unit,
                    amount=Decimal(amount or 0),
                    entry_count=int(count or 0),
                )
            )
            report.items += 1

        # Claim the entries before collecting: the UPDATE and the debit are in
        # one transaction, so either both land or neither does, and there is no
        # window in which money has moved for entries still marked uninvoiced.
        claim = await session.exec(
            update(LedgerEntry)
            .where(*predicate)
            .values(invoice_id=invoice.id, updated_at=_utcnow())
        )
        claimed = claim.rowcount or 0

        report.invoices_issued += 1
        report.entries_invoiced += claimed
        report.amount_invoiced += total
        report.carried_entries += carried[0]
        report.carried_amount += carried[1]

        if total <= 0:
            await self._mark_paid(session, invoice)
            report.invoices_collected += 1
            logger.info(
                f"billing: invoice {invoice.id} for principal {principal_id} "
                f"({period_start:%Y-%m-%d}..{period_end:%Y-%m-%d}) is zero — "
                "marked settled without a debit"
            )
            return invoice

        if await debit_wallet(session, principal_id, total):
            entries_paid = await self._mark_paid(session, invoice)
            report.invoices_collected += 1
            report.amount_collected += total
            if entries_paid != claimed:
                # Cannot happen while one leader runs this pass; if it ever does,
                # the statement and its entries disagree and that must be loud.
                logger.error(
                    f"billing: invoice {invoice.id} claimed {claimed} entries but "
                    f"settled {entries_paid}"
                )
            wallet = await self._wallet_of(session, principal_id)
            if wallet is not None and wallet.suspended:
                if await resume_wallet_if_funded(session, principal_id):
                    if principal_id not in report.resumed:
                        report.resumed.append(principal_id)
            logger.info(
                f"billing: issued and collected invoice {invoice.id} for principal "
                f"{principal_id} — {total} {currency} over {claimed} entries "
                f"({period_start:%Y-%m-%d}..{period_end:%Y-%m-%d})"
            )
            return invoice

        # Could not cover the statement. The invoice stays ISSUED with the
        # reason on it, its entries stay claimed but PENDING, and the org is
        # suspended — the same posture as an unpaid realtime charge, and the
        # next pass retries the collection once money arrives.
        invoice.unpaid_reason = REASON_UNPAID
        session.add(invoice)
        await suspend_wallet(session, principal_id)
        if principal_id not in report.suspended:
            report.suspended.append(principal_id)
        report.unpaid_invoices += 1
        report.amount_unpaid += total
        logger.warning(
            f"billing: issued invoice {invoice.id} for principal {principal_id} "
            f"but the wallet could not cover {total} {currency} — invoice unpaid, "
            "wallet suspended"
        )
        return invoice

    async def _carried_totals(
        self,
        session: AsyncSession,
        *,
        principal_id: int,
        period_start: datetime,
        period_end: datetime,
    ) -> Tuple[int, Decimal]:
        """How much of this statement is usage from *before* its own period.

        Counted so an operator reading a statement that looks too large has the
        number to hand, and so a test can assert late usage is collected rather
        than quietly dropped.
        """
        row = (
            await session.exec(
                select(func.count(LedgerEntry.id), func.sum(LedgerEntry.amount)).where(
                    *self._claim_predicate(
                        period_end=period_start, principal_id=principal_id
                    )
                )
            )
        ).first()
        if row is None:
            return 0, Decimal(0)
        # A two-column select hands back a ``Row``, which is a *sequence* but not
        # a ``tuple`` — so ``isinstance(row, tuple)`` is False and a defensive
        # branch built on it silently yields the whole Row instead of its first
        # field. Index it; that is the only shape this query can return.
        return int(row[0] or 0), Decimal(row[1] or 0)

    @staticmethod
    async def _principal_name(
        session: AsyncSession, principal_id: int, predicate: tuple
    ) -> Optional[str]:
        """The name to print on the statement, from the ledger's own snapshot."""
        # Single-column select: ``session.exec`` yields scalars here, not Rows, so
        # the value comes back directly. Indexing it would take a character.
        return (
            await session.exec(
                select(LedgerEntry.principal_name)
                .where(*predicate, LedgerEntry.principal_name.is_not(None))
                .limit(1)
            )
        ).first()
