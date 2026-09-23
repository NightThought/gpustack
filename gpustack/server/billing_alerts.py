"""Billing alerts: what an operator has to act on, and only once per problem.

There is no general alerting subsystem in this codebase to register with — the
platform's alerting story (Prometheus rules + alertmanager + notification
channels) is still an unbuilt item on the roadmap. So this module emits into the
two surfaces that do exist and that a future alertmanager will read:

* **structured logs**, one line per state change, with a stable ``billing-alert:``
  prefix and a ``kind=``/``key=`` pair, so a log router can match on them today;
* **Prometheus gauges**, published by ``exporter.billing_metrics`` from the
  snapshot this module keeps, so an alertmanager rule can fire on
  ``gpustack:billing_alerts_active{kind="unpaid_invoice"} > 0`` without this
  process having to know any notification channel.

What it does not do is notify anybody directly. A module that owned SMTP or
webhook delivery would own retries, credentials and rate limits, and would be
the second such thing in the codebase.

Deduplication is the part that actually needs a design
=====================================================
Every condition here is a *state*, and states are re-observed on every scan.
Logging each observation would mean one line per problem per interval — a
suspended org producing 288 identical warnings a day, which buries the new
problem that arrives in the middle of them. So a problem is raised once, counted
while it persists, and logged again only when it resolves or changes severity.
The count is kept rather than dropped because "this has been unpriced for nine
days and been seen 4000 times" and "this just appeared" are different urgencies,
and the difference is only visible in the number.

The registry is process-local and deliberately not persisted. Alerts are derived
entirely from billing tables, so a restart re-derives them on the first scan;
persisting them would create a second copy of a truth that can disagree with the
first, and the only thing lost is the memory of how long a problem had already
been firing.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.schemas.billing import (
    Invoice,
    InvoiceStatus,
    LedgerEntry,
    LedgerStatus,
    SettleMode,
    Wallet,
)
from gpustack.server.billing_pricing import as_utc
from gpustack.server.billing_settlement import outstanding_charges
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)

# Prefix every emitted line carries, so a log router has one thing to match.
LOG_PREFIX = "billing-alert:"

# Bounds on the registry. An alert per suspended org is one row of memory, so
# the cap is not about ordinary use — it is about a bug that mints unbounded keys
# (a key built from a row id in a table that grows without limit) turning an
# alerting module into a leak.
MAX_ACTIVE_ALERTS = 10000
MAX_RESOLVED_ALERTS = 1000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AlertSeverity(str, Enum):
    WARNING = "warning"
    CRITICAL = "critical"


class BillingAlertKind(str, Enum):
    """The three things about billing that need a human.

    Each is a condition no automated path can fix: a missing price has to be
    written by somebody who knows what the resource costs, an unpaid invoice has
    to be collected or written off, and a suspended wallet is a tenant whose
    service is dark until money arrives or an operator decides otherwise.
    """

    # Usage the rater could not price. Silent revenue loss, and invisible in
    # every usage dashboard because the rows exist and look counted.
    UNPRICED_GAP = "unpriced_gap"
    # A statement issued and not paid within its grace period.
    UNPAID_INVOICE = "unpaid_invoice"
    # A wallet that ran dry, so its org's keys are refused at both request paths.
    WALLET_SUSPENDED = "wallet_suspended"


@dataclass
class BillingAlert:
    """One live problem, and how long it has been one."""

    kind: BillingAlertKind
    key: str
    severity: AlertSeverity
    summary: str
    detail: str = ""
    first_seen: datetime = field(default_factory=_utcnow)
    last_seen: datetime = field(default_factory=_utcnow)
    # Scans that observed it, not emissions. Emitted once; observed every pass.
    occurrences: int = 1
    resolved_at: Optional[datetime] = None

    @property
    def identity(self) -> str:
        return f"{self.kind.value}:{self.key}"

    @property
    def age(self) -> timedelta:
        return _utcnow() - self.first_seen

    def line(self) -> str:
        return (
            f"{LOG_PREFIX} kind={self.kind.value} key={self.key} "
            f"severity={self.severity.value} occurrences={self.occurrences} "
            f"age_seconds={int(self.age.total_seconds())} — {self.summary}"
            + (f" | {self.detail}" if self.detail else "")
        )


class BillingAlertRegistry:
    """Active alerts, deduplicated, and the snapshot the exporter publishes.

    Split from the detector on purpose: the detector is leader-only and periodic,
    while the registry is read by the metrics scrape on another thread and can be
    written by any code path that notices something (a suspension as it happens,
    rather than at the next scan). Reads hand back copies, so a scrape cannot
    observe a half-updated alert.
    """

    def __init__(self) -> None:
        self._active: Dict[str, BillingAlert] = {}
        self._resolved: List[BillingAlert] = []
        # Cumulative, and never decremented: a gauge of currently-active alerts
        # drops when a problem is fixed, which Prometheus reads as a counter
        # reset if it is used as one.
        self._raised_total: Dict[str, int] = {}
        # Last observed values of the database-derived numbers, refreshed by the
        # detector. The scrape reads these rather than querying: /metrics is
        # served from a thread, and a slow billing query there would show up as a
        # scrape timeout on every dashboard in the deployment.
        self._metrics: Dict[str, float] = {}

    def raise_alert(
        self,
        kind: BillingAlertKind,
        key: str,
        *,
        severity: AlertSeverity,
        summary: str,
        detail: str = "",
    ) -> bool:
        """Record a problem. Returns True when this call emitted a log line.

        Emits on three occasions only: the first time, a severity change, and
        never again while the state holds. Everything else increments a counter
        that the eventual resolution line reports, so the noise a long-running
        problem makes is two lines rather than one per scan.
        """
        identity = f"{kind.value}:{key}"
        now = _utcnow()
        existing = self._active.get(identity)
        if existing is not None:
            existing.occurrences += 1
            existing.last_seen = now
            existing.summary = summary
            existing.detail = detail
            if existing.severity is severity:
                return False
            # Escalation is worth a line: "still unpaid" is noise, "unpaid for a
            # week and now critical" is not.
            existing.severity = severity
            logger.warning(existing.line() + " [severity changed]")
            return True

        if len(self._active) >= MAX_ACTIVE_ALERTS:
            # Refuse to grow without bound, and say so once rather than per call.
            if not self._active.get("__cap__"):
                logger.error(
                    f"{LOG_PREFIX} active alert cap ({MAX_ACTIVE_ALERTS}) reached; "
                    "further alerts are counted in metrics only"
                )
            return False

        alert = BillingAlert(
            kind=kind,
            key=key,
            severity=severity,
            summary=summary,
            detail=detail,
            first_seen=now,
            last_seen=now,
        )
        self._active[identity] = alert
        self._raised_total[kind.value] = self._raised_total.get(kind.value, 0) + 1
        logger.warning(alert.line())
        return True

    def resolve(self, kind: BillingAlertKind, key: str) -> bool:
        """Clear a problem. Returns True when it had been active.

        Logs the resolution with the observation count and the age, which is what
        turns a pair of log lines into an answer to "how long were we exposed".
        """
        identity = f"{kind.value}:{key}"
        alert = self._active.pop(identity, None)
        if alert is None:
            return False
        alert.resolved_at = _utcnow()
        self._resolved.append(alert)
        if len(self._resolved) > MAX_RESOLVED_ALERTS:
            del self._resolved[: len(self._resolved) - MAX_RESOLVED_ALERTS]
        logger.info(
            f"{LOG_PREFIX} resolved kind={alert.kind.value} key={alert.key} "
            f"after {int(alert.age.total_seconds())}s and "
            f"{alert.occurrences} observations — {alert.summary}"
        )
        return True

    def resolve_absent(self, kind: BillingAlertKind, present_keys: List[str]) -> int:
        """Resolve every alert of ``kind`` whose key is not in ``present_keys``.

        How a state-based detector clears: it does not track transitions, it
        reports what is true now and anything it said last time that is no longer
        true is resolved. Cheap and correct across restarts, where no transition
        was ever observed.
        """
        keep = {f"{kind.value}:{key}" for key in present_keys}
        stale = [
            identity
            for identity, alert in self._active.items()
            if alert.kind is kind and identity not in keep
        ]
        for identity in stale:
            alert = self._active[identity]
            self.resolve(alert.kind, alert.key)
        return len(stale)

    def active(self) -> List[BillingAlert]:
        return list(self._active.values())

    def resolved(self) -> List[BillingAlert]:
        return list(self._resolved)

    def raised_total(self) -> Dict[str, int]:
        return dict(self._raised_total)

    def publish_metrics(self, values: Dict[str, float]) -> None:
        """Replace the database-derived numbers the exporter serves."""
        self._metrics = dict(values)

    def metrics(self) -> Dict[str, float]:
        return dict(self._metrics)

    def clear(self) -> None:
        """Drop all state. Tests, and only tests, should need this."""
        self._active.clear()
        self._resolved.clear()
        self._raised_total.clear()
        self._metrics.clear()


# One per process, read by the exporter and written by the detector.
billing_alerts = BillingAlertRegistry()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@dataclass
class DetectionReport:
    """What one scan found — returned for tests and logged for operators."""

    raised: int = 0
    resolved: int = 0
    active: int = 0
    unpriced_skus: int = 0
    unpriced_entries: int = 0
    unpaid_invoices: int = 0
    unpaid_amount: Decimal = field(default_factory=lambda: Decimal(0))
    suspended_wallets: int = 0
    pending_realtime_amount: Decimal = field(default_factory=lambda: Decimal(0))
    pending_deferred_amount: Decimal = field(default_factory=lambda: Decimal(0))
    duration_ms: int = 0

    def summary(self) -> str:
        return (
            f"billing alert scan in {self.duration_ms}ms: {self.raised} raised, "
            f"{self.resolved} resolved, {self.active} active "
            f"({self.unpriced_skus} unpriced sku over {self.unpriced_entries} "
            f"entries, {self.unpaid_invoices} unpaid invoice / "
            f"{self.unpaid_amount}, {self.suspended_wallets} suspended wallet)"
        )


class BillingAlertDetector:
    """Leader-only scan that turns billing state into deduplicated alerts."""

    def __init__(
        self,
        *,
        interval_seconds: Optional[int] = None,
        unpaid_grace_days: Optional[int] = None,
        unpaid_critical_days: Optional[int] = None,
    ) -> None:
        self._interval = (
            interval_seconds
            if interval_seconds is not None
            else envs.BILLING_ALERT_INTERVAL_SECONDS
        )
        self._grace_days = (
            unpaid_grace_days
            if unpaid_grace_days is not None
            else envs.BILLING_ALERT_UNPAID_GRACE_DAYS
        )
        self._critical_days = (
            unpaid_critical_days
            if unpaid_critical_days is not None
            else envs.BILLING_ALERT_UNPAID_CRITICAL_DAYS
        )
        if self._interval <= 0:
            raise ValueError("billing alert interval must be positive")
        if self._critical_days < self._grace_days:
            # Escalating before the alert exists would mean a critical that was
            # never a warning, which reads as a bug rather than as an escalation.
            raise ValueError(
                "billing alert unpaid critical days must be at least the grace days"
            )
        self.last_report: Optional[DetectionReport] = None

    async def start(self) -> None:
        logger.info(
            f"Billing alert detector started (interval={self._interval}s, "
            f"unpaid grace={self._grace_days}d, critical={self._critical_days}d)."
        )
        while True:
            try:
                report = await self.detect_once()
                self.last_report = report
                if report.raised or report.resolved:
                    logger.info(report.summary())
            except Exception as e:
                # Contained like the rater, settler and invoicer: the leader task
                # loop has no per-task supervision, and a detector that dies
                # silently is worse than one that logs.
                logger.error(f"Billing alert scan failed: {e}", exc_info=True)
            await asyncio.sleep(self._interval)

    async def detect_once(self, *, now: Optional[datetime] = None) -> DetectionReport:
        """One scan of all three conditions, then publish the metrics snapshot.

        ``now`` is injectable for the same reason the invoicer takes one: every
        threshold here is an age, and a test that cannot fix the clock cannot
        assert "unpaid for four days" without depending on when it happens to
        run.
        """
        started = time.monotonic()
        moment = as_utc(now) or _utcnow()
        report = DetectionReport()
        async with async_session() as session:
            await self._check_unpriced(session, report, now=moment)
            await self._check_unpaid_invoices(session, report, now=moment)
            await self._check_suspensions(session, report, now=moment)
            await self._measure_backlog(session, report)

        report.active = len(billing_alerts.active())
        billing_alerts.publish_metrics(
            {
                "unpriced_skus": report.unpriced_skus,
                "unpriced_entries": report.unpriced_entries,
                "unpaid_invoices": report.unpaid_invoices,
                "unpaid_amount": float(report.unpaid_amount),
                "suspended_wallets": report.suspended_wallets,
                "pending_realtime_amount": float(report.pending_realtime_amount),
                "pending_deferred_amount": float(report.pending_deferred_amount),
            }
        )
        report.duration_ms = int((time.monotonic() - started) * 1000)
        return report

    # -- unpriced usage ---------------------------------------------------

    async def _check_unpriced(
        self, session: AsyncSession, report: DetectionReport, *, now: datetime
    ):
        """Usage the rater could not price, grouped by the SKU that is missing.

        Read from the ledger's VOID rows rather than from the rater's report: the
        report is a property of one sweep in one process, while the rows are the
        durable fact, and they are what survives a restart, a leader change, and
        a rater that was switched off.

        Keyed by SKU and not by row. Rows come and go as prices are added and
        placeholders are promoted, so a per-row key would raise and resolve
        constantly and tell an operator nothing; the question worth answering is
        "which SKU has no price", and one alert per SKU answers it.
        """
        rows = (
            await session.exec(
                select(
                    LedgerEntry.sku,
                    func.count(LedgerEntry.id),
                    func.min(LedgerEntry.occurred_at),
                )
                .where(
                    LedgerEntry.deleted_at.is_(None),
                    LedgerEntry.status == LedgerStatus.VOID.value,
                )
                .group_by(LedgerEntry.sku)
            )
        ).all()

        keys: List[str] = []
        for row in rows:
            sku, count, oldest = row[0], int(row[1] or 0), row[2]
            if not sku or count <= 0:
                continue
            keys.append(sku)
            report.unpriced_skus += 1
            report.unpriced_entries += count
            if billing_alerts.raise_alert(
                BillingAlertKind.UNPRICED_GAP,
                sku,
                severity=AlertSeverity.WARNING,
                summary=(
                    f"{count} usage record(s) cannot be priced under sku "
                    f"'{sku}' and are not being billed"
                ),
                detail=(
                    f"oldest={as_utc(oldest) if oldest is not None else 'unknown'}; "
                    "add a price-book entry for this sku, or the usage stays "
                    "unbilled and the placeholders keep being retried every sweep"
                ),
            ):
                report.raised += 1
        report.resolved += billing_alerts.resolve_absent(
            BillingAlertKind.UNPRICED_GAP, keys
        )

    # -- unpaid invoices --------------------------------------------------

    async def _check_unpaid_invoices(
        self, session: AsyncSession, report: DetectionReport, *, now: datetime
    ):
        """Statements issued and not paid, once past their grace period.

        The grace period exists because issuing and collecting are one step for a
        funded org and two for an unfunded one: an invoice is briefly ISSUED while
        its own transaction decides whether the wallet can cover it. Alerting on
        that would fire for every statement ever written. What matters is the one
        still unpaid days later, which is a collections problem and not a
        transient.
        """
        grace_cutoff = now - timedelta(days=self._grace_days)
        critical_cutoff = now - timedelta(days=self._critical_days)
        invoices = (
            await session.exec(
                select(Invoice).where(
                    Invoice.deleted_at.is_(None),
                    Invoice.status == InvoiceStatus.ISSUED.value,
                    Invoice.issued_at.is_not(None),
                    Invoice.issued_at < grace_cutoff,
                )
            )
        ).all()

        keys: List[str] = []
        for invoice in invoices:
            issued_at = as_utc(invoice.issued_at) or now
            days = int((now - issued_at).total_seconds() // 86400)
            amount = Decimal(invoice.amount or 0)
            keys.append(str(invoice.id))
            report.unpaid_invoices += 1
            report.unpaid_amount += amount
            severity = (
                AlertSeverity.CRITICAL
                if issued_at < critical_cutoff
                else AlertSeverity.WARNING
            )
            if billing_alerts.raise_alert(
                BillingAlertKind.UNPAID_INVOICE,
                str(invoice.id),
                severity=severity,
                summary=(
                    f"invoice {invoice.id} for principal {invoice.principal_id} "
                    f"({invoice.principal_name or 'unnamed'}) has been unpaid for "
                    f"{days} day(s): {amount} {invoice.currency}"
                ),
                detail=(
                    f"period={invoice.period_start:%Y-%m-%d}.."
                    f"{invoice.period_end:%Y-%m-%d}; the org is suspended while it "
                    "owes this, so either collect it (a top-up or an adjustment) "
                    "or void the statement explicitly"
                ),
            ):
                report.raised += 1
        report.resolved += billing_alerts.resolve_absent(
            BillingAlertKind.UNPAID_INVOICE, keys
        )

    # -- suspensions ------------------------------------------------------

    async def _check_suspensions(
        self, session: AsyncSession, report: DetectionReport, *, now: datetime
    ):
        """Wallets that ran dry, with what it would take to resume them.

        The detail carries the outstanding figure rather than just the balance,
        because "balance 0.05" does not tell an operator whether a top-up of 1
        resumes the org or whether 400 is owed. That number is the difference
        between an alert that can be acted on and one that has to be
        investigated first.
        """
        wallets = (
            await session.exec(
                select(Wallet).where(
                    Wallet.deleted_at.is_(None), Wallet.suspended.is_(True)
                )
            )
        ).all()

        keys: List[str] = []
        for wallet in wallets:
            keys.append(str(wallet.principal_id))
            report.suspended_wallets += 1
            owed = await outstanding_charges(session, wallet.principal_id)
            suspended_at = as_utc(wallet.suspended_at)
            hours = (
                int((now - suspended_at).total_seconds() // 3600)
                if suspended_at is not None
                else None
            )
            if billing_alerts.raise_alert(
                BillingAlertKind.WALLET_SUSPENDED,
                str(wallet.principal_id),
                severity=AlertSeverity.WARNING,
                summary=(
                    f"principal {wallet.principal_id} "
                    f"({wallet.principal_name or 'unnamed'}) is suspended for "
                    f"arrears: balance {wallet.balance} {wallet.currency}, "
                    f"outstanding {owed}"
                ),
                detail=(
                    "its api keys are refused with 402 on both request paths"
                    + (f"; suspended {hours} hour(s) ago" if hours is not None else "")
                    + f"; a top-up or adjustment covering {owed} resumes it"
                ),
            ):
                report.raised += 1
        report.resolved += billing_alerts.resolve_absent(
            BillingAlertKind.WALLET_SUSPENDED, keys
        )

    # -- backlog ----------------------------------------------------------

    async def _measure_backlog(self, session: AsyncSession, report: DetectionReport):
        """What is rated but not yet collected, by settle mode.

        Not an alert — nothing here is wrong on its own, and deferred charges are
        *supposed* to sit uncollected until their period closes. It is published
        as a gauge because it is the number that answers "is the pipeline keeping
        up": a realtime figure that grows across scans means the settler is behind
        or failing, which is the one billing outage that is invisible to tenants
        until it is not.
        """
        rows = (
            await session.exec(
                select(
                    LedgerEntry.settle_mode,
                    func.count(LedgerEntry.id),
                    func.sum(LedgerEntry.amount),
                )
                .where(
                    LedgerEntry.deleted_at.is_(None),
                    LedgerEntry.status == LedgerStatus.PENDING.value,
                )
                .group_by(LedgerEntry.settle_mode)
            )
        ).all()
        for row in rows:
            mode, _count, total = row[0], int(row[1] or 0), Decimal(row[2] or 0)
            if mode == SettleMode.REALTIME.value:
                report.pending_realtime_amount += total
            elif mode == SettleMode.DEFERRED.value:
                report.pending_deferred_amount += total

    # -- direct emission --------------------------------------------------

    @staticmethod
    def note_suspension(
        principal_id: int, *, balance: Decimal, owed: Decimal, currency: str = "CNY"
    ) -> None:
        """Raise a suspension alert at the moment it happens.

        Called from the settlement path, so an operator hears about a tenant going
        dark within the sweep that suspended it rather than at the next scan. The
        scan still owns resolution: it is the only place that can tell a wallet
        that resumed from one that was deleted.
        """
        billing_alerts.raise_alert(
            BillingAlertKind.WALLET_SUSPENDED,
            str(principal_id),
            severity=AlertSeverity.WARNING,
            summary=(
                f"principal {principal_id} just suspended for arrears: balance "
                f"{balance} {currency}, outstanding {owed}"
            ),
            detail="emitted at suspension time; the next alert scan keeps it "
            "current and resolves it when the wallet is funded again",
        )
