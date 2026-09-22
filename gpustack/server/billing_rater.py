"""Rating: turn metered usage into priced ledger entries (WP3).

The rater is the only thing that reads the platform's two metering tables and
writes ``billing_ledger``. It is deliberately the *only* writer: one code path
knows how a token count or a GPU-second becomes money, so a bill can always be
reproduced from the usage rows plus the price book.

Sources, and what each contributes:

* ``model_usage_details`` — one row per request, already gated by ``completed``
  (an interrupted stream is estimated, and the platform's own rule is that an
  interrupted request is not charged). Rated into up to three token SKUs:
  prompt / cached / completion. Settles **realtime**.
* ``metered_usage`` — hourly resource buckets, rated only once ``sealed_at`` is
  set, because an open bucket can still grow and charging it early would bill an
  hour that has not happened yet. Settles **deferred** (at invoicing, WP6).

Idempotency and the VOID placeholder
------------------------------------
A source row counts as rated when a **non-VOID** ledger row exists for it. Rows
that cannot be priced — a model nobody priced yet, a CPU-only instance with no
SKU, a usage row with no attributable payer — get a VOID entry instead of being
silently skipped. That buys two things: the gap is queryable (``status='void'``
is the "unbilled backlog" report), and the row keeps being retried, so the
moment a price appears the VOID entry is promoted to a real charge rather than
lost. The unique key ``(source_table, source_id, sku)`` makes the promotion an
update, never a duplicate.

Modes (``GPUSTACK_BILLING_MODE``)
--------------------------------
``off`` writes nothing. ``shadow`` writes the ledger but never touches a wallet
— the reconciliation mode, and the default. ``enforce`` adds settlement (WP4).
Rating is identical in shadow and enforce, which is the point: what you
reconciled is what you later charge.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple

from sqlalchemy import exists
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.schemas.billing import (
    SKU_GPU_HOUR_PREFIX,
    SKU_STORAGE_GB_HOUR,
    SKU_TOKEN_CACHED,
    SKU_TOKEN_COMPLETION,
    SKU_TOKEN_PROMPT,
    UNIT_GB_HOURS,
    UNIT_GPU_HOURS,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    PriceBookEntry,
    SettleMode,
    settle_mode_for_sku,
    unit_for_sku,
)
from gpustack.schemas.metered_usage import (
    METER_INSTANCE_UPTIME,
    METER_STORAGE_CAPACITY,
    MeteredUsage,
)
from gpustack.schemas.model_usage_details import ModelUsageDetails
from gpustack.server.billing_pricing import (
    as_utc,
    compute_amount,
    invalidate_price_cache,
    resolve_price,
)
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)

# Ledger provenance keys — the ``source_table`` half of the idempotency key.
SOURCE_USAGE_DETAILS = "model_usage_details"
SOURCE_METERED_USAGE = "metered_usage"

_SECONDS_PER_HOUR = Decimal(3600)
_MIB_PER_GB = Decimal(1024)

# SKU stamped on a VOID entry whose source has no billable SKU at all (a
# CPU-only instance, say). Not a priced SKU: it exists to occupy the idempotency
# key so the row is visibly skipped rather than rescanned forever.
SKU_OUT_OF_SCOPE = "unpriced.out_of_scope"
# Unit for VOID entries, which by definition price nothing.
UNIT_UNPRICED = "unpriced"


class BillingMode(str, Enum):
    """Master switch for the billing pipeline."""

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


def billing_mode(raw: Optional[str] = None) -> BillingMode:
    """Parse the configured mode, refusing an unknown value.

    Falling back to a default here would be the wrong failure: an operator who
    typo'd ``GPUSTACK_BILLING_MODE=enfoce`` should get a startup error, not a
    platform that quietly never bills anyone.
    """
    value = (raw if raw is not None else envs.BILLING_MODE).strip().lower()
    try:
        return BillingMode(value)
    except ValueError:
        raise ValueError(
            f"Invalid GPUSTACK_BILLING_MODE {value!r}; "
            f"expected one of: {', '.join(m.value for m in BillingMode)}"
        ) from None


@dataclass
class RatingReport:
    """What one sweep did — returned for tests and logged for operators."""

    mode: BillingMode = BillingMode.SHADOW
    token_sources: int = 0
    token_entries: int = 0
    token_unpriced: int = 0
    resource_sources: int = 0
    resource_entries: int = 0
    resource_unpriced: int = 0
    promoted: int = 0  # VOID entries turned into real charges
    duration_ms: int = 0
    unpriced_reasons: List[str] = field(default_factory=list)

    @property
    def entries(self) -> int:
        return self.token_entries + self.resource_entries

    def summary(self) -> str:
        return (
            f"billing rating sweep ({self.mode.value}) in {self.duration_ms}ms: "
            f"{self.token_entries} token entries from {self.token_sources} requests, "
            f"{self.resource_entries} resource entries from "
            f"{self.resource_sources} buckets, {self.promoted} promoted, "
            f"{self.token_unpriced + self.resource_unpriced} unpriced"
        )


class BillingRater:
    """Leader-only loop that prices unrated usage into the ledger.

    Interval-based rather than cron-based: rating is a drain-the-backlog sweep
    whose cadence trades bill freshness against query load, not a job that must
    land in a particular window (the archiver beside it is cron-based for exactly
    the opposite reason).
    """

    def __init__(
        self,
        *,
        mode: Optional[BillingMode] = None,
        interval_seconds: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self._mode = mode or billing_mode()
        self._interval = (
            interval_seconds
            if interval_seconds is not None
            else envs.BILLING_RATE_INTERVAL_SECONDS
        )
        self._batch_size = (
            batch_size if batch_size is not None else envs.BILLING_RATE_BATCH_SIZE
        )
        if self._interval <= 0:
            raise ValueError("billing rate interval must be positive")
        if self._batch_size <= 0:
            raise ValueError("billing rate batch size must be positive")
        self.last_report: Optional[RatingReport] = None

    @property
    def mode(self) -> BillingMode:
        return self._mode

    async def start(self) -> None:
        if self._mode is BillingMode.OFF:
            logger.info(
                "Billing rater disabled (GPUSTACK_BILLING_MODE=off); "
                "usage will not be rated into the ledger."
            )
            return

        logger.info(
            f"Billing rater started (mode={self._mode.value}, "
            f"interval={self._interval}s, batch={self._batch_size})."
        )
        while True:
            try:
                report = await self.rate_once()
                self.last_report = report
                if report.entries or report.promoted or report.unpriced_reasons:
                    logger.info(report.summary())
                    for reason in report.unpriced_reasons[:10]:
                        logger.warning(f"billing: unpriced usage — {reason}")
            except Exception as e:
                # Never let a rating failure escape: the leader task loop has no
                # per-task supervision, and a billing bug must not take down the
                # scheduler or controllers running beside it.
                logger.error(f"Billing rating sweep failed: {e}", exc_info=True)
            await asyncio.sleep(self._interval)

    async def rate_once(self) -> RatingReport:
        """One sweep over both sources. Safe to call repeatedly."""
        started = time.monotonic()
        report = RatingReport(mode=self._mode)
        if self._mode is BillingMode.OFF:
            report.duration_ms = int((time.monotonic() - started) * 1000)
            return report

        # Refresh the price book at the top of every sweep. The cache exists to
        # keep one sweep from re-reading prices per usage row, not to survive
        # between sweeps: with the TTL alone, a price added after a sweep would
        # stay invisible for minutes and every request in that window would be
        # VOIDed as unpriced — visible churn and a delayed bill. One extra query
        # per tick is nothing against the rows the tick is about to read, and it
        # also covers prices written through another server instance, whose own
        # cache invalidation cannot reach this process.
        invalidate_price_cache()

        async with async_session() as session:
            await self._rate_token_details(session, report)
            await self._rate_resource_buckets(session, report)
            await session.commit()

        report.duration_ms = int((time.monotonic() - started) * 1000)
        return report

    # ------------------------------------------------------------------
    # Token side — model_usage_details
    # ------------------------------------------------------------------

    async def _rate_token_details(
        self, session: AsyncSession, report: RatingReport
    ) -> None:
        details = await self._unrated(session, ModelUsageDetails, SOURCE_USAGE_DETAILS)
        if not details:
            return
        report.token_sources = len(details)
        existing = await self._existing_entries(
            session, SOURCE_USAGE_DETAILS, [d.id for d in details]
        )

        for detail in details:
            occurred_at = as_utc(detail.completed_at or detail.created_at) or _utcnow()
            payer = detail.consumer_principal_id or detail.owner_principal_id
            quantities = self._token_quantities(detail)
            entries = await self._price_quantities(
                session,
                quantities=quantities,
                source_table=SOURCE_USAGE_DETAILS,
                source_id=detail.id,
                payer=payer,
                occurred_at=occurred_at,
                model_name=detail.model_name,
                report=report,
                attribution=dict(
                    user_id=detail.user_id,
                    user_name=detail.user_name,
                    api_key_id=detail.api_key_id,
                    api_key_name=detail.api_key_name,
                    model_name=detail.model_name,
                    cluster_id=detail.cluster_id,
                    request_id=detail.request_id,
                ),
            )
            self._stage(session, entries, existing, report)

    @staticmethod
    def _token_quantities(detail: ModelUsageDetails) -> List[Tuple[str, Decimal]]:
        """Split one request into the SKUs it is billed under.

        ``prompt_cached_token_count`` is the subset of the prompt served from
        cache, so the prompt SKU is billed on the remainder — charging the full
        prompt count *and* the cached count would bill those tokens twice.
        Zero-quantity SKUs are dropped: a request with no cache hit should not
        produce a zero-amount ledger row.
        """
        cached = Decimal(detail.prompt_cached_token_count or 0)
        prompt_total = Decimal(detail.prompt_token_count or 0)
        prompt_billable = prompt_total - cached
        if prompt_billable < 0:
            # A cached count above the prompt count is an upstream reporting
            # bug; clamp rather than emit a negative charge.
            logger.warning(
                f"usage detail {detail.id}: cached tokens ({cached}) exceed prompt "
                f"tokens ({prompt_total}); clamping the prompt SKU to zero"
            )
            prompt_billable = Decimal(0)

        pairs = [
            (SKU_TOKEN_PROMPT, prompt_billable),
            (SKU_TOKEN_CACHED, cached),
            (SKU_TOKEN_COMPLETION, Decimal(detail.completion_token_count or 0)),
        ]
        return [(sku, qty) for sku, qty in pairs if qty > 0]

    # ------------------------------------------------------------------
    # Resource side — metered_usage
    # ------------------------------------------------------------------

    async def _rate_resource_buckets(
        self, session: AsyncSession, report: RatingReport
    ) -> None:
        buckets = await self._unrated(
            session, MeteredUsage, SOURCE_METERED_USAGE, sealed_only=True
        )
        if not buckets:
            return
        report.resource_sources = len(buckets)
        existing = await self._existing_entries(
            session, SOURCE_METERED_USAGE, [b.id for b in buckets]
        )

        for bucket in buckets:
            sku_quantity = self._resource_sku_quantity(bucket)
            payer = bucket.consumer_principal_id or bucket.owner_principal_id
            entries: List[LedgerEntry] = []
            if sku_quantity is None:
                # Out of v1 scope (CPU-only instance, or a GPU row with no
                # accelerator type to price by). VOID it so the gap is visible
                # and the row is not rescanned as though it were new.
                report.resource_unpriced += 1
                report.unpriced_reasons.append(
                    f"metered_usage id={bucket.id} meter={bucket.meter_key} "
                    f"resource={bucket.resource_name} has no billable SKU"
                )
                entries.append(
                    self._void_entry(
                        source_table=SOURCE_METERED_USAGE,
                        source_id=bucket.id,
                        sku=SKU_OUT_OF_SCOPE,
                        payer=payer,
                        occurred_at=as_utc(bucket.bucket_start) or _utcnow(),
                        attribution=self._resource_attribution(bucket),
                    )
                )
            else:
                sku, quantity, unit = sku_quantity
                entries = await self._price_quantities(
                    session,
                    quantities=[(sku, quantity)],
                    source_table=SOURCE_METERED_USAGE,
                    source_id=bucket.id,
                    payer=payer,
                    occurred_at=as_utc(bucket.bucket_start) or _utcnow(),
                    model_name=None,
                    report=report,
                    attribution=self._resource_attribution(bucket),
                    unpriced_counter="resource",
                )
            self._stage(session, entries, existing, report)

    @staticmethod
    def _resource_sku_quantity(
        bucket: MeteredUsage,
    ) -> Optional[Tuple[str, Decimal, str]]:
        """Map a metered bucket onto ``(sku, quantity_in_billing_units, unit)``.

        Conversions are the ones ``metered_usage``'s own contract documents:
        GPU-hours = seconds × card count / 3600 (``sku_count`` is fractional for
        a sliced card, which is why the multiplier is a Decimal), and GB-hours =
        MiB-seconds / 1024 / 3600.
        """
        quantity_seconds = Decimal(bucket.quantity or 0)
        if bucket.meter_key == METER_INSTANCE_UPTIME:
            gpu_type = (bucket.dimensions or {}).get("gpu_type")
            cards = Decimal(str(bucket.sku_count or 0))
            if not gpu_type or cards <= 0:
                return None
            gpu_hours = quantity_seconds * cards / _SECONDS_PER_HOUR
            if gpu_hours <= 0:
                return None
            return f"{SKU_GPU_HOUR_PREFIX}{gpu_type}", gpu_hours, UNIT_GPU_HOURS
        if bucket.meter_key == METER_STORAGE_CAPACITY:
            gb_hours = quantity_seconds / _MIB_PER_GB / _SECONDS_PER_HOUR
            if gb_hours <= 0:
                return None
            return SKU_STORAGE_GB_HOUR, gb_hours, UNIT_GB_HOURS
        return None

    @staticmethod
    def _resource_attribution(bucket: MeteredUsage) -> Dict[str, object]:
        return dict(
            principal_name=bucket.consumer_name or bucket.owner_name,
            cluster_id=bucket.cluster_id,
            resource_id=bucket.resource_id,
            resource_name=bucket.resource_name,
            user_id=bucket.creator_id,
            user_name=bucket.creator_name,
        )

    # ------------------------------------------------------------------
    # Shared machinery
    # ------------------------------------------------------------------

    async def _unrated(
        self, session: AsyncSession, model, source_table: str, sealed_only=False
    ) -> List:
        """Source rows with no non-VOID ledger entry yet.

        VOID rows deliberately do not count as rated: they are a marker for
        "could not price this", and the row must be retried so a price added
        later still bills the usage. Bounded by ``batch_size`` per sweep.
        """
        rated = exists(
            select(LedgerEntry.id).where(
                LedgerEntry.source_table == source_table,
                LedgerEntry.source_id == model.id,
                LedgerEntry.status != LedgerStatus.VOID.value,
            )
        )
        conditions = [model.deleted_at.is_(None), ~rated]
        if sealed_only:
            # An open bucket is still accumulating; rating it would bill an hour
            # that has not finished happening.
            conditions.append(MeteredUsage.sealed_at.is_not(None))
        if model is ModelUsageDetails:
            # The platform's billing gate: an interrupted request's token counts
            # are server-side estimates, and the rule is not to charge it.
            conditions.append(ModelUsageDetails.completed.is_(True))
        stmt = (
            select(model).where(*conditions).order_by(model.id).limit(self._batch_size)
        )
        return list((await session.exec(stmt)).all())

    async def _existing_entries(
        self, session: AsyncSession, source_table: str, source_ids: List[int]
    ) -> Dict[Tuple[int, str], LedgerEntry]:
        """Ledger rows already written for this batch, keyed by (source, sku).

        One query for the whole batch: per-row lookups would turn a 500-row sweep
        into 1500 round trips.
        """
        if not source_ids:
            return {}
        stmt = select(LedgerEntry).where(
            LedgerEntry.source_table == source_table,
            LedgerEntry.source_id.in_(source_ids),
        )
        rows = (await session.exec(stmt)).all()
        return {(row.source_id, row.sku): row for row in rows}

    async def _price_quantities(
        self,
        session: AsyncSession,
        *,
        quantities: List[Tuple[str, Decimal]],
        source_table: str,
        source_id: int,
        payer: Optional[int],
        occurred_at: datetime,
        model_name: Optional[str],
        report: RatingReport,
        attribution: Dict[str, object],
        unpriced_counter: str = "token",
    ) -> List[LedgerEntry]:
        """Price each (sku, quantity) pair, VOIDing what has no price.

        All-or-nothing per source row: if any SKU of a request is unpriced the
        whole row is VOIDed, because a bill missing one third of a request is
        harder to notice than one missing the request.
        """
        if payer is None:
            report.unpriced_reasons.append(
                f"{source_table} id={source_id} has no attributable payer "
                "(neither consumer nor owner principal)"
            )
            if unpriced_counter == "token":
                report.token_unpriced += 1
            else:
                report.resource_unpriced += 1
            return [
                self._void_entry(
                    source_table=source_table,
                    source_id=source_id,
                    sku=quantities[0][0] if quantities else SKU_OUT_OF_SCOPE,
                    payer=None,
                    occurred_at=occurred_at,
                    attribution=attribution,
                )
            ]

        priced: List[Tuple[str, Decimal, PriceBookEntry, Decimal]] = []
        missing: List[str] = []
        for sku, quantity in quantities:
            entry = await resolve_price(
                session, sku=sku, model_name=model_name, at=occurred_at
            )
            if entry is None:
                missing.append(sku)
                continue
            priced.append((sku, quantity, entry, compute_amount(entry, quantity)))

        if missing:
            report.unpriced_reasons.append(
                f"{source_table} id={source_id} model={model_name!r} has no active "
                f"price for: {', '.join(missing)}"
            )
            if unpriced_counter == "token":
                report.token_unpriced += 1
            else:
                report.resource_unpriced += 1
            return [
                self._void_entry(
                    source_table=source_table,
                    source_id=source_id,
                    sku=sku,
                    payer=payer,
                    occurred_at=occurred_at,
                    attribution=attribution,
                )
                for sku in {sku for sku, _ in quantities}
            ]

        entries = []
        for sku, quantity, price_entry, amount in priced:
            entries.append(
                LedgerEntry(
                    source_table=source_table,
                    source_id=source_id,
                    principal_id=payer,
                    sku=sku,
                    quantity=quantity,
                    unit=unit_for_sku(sku),
                    unit_price=price_entry.price,
                    price_book_id=price_entry.id,
                    price_book_version=price_entry.version,
                    # Signed: a debit is money leaving the wallet, so the column
                    # sums to the net charge over any period.
                    amount=amount,
                    currency=price_entry.currency,
                    direction=LedgerDirection.DEBIT,
                    settle_mode=settle_mode_for_sku(sku),
                    status=LedgerStatus.PENDING,
                    occurred_at=occurred_at,
                    **attribution,
                )
            )
        if unpriced_counter == "token":
            report.token_entries += len(entries)
        else:
            report.resource_entries += len(entries)
        return entries

    def _void_entry(
        self,
        *,
        source_table: str,
        source_id: int,
        sku: str,
        payer: Optional[int],
        occurred_at: datetime,
        attribution: Dict[str, object],
    ) -> LedgerEntry:
        """A marker for usage that could not be priced.

        Carries zero quantity and zero amount so it never contributes to a bill,
        and keeps the source's identity so an operator can find what is missing.
        """
        return LedgerEntry(
            source_table=source_table,
            source_id=source_id,
            principal_id=payer,
            sku=sku,
            quantity=Decimal(0),
            unit=UNIT_UNPRICED,
            unit_price=Decimal(0),
            amount=Decimal(0),
            direction=LedgerDirection.DEBIT,
            settle_mode=settle_mode_for_sku(sku)
            if _is_known_sku(sku)
            else SettleMode.DEFERRED,
            status=LedgerStatus.VOID,
            occurred_at=occurred_at,
            **attribution,
        )

    @staticmethod
    def _stage(
        session: AsyncSession,
        entries: List[LedgerEntry],
        existing: Dict[Tuple[int, str], LedgerEntry],
        report: RatingReport,
    ) -> None:
        """Add new entries; promote VOID ones whose price has since appeared."""
        for entry in entries:
            prior = existing.get((entry.source_id, entry.sku))
            if prior is None:
                session.add(entry)
                continue
            if prior.status == LedgerStatus.VOID.value and entry.status != (
                LedgerStatus.VOID
            ):
                # The price exists now: fill the placeholder in place, so the
                # charge lands without a second row for the same source.
                prior.status = LedgerStatus.PENDING.value
                prior.quantity = entry.quantity
                prior.unit = entry.unit
                prior.unit_price = entry.unit_price
                prior.amount = entry.amount
                prior.currency = entry.currency
                prior.price_book_id = entry.price_book_id
                prior.price_book_version = entry.price_book_version
                prior.principal_id = entry.principal_id or prior.principal_id
                session.add(prior)
                report.promoted += 1
            # Otherwise (already a real charge, or still unpriced) leave it alone:
            # re-rating settled history is how a bill stops being reproducible.

    async def rate_and_settle_once(self) -> RatingReport:
        """Rating plus settlement — the enforce-mode entry point.

        Settlement itself lands with WP4; until then enforce mode rates exactly
        as shadow does and says so, rather than pretending wallets were debited.
        """
        report = await self.rate_once()
        if self._mode is BillingMode.ENFORCE:
            logger.warning(
                "billing mode is 'enforce' but settlement is not implemented yet "
                "(WP4); the ledger was written and no wallet was debited"
            )
        return report


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _is_known_sku(sku: str) -> bool:
    """Whether ``sku`` is one this platform knows how to price.

    VOID entries can carry ``SKU_OUT_OF_SCOPE``, which is not a priced SKU, so
    the settle mode of such a row cannot come from ``settle_mode_for_sku``.
    """
    try:
        unit_for_sku(sku)
        return True
    except ValueError:
        return False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
