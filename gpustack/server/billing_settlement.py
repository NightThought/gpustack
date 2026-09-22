"""Settlement: move rated charges onto wallets (WP4).

The rater prices usage into ``billing_ledger``; this module is the only thing
that turns a ledger entry into money leaving a wallet. Keeping the two apart is
what makes shadow mode meaningful — rating can run for weeks, reconciled against
the usage tables, without a single balance moving.

Prepaid and non-negative, per the WP0 decision: a wallet is debited with one
conditional UPDATE (``balance = balance - :n WHERE balance >= :n``), so the
balance check and the deduction are a single statement and concurrent charges
cannot overdraw it. No row locks, no version column, no read-then-write race —
the SQL-level equivalent of the Lua reserve script new-api uses.

Deferred entries are *not* settled here. ``gpu.hour.*`` and ``storage.gb.hour``
charges wait for the invoice that covers their period (WP6); settling them on
arrival would debit a wallet for an hour the operator has not been billed for
yet, and would make the invoice a report rather than the thing that charges.

Insufficient balance settles partially, oldest first, and suspends the wallet:
collecting what the balance covers beats collecting nothing, and the remainder
stays PENDING so a top-up is followed by the arrears being settled on the next
sweep rather than written off.

Each sweep per principal produces one ``BillingSession`` — the batch-level
counterpart of new-api's per-request session. Its ``settled`` / ``refunded``
flags are one-way and mutually exclusive, which is what makes both a repeated
sweep and a reversal safe to retry. A session whose ``actual`` is 0 is kept, not
discarded: "we tried to collect X and got nothing" is exactly the record an
operator needs when a tenant disputes a suspension.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from sqlalchemy import update
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.api.exceptions import AlreadyExistsException, InvalidException
from gpustack.schemas.billing import (
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
from gpustack.server.billing_pricing import as_utc
from gpustack.server.billing_enforcement import (
    resume_keys_for_wallet,
    suspend_keys_for_wallet,
)
from gpustack.server.billing_rater import BillingMode, billing_mode
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)

# Ledger provenance for wallet movements that do not come from a usage table.
SOURCE_REDEMPTION = "billing_redemption"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Wallet primitives
# ---------------------------------------------------------------------------


async def get_or_create_wallet(
    session: AsyncSession, principal_id: int, *, commit: bool = False
) -> Wallet:
    """The wallet for one principal, created on first use.

    Created lazily rather than when an org is provisioned: a wallet that has
    never held money is noise, and the first charge or top-up is a natural
    moment. ``uq_wallet_principal`` makes a concurrent create lose cleanly.
    """
    wallet = (
        await session.exec(
            select(Wallet).where(
                Wallet.principal_id == principal_id, Wallet.deleted_at.is_(None)
            )
        )
    ).first()
    if wallet is not None:
        return wallet

    wallet = Wallet(principal_id=principal_id, balance=Decimal(0))
    session.add(wallet)
    await session.flush()
    if commit:
        await session.commit()
    return wallet


async def debit_wallet(
    session: AsyncSession, principal_id: int, amount: Decimal
) -> bool:
    """Take ``amount`` from a wallet if and only if it can cover it.

    Returns False when the balance is short — the caller decides what that
    means (partial settlement, suspension). A non-positive amount is a
    programming error, not a no-op: silently accepting it would let a negative
    charge become a credit.
    """
    if amount <= 0:
        raise ValueError(f"debit amount must be positive, got {amount}")
    result = await session.exec(
        update(Wallet)
        .where(Wallet.principal_id == principal_id, Wallet.balance >= amount)
        .values(balance=Wallet.balance - amount)
    )
    return result.rowcount == 1


async def credit_wallet(
    session: AsyncSession, principal_id: int, amount: Decimal
) -> None:
    """Add ``amount`` to a wallet.

    A plain unconditional increment: crediting is not a race the way debiting
    is, because there is no ceiling a top-up can breach (the column is
    Numeric(20, 8), and the amount comes from a redemption code or an operator,
    not from a request). Note this is NOT idempotent — calling it twice credits
    twice — so every caller pairs it with something that is (a redemption's CAS
    flip, a ledger unique key).
    """
    if amount <= 0:
        raise ValueError(f"credit amount must be positive, got {amount}")
    await get_or_create_wallet(session, principal_id)
    await session.exec(
        update(Wallet)
        .where(Wallet.principal_id == principal_id)
        .values(balance=Wallet.balance + amount)
    )


async def suspend_wallet(
    session: AsyncSession, principal_id: int, *, commit: bool = False
) -> Wallet:
    """Flag a wallet whose balance ran dry, and refuse its keys.

    Two writes that must travel together. The wallet flag is what the settlement
    side reads; the key flag is what the gateway's local auth table is built
    from. Suspending only the wallet would leave the plugin verifying the key
    locally — and a locally-verified key never reaches ``/token-auth``, so the
    org would keep being served until some later reconcile dropped it.
    """
    wallet = await get_or_create_wallet(session, principal_id)
    if not wallet.suspended:
        wallet.suspended = True
        wallet.suspended_at = _utcnow()
        session.add(wallet)
        await session.flush()
    # Propagated even when the wallet was already flagged: a key created after
    # the suspension would otherwise never be flagged, and this pass is cheap.
    await suspend_keys_for_wallet(session, principal_id, commit=False)
    if commit:
        await session.commit()
    return wallet


async def outstanding_realtime_charges(
    session: AsyncSession, principal_id: int
) -> Decimal:
    """What one principal still owes on realtime charges.

    The number a suspension decision has to be made against: a balance above
    zero is not the same as a balance that covers the arrears, and treating it
    as such resumes a tenant who has paid part of what they owe.
    """
    rows = (
        await session.exec(
            select(LedgerEntry.amount).where(
                LedgerEntry.principal_id == principal_id,
                LedgerEntry.status == LedgerStatus.PENDING.value,
                LedgerEntry.settle_mode == SettleMode.REALTIME.value,
                LedgerEntry.deleted_at.is_(None),
            )
        )
    ).all()
    return sum((Decimal(r[0] if isinstance(r, tuple) else r) for r in rows), Decimal(0))


async def resume_wallet_if_funded(
    session: AsyncSession, principal_id: int, *, commit: bool = False
) -> bool:
    """Clear a suspension once the balance can cover what is owed.

    Two conditions, and both matter. ``balance > 0``: a prepaid wallet with
    nothing in it gets no service, owed or not. ``balance >= outstanding``: a
    top-up that covers part of the arrears must not resume the tenant, or they
    would keep consuming while the rest waits. Called after a top-up and after a
    settlement sweep, so an org is never left suspended once it has actually
    paid — the failure mode a tenant notices first.
    """
    wallet = await get_or_create_wallet(session, principal_id)
    if not wallet.suspended:
        return True
    if wallet.balance <= 0:
        return False
    if wallet.balance < await outstanding_realtime_charges(session, principal_id):
        return False
    wallet.suspended = False
    wallet.suspended_at = None
    session.add(wallet)
    await session.flush()
    # Clearing the wallet without clearing its keys would leave the tenant
    # paying but still refused at the gateway, which is the complaint that
    # reaches an operator first.
    await resume_keys_for_wallet(session, principal_id, commit=False)
    if commit:
        await session.commit()
    return True


# ---------------------------------------------------------------------------
# Top-up by redemption code
# ---------------------------------------------------------------------------


async def redeem_code(
    session: AsyncSession,
    *,
    code: str,
    principal_id: int,
    user_id: Optional[int] = None,
) -> Redemption:
    """Apply a single-use top-up code to a principal's wallet.

    One transaction, three steps that must not be separated: lock the row, flip
    its status under a ``WHERE status = 'enabled'`` guard, credit the wallet.
    The guard is the whole safety argument — two concurrent redemptions of one
    code both read ENABLED, but only one UPDATE matches the guard, and the loser
    sees zero rows and aborts before any money moves.
    """
    statement = select(Redemption).where(Redemption.code == code)
    try:
        statement = statement.with_for_update()
    except Exception:  # pragma: no cover - backends without row locks
        logger.debug("SELECT ... FOR UPDATE unavailable; relying on the CAS guard")

    redemption = (await session.exec(statement)).first()
    if redemption is None or redemption.deleted_at is not None:
        raise InvalidException(message=f"Redemption code not found: {code}")
    if redemption.status != RedemptionStatus.ENABLED.value:
        raise AlreadyExistsException(message=f"Redemption code already used: {code}")
    expires_at = as_utc(redemption.expires_at)
    if expires_at is not None and expires_at <= _utcnow():
        raise InvalidException(message=f"Redemption code expired: {code}")

    # CAS: the guard is the status, so a concurrent winner makes this affect
    # zero rows and the credit below never runs.
    result = await session.exec(
        update(Redemption)
        .where(
            Redemption.id == redemption.id,
            Redemption.status == RedemptionStatus.ENABLED.value,
        )
        .values(
            status=RedemptionStatus.USED.value,
            used_by_principal_id=principal_id,
            used_by_user_id=user_id,
            used_at=_utcnow(),
        )
    )
    if result.rowcount != 1:
        await session.rollback()
        raise AlreadyExistsException(message=f"Redemption code already used: {code}")

    await credit_wallet(session, principal_id, Decimal(redemption.amount))
    # The ledger row is what makes the credit auditable and idempotent: the
    # unique key (source_table, source_id, sku) means a code can only ever
    # produce one top-up entry, whatever retries happen above it.
    session.add(
        LedgerEntry(
            source_table=SOURCE_REDEMPTION,
            source_id=redemption.id,
            principal_id=principal_id,
            user_id=user_id,
            sku=SKU_WALLET_TOPUP,
            quantity=Decimal(redemption.amount),
            unit=UNIT_CURRENCY,
            unit_price=Decimal(1),
            amount=Decimal(redemption.amount),
            currency=redemption.currency,
            direction=LedgerDirection.CREDIT,
            settle_mode=SettleMode.REALTIME,
            status=LedgerStatus.SETTLED,
            settled_at=_utcnow(),
            occurred_at=_utcnow(),
        )
    )
    await session.commit()
    await session.refresh(redemption)
    # A top-up may clear a suspension; check rather than assume, since the
    # credit alone does not say whether the balance is now usable.
    await resume_wallet_if_funded(session, principal_id, commit=True)
    return redemption


# ---------------------------------------------------------------------------
# Settlement sweep
# ---------------------------------------------------------------------------


@dataclass
class SettlementReport:
    """What one settlement sweep collected."""

    mode: BillingMode = BillingMode.SHADOW
    principals: int = 0
    sessions: int = 0
    entries_settled: int = 0
    amount_settled: Decimal = Decimal(0)
    entries_unpaid: int = 0
    amount_unpaid: Decimal = Decimal(0)
    suspended: List[int] = field(default_factory=list)
    resumed: List[int] = field(default_factory=list)
    deferred_skipped: int = 0
    duration_ms: int = 0

    def summary(self) -> str:
        return (
            f"billing settlement sweep ({self.mode.value}) in {self.duration_ms}ms: "
            f"{self.entries_settled} entries / {self.amount_settled} collected "
            f"across {self.principals} principals ({self.sessions} sessions), "
            f"{self.entries_unpaid} entries / {self.amount_unpaid} unpaid, "
            f"suspended={self.suspended} resumed={self.resumed}, "
            f"{self.deferred_skipped} deferred left for invoicing"
        )


class BillingSettler:
    """Leader-only loop that debits wallets for rated realtime charges."""

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
            raise ValueError("billing settle interval must be positive")
        if self._batch_size <= 0:
            raise ValueError("billing settle batch size must be positive")
        self.last_report: Optional[SettlementReport] = None

    @property
    def mode(self) -> BillingMode:
        return self._mode

    async def start(self) -> None:
        if self._mode is not BillingMode.ENFORCE:
            logger.info(
                f"Billing settler idle (mode={self._mode.value}); the ledger is "
                "written but no wallet is debited."
            )
            return

        logger.info(
            f"Billing settler started (mode=enforce, interval={self._interval}s, "
            f"batch={self._batch_size})."
        )
        while True:
            try:
                report = await self.settle_once()
                self.last_report = report
                if report.entries_settled or report.entries_unpaid or report.suspended:
                    logger.info(report.summary())
            except Exception as e:
                # Contained for the same reason as the rater: the leader task
                # loop has no per-task supervision.
                logger.error(f"Billing settlement sweep failed: {e}", exc_info=True)
            await asyncio.sleep(self._interval)

    async def settle_once(self) -> SettlementReport:
        """One sweep. A no-op unless the mode is ``enforce``."""
        started = time.monotonic()
        report = SettlementReport(mode=self._mode)
        if self._mode is not BillingMode.ENFORCE:
            report.duration_ms = int((time.monotonic() - started) * 1000)
            return report

        async with async_session() as session:
            entries = await self._pending_realtime(session)
            report.deferred_skipped = await self._count_pending_deferred(session)

            by_principal: Dict[int, List[LedgerEntry]] = {}
            for entry in entries:
                # An entry with no payer cannot be collected; the rater only
                # writes one when attribution failed, and it stays PENDING so
                # the gap is still visible rather than silently dropped.
                if entry.principal_id is None:
                    report.entries_unpaid += 1
                    report.amount_unpaid += Decimal(entry.amount)
                    continue
                by_principal.setdefault(entry.principal_id, []).append(entry)

            report.principals = len(by_principal)
            for principal_id, group in by_principal.items():
                await self._settle_principal(session, principal_id, group, report)
                # Commit per principal: one org's failure (a lock timeout, a
                # deleted wallet) must not roll back what another org paid.
                await session.commit()

        report.duration_ms = int((time.monotonic() - started) * 1000)
        return report

    async def _pending_realtime(self, session: AsyncSession) -> List[LedgerEntry]:
        """Unsettled realtime charges, oldest first.

        Oldest-first is what makes partial settlement fair: when a balance runs
        out mid-sweep, the charges the org incurred first are the ones collected,
        and the arrears are the most recent — the opposite order would let a
        tenant keep consuming while old debt waits.
        """
        rows = (
            await session.exec(
                select(LedgerEntry)
                .where(
                    LedgerEntry.status == LedgerStatus.PENDING.value,
                    LedgerEntry.settle_mode == SettleMode.REALTIME.value,
                    LedgerEntry.deleted_at.is_(None),
                )
                .order_by(
                    LedgerEntry.principal_id,
                    LedgerEntry.occurred_at,
                    LedgerEntry.id,
                )
                .limit(self._batch_size)
            )
        ).all()
        return list(rows)

    async def _count_pending_deferred(self, session: AsyncSession) -> int:
        rows = (
            await session.exec(
                select(LedgerEntry.id).where(
                    LedgerEntry.status == LedgerStatus.PENDING.value,
                    LedgerEntry.settle_mode == SettleMode.DEFERRED.value,
                    LedgerEntry.deleted_at.is_(None),
                )
            )
        ).all()
        return len(rows)

    async def _settle_principal(
        self,
        session: AsyncSession,
        principal_id: int,
        entries: List[LedgerEntry],
        report: SettlementReport,
    ) -> None:
        wallet = await get_or_create_wallet(session, principal_id)
        billing_session = BillingSession(
            principal_id=principal_id,
            estimate=sum((Decimal(e.amount) for e in entries), Decimal(0)),
        )
        session.add(billing_session)
        await session.flush()  # need its id on the entries
        report.sessions += 1

        collected = Decimal(0)
        settled_count = 0
        suspended_now = False
        for entry in entries:
            amount = Decimal(entry.amount)
            if await debit_wallet(session, principal_id, amount):
                entry.status = LedgerStatus.SETTLED.value
                entry.settled_at = _utcnow()
                entry.billing_session_id = billing_session.id
                session.add(entry)
                collected += amount
                settled_count += 1
            else:
                # Balance ran out. Everything from here stays PENDING and the
                # wallet is suspended; a top-up resumes both.
                report.entries_unpaid += len(entries) - settled_count
                report.amount_unpaid += sum(
                    (Decimal(e.amount) for e in entries[settled_count:]), Decimal(0)
                )
                await suspend_wallet(session, principal_id)
                report.suspended.append(principal_id)
                suspended_now = True
                logger.warning(
                    f"billing: principal {principal_id} balance insufficient — "
                    f"settled {settled_count}/{len(entries)} entries, "
                    f"{len(entries) - settled_count} left pending, wallet suspended"
                )
                break

        billing_session.actual = collected
        billing_session.settled = True
        billing_session.settled_at = _utcnow()
        session.add(billing_session)

        report.entries_settled += settled_count
        report.amount_settled += collected
        # Only consider resuming when this sweep did not just suspend the
        # wallet: a suspension means arrears remain, and the outstanding-charges
        # check below would answer False anyway. The case this covers is a
        # wallet suspended by an earlier sweep whose arrears this sweep cleared,
        # possibly after a top-up landed out of band.
        if wallet.suspended and not suspended_now:
            if await resume_wallet_if_funded(session, principal_id):
                report.resumed.append(principal_id)

    async def refund_session(
        self, session: AsyncSession, billing_session_id: int
    ) -> Decimal:
        """Reverse a settlement, once.

        Idempotent through the same mutually exclusive flags new-api uses: a
        session that is already refunded, or that never settled, refunds zero.
        Money is returned before the entries are re-opened, and the ordering is
        deliberate — if the second step fails the operator has an org with
        credit and PENDING charges, which the next sweep collects; the reverse
        order would leave charges marked unpaid with no money returned.
        """
        billing_session = (
            await session.exec(
                select(BillingSession).where(BillingSession.id == billing_session_id)
            )
        ).first()
        if billing_session is None:
            raise InvalidException(
                message=f"Billing session {billing_session_id} not found"
            )
        if billing_session.refunded or not billing_session.settled:
            return Decimal(0)

        amount = Decimal(billing_session.actual or 0)
        if billing_session.principal_id is not None and amount > 0:
            await credit_wallet(session, billing_session.principal_id, amount)

        billing_session.refunded = True
        billing_session.settled_at = _utcnow()
        session.add(billing_session)

        entries = (
            await session.exec(
                select(LedgerEntry).where(
                    LedgerEntry.billing_session_id == billing_session_id
                )
            )
        ).all()
        for entry in entries:
            entry.status = LedgerStatus.PENDING.value
            entry.settled_at = None
            entry.billing_session_id = None
            session.add(entry)

        await session.commit()
        return amount
