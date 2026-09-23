"""Quota ceilings and the pre-check that enforces them (WP5.4).

A quota answers a different question from a wallet. The wallet says whether the
org can pay; a quota says how much of an allowance this subject may consume in
this window, whether or not money remains. Both refuse service, and the status
codes are deliberately different so a client can tell what to do:

* **429** — the window's allowance is spent. Retry after the window rolls over,
  or ask the operator to raise the ceiling. Nothing to pay.
* **402** — the wallet ran dry (``billing_enforcement``). Top up.

Conflating them makes one of the two remediations unreachable: a client told 402
for a daily token cap will top up and still be refused, and one told 429 for an
empty wallet will back off forever.

Three properties the design has to keep
---------------------------------------

**The hot path issues no query for a caller with no quotas.** Quota definitions
are cached in-process (``QUOTA_CACHE_TTL_SECONDS``, invalidated by every write),
so matching is a list scan. Only a caller that actually has a ceiling costs a
read, and only to fetch ``used`` — the one field that changes between requests.
A per-request aggregate over the ledger would put a full scan on the
authorization path of every inference call.

**A counter cannot be reset by the tenant.** ``used`` / ``window_start`` are the
rater's state and are absent from the editable surface (``QuotaUpdate``). The
window rolls by comparison rather than by a cron job, so a daily limit needs no
scheduler and survives a server that was down at midnight.

**A quota created mid-window counts the usage already in that window.** The
first read of a row whose ``window_start`` is NULL or older than the current
window recomputes ``used`` from the ledger — which is also how a rollover is
applied, so the two cases share one code path and one conditional UPDATE.

Concurrency: the rater advances counters with ``used = used + :delta`` guarded by
``window_start = :ws``, and the rollover writes under
``window_start IS NULL OR window_start < :ws``. The two guards are mutually
exclusive, so a sweep racing a rollover loses nothing: whichever lands second
either adds to the rolled row or finds it already rolled and re-reads it.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import func, or_, update
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import InvalidException, TooManyRequestsException
from gpustack.schemas.billing import (
    SKU_TOKEN_CACHED,
    SKU_TOKEN_COMPLETION,
    SKU_TOKEN_PROMPT,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    Quota,
    QuotaLimitType,
    QuotaScope,
)
from gpustack.server.billing_pricing import as_utc

logger = logging.getLogger(__name__)

# How long a process serves a cached copy of the quota table. Writes invalidate
# locally; the TTL bounds staleness elsewhere. Same trade-off as the price book.
QUOTA_CACHE_TTL_SECONDS = 300

# The SKUs whose quantity is a token count. A daily-token ceiling must not be
# advanced by gpu-hours or GiB-hours, which share the quantity column's shape
# but not its meaning.
TOKEN_SKUS = frozenset({SKU_TOKEN_PROMPT, SKU_TOKEN_COMPLETION, SKU_TOKEN_CACHED})

_DAILY_TYPES = frozenset({QuotaLimitType.DAILY_TOKENS, QuotaLimitType.DAILY_AMOUNT})

_cache: Optional[Tuple[Sequence["QuotaSpec"], float]] = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class QuotaSpec:
    """The immutable half of a quota row — everything except the counter.

    Cached as a value object rather than as an ORM entity: a detached instance
    carries whatever ``used`` it happened to have when it was loaded, and a
    limit check reading a stale counter is the one staleness that matters here.
    The counter is always read fresh, by id, at check time.
    """

    id: int
    scope: QuotaScope
    principal_id: Optional[int]
    user_id: Optional[int]
    api_key_id: Optional[int]
    model_name: Optional[str]
    limit_type: QuotaLimitType
    limit_value: Decimal

    @property
    def subject_id(self) -> Optional[int]:
        return {
            QuotaScope.ORGANIZATION: self.principal_id,
            QuotaScope.USER: self.user_id,
            QuotaScope.API_KEY: self.api_key_id,
        }[self.scope]

    @property
    def is_token_limit(self) -> bool:
        return self.limit_type == QuotaLimitType.DAILY_TOKENS

    def describe_window(self, window_start: datetime) -> str:
        if self.limit_type == QuotaLimitType.MONTHLY_AMOUNT:
            return f"month of {window_start:%Y-%m}"
        return f"day of {window_start:%Y-%m-%d}"


def invalidate_quota_cache() -> None:
    """Drop the cached quota table so the next read goes to the database.

    Called by every write path. Cheap and idempotent; a missed call costs one
    TTL of staleness, which for a ceiling means admitting slightly more than the
    configured limit for a few minutes rather than refusing traffic.
    """
    global _cache
    _cache = None


async def enabled_quota_specs(session: AsyncSession) -> Sequence[QuotaSpec]:
    """Every enabled quota, as value objects, from cache when possible."""
    global _cache
    now = time.monotonic()
    if _cache is not None and (now - _cache[1]) < QUOTA_CACHE_TTL_SECONDS:
        return _cache[0]

    rows = (await session.exec(select(Quota).where(Quota.enabled.is_(True)))).all()
    specs = tuple(
        QuotaSpec(
            id=row.id,
            scope=QuotaScope(row.scope),
            principal_id=row.principal_id,
            user_id=row.user_id,
            api_key_id=row.api_key_id,
            model_name=row.model_name,
            limit_type=QuotaLimitType(row.limit_type),
            limit_value=row.limit_value,
        )
        for row in rows
    )
    _cache = (specs, now)
    return specs


def specs_for_caller(
    specs: Sequence[QuotaSpec],
    *,
    api_key_id: Optional[int] = None,
    user_id: Optional[int] = None,
    principal_id: Optional[int] = None,
    model_name: Optional[str] = None,
) -> List[QuotaSpec]:
    """Which cached quotas bind this caller and model. Pure: no I/O.

    A row's ``model_name`` of NULL means "every model", so it matches any
    request; a row naming a model matches only that model. The subject is
    matched by the column the row's own scope names, which is why the schema
    rejects a row whose subject columns disagree with its scope — such a row
    would match nothing and limit nothing.
    """
    subject_by_scope = {
        QuotaScope.ORGANIZATION: principal_id,
        QuotaScope.USER: user_id,
        QuotaScope.API_KEY: api_key_id,
    }
    matched = []
    for spec in specs:
        caller_id = subject_by_scope.get(spec.scope)
        if caller_id is None or spec.subject_id != caller_id:
            continue
        if spec.model_name is not None and spec.model_name != model_name:
            continue
        matched.append(spec)
    return matched


def window_start_for(
    limit_type: QuotaLimitType, at: Optional[datetime] = None
) -> datetime:
    """Start of the window ``limit_type`` counts, in UTC.

    Calendar windows rather than rolling ones: "tokens per day" means per UTC
    day to an operator reading a report, and a rolling window cannot be
    recomputed from the ledger after the fact — which is what makes the
    self-healing rollback below possible at all.
    """
    moment = as_utc(at) or _utcnow()
    if limit_type in _DAILY_TYPES:
        return datetime(moment.year, moment.month, moment.day, tzinfo=timezone.utc)
    return datetime(moment.year, moment.month, 1, tzinfo=timezone.utc)


def _subject_predicate(spec: QuotaSpec):
    """The ledger filter that selects this quota's subject."""
    if spec.scope == QuotaScope.API_KEY:
        return LedgerEntry.api_key_id == spec.api_key_id
    if spec.scope == QuotaScope.USER:
        return LedgerEntry.user_id == spec.user_id
    return LedgerEntry.principal_id == spec.principal_id


async def usage_in_window(
    session: AsyncSession, *, spec: QuotaSpec, window_start: datetime
) -> Decimal:
    """What this subject has consumed since ``window_start``, from the ledger.

    The ledger is the source of truth rather than the counter: recomputing from
    it is what lets a window be rolled, and a quota be created mid-window,
    without either losing or double-counting usage.

    Only DEBIT rows count. A wallet top-up writes a CREDIT row, and letting one
    raise a spend ceiling would mean an org could buy itself allowance it was
    never granted.
    """
    metric = (
        func.sum(LedgerEntry.quantity)
        if spec.is_token_limit
        else func.sum(LedgerEntry.amount)
    )
    statement = select(metric).where(
        LedgerEntry.deleted_at.is_(None),
        LedgerEntry.direction == LedgerDirection.DEBIT.value,
        LedgerEntry.status != LedgerStatus.VOID.value,
        LedgerEntry.occurred_at >= window_start,
        _subject_predicate(spec),
    )
    if spec.is_token_limit:
        statement = statement.where(LedgerEntry.sku.in_(list(TOKEN_SKUS)))
    if spec.model_name is not None:
        statement = statement.where(LedgerEntry.model_name == spec.model_name)

    total = (await session.exec(statement)).first()
    return Decimal(total) if total is not None else Decimal(0)


async def _roll_window(
    session: AsyncSession, spec: QuotaSpec, window_start: datetime, used: Decimal
) -> bool:
    """Install a freshly computed window, unless another writer already did.

    The guard is what makes this safe to attempt from both the request path and
    the leader's sweep: exactly one of them wins, and the loser re-reads.
    """
    result = await session.exec(
        update(Quota)
        .where(
            Quota.id == spec.id,
            or_(Quota.window_start.is_(None), Quota.window_start < window_start),
        )
        .values(window_start=window_start, used=used)
    )
    return result.rowcount == 1


async def window_used(
    session: AsyncSession, spec: QuotaSpec, now: Optional[datetime] = None
) -> Decimal:
    """How much of ``spec``'s current window is consumed, rolling it if needed.

    Reads the counter with the caller's session; performs the (rare) rollover
    write on a session of its own, because a request handler's session is not
    this module's to commit and an uncommitted rollover would be recomputed by
    every following request.
    """
    moment = as_utc(now) or _utcnow()
    window_start = window_start_for(spec.limit_type, moment)

    row = (await session.exec(select(Quota).where(Quota.id == spec.id))).first()
    if row is None or not row.enabled:
        # Deleted or disabled since the cache was filled. Reading a ceiling that
        # no longer exists would refuse traffic on an operator's revoked row.
        return Decimal(0)

    current = as_utc(row.window_start)
    if current is not None and current >= window_start:
        return Decimal(row.used)

    used = await usage_in_window(session, spec=spec, window_start=window_start)
    from gpustack.server.db import async_session

    async with async_session() as write_session:
        rolled = await _roll_window(write_session, spec, window_start, used)
        await write_session.commit()
    if not rolled:
        # Someone rolled it between the read and the write; their value is the
        # one that stands, and it was computed from the same ledger.
        row = (await session.exec(select(Quota).where(Quota.id == spec.id))).first()
        return Decimal(row.used) if row is not None else used
    return used


async def check_quota(
    session: AsyncSession,
    *,
    api_key_id: Optional[int] = None,
    user_id: Optional[int] = None,
    principal_id: Optional[int] = None,
    model_name: Optional[str] = None,
    openai_shaped: bool = False,
) -> None:
    """Refuse with 429 when a ceiling binding this caller is already met.

    Called from both request paths, next to the suspension check, so a subject
    is limited identically whichever way its traffic arrives.

    ``>=`` rather than ``>``: a ceiling of N means N is the last unit admitted,
    and the request that would take usage past it is the one refused. Reading it
    the other way would let every subject overshoot by one request, which on a
    monthly amount ceiling is exactly the overshoot the ceiling exists to stop.

    A failure to read billing state is logged and let through. Refusing inference
    because a counter could not be read turns a billing problem into an outage,
    and the exposure is bounded by the window — the same posture
    ``billing_enforcement.inherited_suspension`` takes.
    """
    try:
        specs = specs_for_caller(
            await enabled_quota_specs(session),
            api_key_id=api_key_id,
            user_id=user_id,
            principal_id=principal_id,
            model_name=model_name,
        )
    except Exception as e:
        logger.warning(
            f"billing: could not load quotas for a request ({e}); allowing it"
        )
        return
    if not specs:
        return

    now = _utcnow()
    for spec in specs:
        try:
            used = await window_used(session, spec, now)
        except Exception as e:
            logger.warning(
                f"billing: could not evaluate quota {spec.id} "
                f"({spec.limit_type.value}) for this request ({e}); allowing it"
            )
            continue
        if used >= spec.limit_value:
            window_start = window_start_for(spec.limit_type, now)
            unit = "tokens" if spec.is_token_limit else "CNY"
            logger.info(
                f"billing: quota {spec.id} ({spec.scope.value} "
                f"{spec.subject_id}, model={spec.model_name or '*'}) exhausted — "
                f"{used} of {spec.limit_value} {unit} in "
                f"{spec.describe_window(window_start)}"
            )
            raise TooManyRequestsException(
                message=(
                    f"{spec.limit_type.value} quota exhausted for "
                    f"{spec.scope.value} {spec.subject_id}"
                    f"{f' on model {model_name}' if model_name else ''}: "
                    f"{used} of {spec.limit_value} {unit} used in "
                    f"{spec.describe_window(window_start)}"
                ),
                is_openai_exception=openai_shaped,
            )


async def apply_usage(
    session: AsyncSession,
    entries: Sequence[LedgerEntry],
    *,
    now: Optional[datetime] = None,
) -> int:
    """Advance quota counters for ledger entries the rater just wrote.

    Called by the rater with the entries it actually inserted or promoted this
    sweep — never with the whole table, and never with rows it skipped, because
    a counter advanced twice for one request is a ceiling that bites early.

    Deltas land with ``used = used + :delta`` under ``window_start = :ws``,
    grouped by the window each entry's ``occurred_at`` falls in, so a backfill
    that spans a midnight does not pour yesterday's tokens into today's counter.
    When a window has not been established yet the counter is recomputed from
    the ledger instead of incremented — and *not* then incremented, because the
    entries being applied are already visible to that same transaction's SUM.
    That is a precondition, not a coincidence: **the entries passed in must
    already be added to ``session``**, which is what the rater does when it
    stages them. Passing rows the session has not seen would leave the
    recomputed counter short by exactly those rows, and the next sweep would not
    notice — it skips sources that already have ledger entries.

    Writes use the caller's session: the rater commits the sweep, and a counter
    that moved without its ledger rows would be unreproducible.
    """
    if not entries:
        return 0
    try:
        specs = await enabled_quota_specs(session)
    except Exception as e:
        logger.warning(f"billing: could not load quotas to advance ({e}); skipping")
        return 0
    if not specs:
        return 0

    moment = as_utc(now) or _utcnow()
    # (spec id, window start) -> delta
    deltas: Dict[Tuple[int, datetime], Decimal] = {}
    for entry in entries:
        if entry.status == LedgerStatus.VOID.value:
            continue
        if (entry.direction or LedgerDirection.DEBIT.value) != (
            LedgerDirection.DEBIT.value
        ):
            continue
        for spec in specs_for_caller(
            specs,
            api_key_id=entry.api_key_id,
            user_id=entry.user_id,
            principal_id=entry.principal_id,
            model_name=entry.model_name,
        ):
            occurred = as_utc(entry.occurred_at) or moment
            window_start = window_start_for(spec.limit_type, occurred)
            key = (spec.id, window_start)
            amount = entry.quantity if spec.is_token_limit else entry.amount
            deltas[key] = deltas.get(key, Decimal(0)) + Decimal(amount or 0)

    if not deltas:
        return 0

    spec_by_id = {spec.id: spec for spec in specs}
    advanced = 0
    for (spec_id, window_start), delta in sorted(deltas.items()):
        spec = spec_by_id[spec_id]
        result = await session.exec(
            update(Quota)
            .where(Quota.id == spec_id, Quota.window_start == window_start)
            .values(used=Quota.used + delta, updated_at=moment)
        )
        if result.rowcount == 1:
            advanced += 1
            continue
        # No established window to add to: compute it from the ledger, which
        # already sees these entries in this transaction.
        used = await usage_in_window(session, spec=spec, window_start=window_start)
        if await _roll_window(session, spec, window_start, used):
            advanced += 1
    return advanced


async def assert_no_duplicate(
    session: AsyncSession,
    *,
    scope: QuotaScope,
    principal_id: Optional[int] = None,
    user_id: Optional[int] = None,
    api_key_id: Optional[int] = None,
    model_name: Optional[str] = None,
    limit_type: QuotaLimitType,
    exclude_id: Optional[int] = None,
) -> None:
    """Reject a second ceiling for one (subject, model, limit type).

    ``uq_quota_scope_limit`` cannot do this on its own: SQL treats NULLs as
    distinct, and every subject column but the scope's own one is NULL, so an
    API_KEY quota on all models would be insertable twice and the two rows would
    each count usage against the same window — the stricter limit silently
    overridden by whichever the gate reached first.
    """
    statement = select(Quota.id).where(
        Quota.deleted_at.is_(None),
        Quota.scope == scope.value,
        Quota.limit_type == limit_type.value,
    )
    for column, value in (
        (Quota.principal_id, principal_id),
        (Quota.user_id, user_id),
        (Quota.api_key_id, api_key_id),
        (Quota.model_name, model_name),
    ):
        statement = statement.where(
            column.is_(None) if value is None else column == value
        )
    if exclude_id is not None:
        statement = statement.where(Quota.id != exclude_id)

    if (await session.exec(statement)).first() is not None:
        raise InvalidException(
            message=(
                f"a {str(limit_type.value)} quota already exists for this "
                f"{str(scope.value)} and model "
                f"({model_name or 'all models'}); update that row instead"
            )
        )
