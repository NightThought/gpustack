"""Price resolution and price-book invariants for the billing engine (WP2).

Two responsibilities live here so that neither the rater (WP3) nor the routes
(WP2 CRUD) has to know how a price is found or what makes one well-formed:

* **Resolution** — :func:`resolve_price` answers "what does one unit of this
  SKU cost for this model at this instant", picking the most specific live row
  and, among equally specific ones, the most recent window. Reads go through a
  short-lived in-process cache because rating scans thousands of usage rows per
  tick and the price book changes a few times a month.
* **Invariants** — :func:`assert_window_available` keeps one (sku, model,
  group) timeline unambiguous, and :func:`assert_identity_unchanged` keeps a
  row's identity stable so the ``price_book_version`` stamped on a ledger entry
  always points at a row that still means what it meant.

Cache coherence: the cache is per-process and is invalidated on every write
made *through this process*. Another server instance keeps serving its own copy
for up to ``PRICE_CACHE_TTL_SECONDS``, which is the same trade-off the runtime
metrics config cache makes — a price edit taking effect within minutes on every
instance, and immediately on the one that made it. Only the leader runs the
rater, so a stale follower cache cannot mis-price anything; it can only show an
admin the previous price for a moment.
"""

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Sequence, Tuple

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import InvalidException
from gpustack.schemas.billing import (
    MONEY_SCALE,
    PriceBookEntry,
    unit_for_sku,
)

logger = logging.getLogger(__name__)

# How long a process serves a cached copy of the price book. Writes invalidate
# locally; this bounds the staleness on the other instances.
PRICE_CACHE_TTL_SECONDS = 300

# Smallest money quantum a ledger amount is rounded to. Matches the Numeric
# scale of the money columns so the stored value is exactly the computed one.
_MONEY_QUANTUM = Decimal(1).scaleb(-MONEY_SCALE)

_cache: Optional[Tuple[Sequence[PriceBookEntry], float]] = None


def _utcnow() -> datetime:
    """Aware UTC now, the shape ``UTCDateTime`` hands back on read."""
    return datetime.now(timezone.utc)


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime to aware UTC before comparing it.

    ``UTCDateTime`` strips tzinfo on the way in and re-attaches UTC on the way
    out, so rows read from the database carry aware datetimes while values that
    never made a round trip — a caller's naive ``at=``, a JSON body parsed
    without an offset — are naive. Comparing the two raises ``TypeError`` rather
    than answering, and in the rater that would surface as a rating run that
    dies halfway through a batch. Every comparison here goes through this.
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def invalidate_price_cache() -> None:
    """Drop the cached price book so the next read goes to the database.

    Called by every write path (create / update / delete). Cheap and idempotent;
    a missed call would be a stale price for up to one TTL, which is why the
    routes call it unconditionally rather than only when a row changed.
    """
    global _cache
    _cache = None


async def _load_active_entries(session: AsyncSession) -> Sequence[PriceBookEntry]:
    """All live price rows, from cache when it is fresh."""
    global _cache
    now = time.monotonic()
    if _cache is not None:
        entries, fetched_at = _cache
        if now - fetched_at < PRICE_CACHE_TTL_SECONDS:
            return entries

    entries = await PriceBookEntry.all_by_fields(
        session, fields={"deleted_at": None, "is_active": True}
    )
    _cache = (entries, now)
    return entries


def windows_overlap(
    a_from: datetime,
    a_to: Optional[datetime],
    b_from: datetime,
    b_to: Optional[datetime],
) -> bool:
    """True when half-open windows ``[a_from, a_to)`` and ``[b_from, b_to)``
    intersect. ``None`` on the right edge means open-ended (still current).

    Both sides are normalized first: one edge typically comes from the database
    (aware) and the other from the request being validated (possibly naive).
    """
    a_from, a_to = _as_utc(a_from), _as_utc(a_to)
    b_from, b_to = _as_utc(b_from), _as_utc(b_to)
    if a_to is not None and a_to <= b_from:
        return False
    if b_to is not None and b_to <= a_from:
        return False
    return True


async def candidates_for(
    session: AsyncSession,
    *,
    sku: str,
    model_name: Optional[str] = None,
    group_name: Optional[str] = None,
) -> List[PriceBookEntry]:
    """Every live row on one (sku, model, group) timeline — its price history.

    Reads the database rather than the cache: this backs the overlap check on
    write, where a stale snapshot could admit a window that conflicts with a
    price another admin just added.
    """
    rows = await PriceBookEntry.all_by_fields(
        session,
        fields={
            "deleted_at": None,
            "sku": sku,
            "model_name": model_name,
            "group_name": group_name,
        },
    )
    return list(rows)


async def assert_window_available(
    session: AsyncSession,
    *,
    sku: str,
    model_name: Optional[str],
    group_name: Optional[str],
    effective_from: datetime,
    effective_to: Optional[datetime],
    exclude_id: Optional[int] = None,
) -> None:
    """Reject a window that overlaps another on the same timeline.

    Two overlapping prices for one SKU make rating ambiguous — the answer would
    depend on row order, which no one reading a bill can see. ``exclude_id``
    lets an update re-check its own row without colliding with itself.
    """
    for row in await candidates_for(
        session, sku=sku, model_name=model_name, group_name=group_name
    ):
        if exclude_id is not None and row.id == exclude_id:
            continue
        if windows_overlap(
            effective_from, effective_to, row.effective_from, row.effective_to
        ):
            existing_to = row.effective_to or "open-ended"
            raise InvalidException(
                message=(
                    f"Price window [{effective_from}, {effective_to or 'open-ended'}) "
                    f"overlaps existing price id={row.id} "
                    f"[{row.effective_from}, {existing_to}) for sku '{sku}'"
                    + (f" model '{model_name}'" if model_name else "")
                    + ". Close the existing window or move this one."
                )
            )


# Identity fields: changing one would make the row a different price, so the
# route refuses and asks for a new row instead. ``price`` / ``per_quantity`` /
# ``effective_to`` / ``is_active`` / ``group_name`` are the editable surface.
_IDENTITY_FIELDS = ("sku", "model_name", "unit", "currency", "effective_from")


def assert_identity_unchanged(
    existing: PriceBookEntry, incoming: PriceBookEntry
) -> None:
    """Guard the fields a ledger snapshot's meaning depends on.

    Datetimes are normalized before comparing: the stored value comes back from
    ``UTCDateTime`` aware, while the same instant arriving in a request body is
    naive, and a raw ``!=`` would report every update as an identity change —
    which reads as "prices can never be edited" rather than as a timezone bug.
    """
    changed = []
    for name in _IDENTITY_FIELDS:
        current, proposed = getattr(existing, name), getattr(incoming, name)
        if isinstance(current, datetime) or isinstance(proposed, datetime):
            current, proposed = _as_utc(current), _as_utc(proposed)
        if current != proposed:
            changed.append(name)
    if changed:
        raise InvalidException(
            message=(
                f"Cannot change {', '.join(changed)} on an existing price "
                "(ledger entries snapshot this row's identity). Create a new "
                "price with the intended window instead."
            )
        )


def _specificity(entry: PriceBookEntry) -> Tuple[bool, bool]:
    """More specific wins: a model+group price beats a model price beats a
    family default."""
    return entry.model_name is not None, entry.group_name is not None


def _scope_matches(row_value: Optional[str], requested: Optional[str]) -> bool:
    """Whether a row's scope dimension applies to the requested one.

    A row scoped to a specific model/group only ever applies to that exact
    value; an unscoped row (``None``) is the family default and applies to any
    requested value. The asymmetry matters: a caller with no group must NOT be
    given a group's discounted price, and a resource SKU (no model) must not
    pick up some model's rate. Getting this backwards silently bills the wrong
    party at the wrong price.
    """
    if row_value is None:
        return True
    return row_value == requested


def pick_price(
    entries: Sequence[PriceBookEntry],
    *,
    sku: str,
    at: datetime,
    model_name: Optional[str] = None,
    group_name: Optional[str] = None,
) -> Optional[PriceBookEntry]:
    """Choose the price that applies to ``(sku, model_name, group_name)`` at
    ``at``, out of rows already known to be live.

    Specificity first, then recency: among rows of equal specificity the one
    whose window started latest is the current price, which is what makes
    "add a new window starting now" supersede an open-ended row without having
    to edit it. ``id`` is the final tie-break so the answer is deterministic
    even if two rows somehow claim the same window (the overlap check forbids
    it, but a bill that depends on row order is worse than either candidate).
    Returns ``None`` when nothing covers ``at`` — the rater treats that as
    "unpriced, do not guess" and surfaces it rather than billing zero.
    """
    at = _as_utc(at)
    matches = [
        entry
        for entry in entries
        if entry.sku == sku
        and _as_utc(entry.effective_from) <= at
        and (entry.effective_to is None or at < _as_utc(entry.effective_to))
        and _scope_matches(entry.model_name, model_name)
        and _scope_matches(entry.group_name, group_name)
    ]
    if not matches:
        return None

    def sort_key(entry: PriceBookEntry):
        # Exact model/group match beats a family default; among equals the
        # newest window wins; id last, purely for determinism.
        model_exact = entry.model_name is not None and entry.model_name == model_name
        group_exact = entry.group_name is not None and entry.group_name == group_name
        return (
            model_exact,
            group_exact,
            *_specificity(entry),
            _as_utc(entry.effective_from),
            entry.id or 0,
        )

    return max(matches, key=sort_key)


async def resolve_price(
    session: AsyncSession,
    *,
    sku: str,
    model_name: Optional[str] = None,
    group_name: Optional[str] = None,
    at: Optional[datetime] = None,
) -> Optional[PriceBookEntry]:
    """The price row that applies to one SKU at one instant (or now).

    ``None`` means unpriced: the caller must not invent a rate. Rating a row
    with no price is an operational error (a new model went live before anyone
    priced it), and billing it at zero is the kind of mistake a reconciliation
    only finds months later.
    """
    # An unknown SKU is a programming error, not an empty price book: fail here
    # so the rater's caller sees it instead of silently rating nothing.
    unit_for_sku(sku)
    entries = await _load_active_entries(session)
    return pick_price(
        entries,
        sku=sku,
        at=at or _utcnow(),
        model_name=model_name,
        group_name=group_name,
    )


def compute_amount(entry: PriceBookEntry, quantity: Decimal) -> Decimal:
    """What ``quantity`` units cost under ``entry``.

    Multiplies before dividing so the intermediate keeps full precision, then
    rounds once, half away from zero, to the money scale the ledger column
    stores — one rounding step means the sum of per-entry amounts equals the
    amount a single re-computation would produce, which is what reconciliation
    compares.
    """
    if entry.per_quantity <= 0:
        raise ValueError(f"price book entry {entry.id} has non-positive per_quantity")
    amount = (Decimal(quantity) * entry.price) / entry.per_quantity
    return amount.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)
