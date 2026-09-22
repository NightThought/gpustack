"""Price resolution and price-book invariants (WP2).

Runs against a real (in-memory) database rather than stubs, because what is
being pinned here is the query behaviour — which rows a resolver considers, and
whether a write-time check sees rows another writer just added.

The properties that matter, and why:

* **Specificity then recency** — a model price beats a family default, and a
  newer window beats an older open-ended one, so "add a price effective now"
  supersedes without editing history.
* **No overlap on one timeline** — two rows covering the same instant would make
  a bill depend on row order, which no one reading it can see.
* **Unpriced is not zero** — a missing price returns ``None`` and the rater must
  refuse to guess.
* **Cache invalidation** — a write through the API is visible immediately in the
  same process, and the TTL bounds staleness elsewhere.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import InvalidException
from gpustack.schemas.billing import (
    PriceBookEntry,
    PriceBookEntryCreate,
    SKU_GPU_HOUR_PREFIX,
    SKU_STORAGE_GB_HOUR,
    SKU_TOKEN_CACHED,
    SKU_TOKEN_PROMPT,
    UNIT_GPU_HOURS,
    UNIT_TOKENS,
)
from gpustack.server.billing_pricing import (
    assert_identity_unchanged,
    assert_window_available,
    compute_amount,
    invalidate_price_cache,
    pick_price,
    resolve_price,
    windows_overlap,
)

NOW = datetime(2026, 9, 22, 12, 0, 0)
T0 = datetime(2026, 9, 1, 0, 0, 0)
T1 = datetime(2026, 10, 1, 0, 0, 0)
T2 = datetime(2026, 11, 1, 0, 0, 0)


def _price(
    id_,
    sku=SKU_TOKEN_PROMPT,
    model_name="Qwen3-8B",
    price="0.002",
    per_quantity="1000",
    unit=UNIT_TOKENS,
    effective_from=T0,
    effective_to=None,
    group_name=None,
    is_active=True,
    deleted_at=None,
):
    return PriceBookEntry(
        id=id_,
        sku=sku,
        model_name=model_name,
        group_name=group_name,
        unit=unit,
        price=Decimal(price),
        per_quantity=Decimal(per_quantity),
        currency="CNY",
        version=1,
        effective_from=effective_from,
        effective_to=effective_to,
        is_active=is_active,
        created_at=NOW,
        updated_at=NOW,
        deleted_at=deleted_at,
    )


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(PriceBookEntry.__table__.create)
    # The price cache is module state: a row one test wrote must not be served
    # to the next one.
    invalidate_price_cache()
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    invalidate_price_cache()


# ---------------------------------------------------------------------------
# Window arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a_from,a_to,b_from,b_to,expected",
    [
        (T0, T1, T1, T2, False),  # adjacent half-open windows do not overlap
        (T0, T1, T0, T2, True),  # contained
        (T0, T2, T1, None, True),  # open-ended tail overlaps anything later
        (T0, None, T1, T2, True),  # open-ended existing overlaps a new window
        (T0, None, T1, None, True),  # two open-ended rows on one timeline
        (T1, T2, T0, T1, False),  # reversed adjacency
        (T0, T1, T2, None, False),  # disjoint
    ],
)
def test_windows_overlap(a_from, a_to, b_from, b_to, expected):
    assert windows_overlap(a_from, a_to, b_from, b_to) is expected


@pytest.mark.asyncio
async def test_overlap_rejected(session):
    session.add(_price(1, effective_from=T0, effective_to=None))
    await session.commit()

    # ``match=`` reads str(exc), and HTTPException keeps its text on .message
    # rather than passing it to Exception — assert on the attribute.
    with pytest.raises(InvalidException) as exc_info:
        await assert_window_available(
            session,
            sku=SKU_TOKEN_PROMPT,
            model_name="Qwen3-8B",
            group_name=None,
            effective_from=T1,
            effective_to=None,
        )
    assert "overlaps existing price id=1" in exc_info.value.message


@pytest.mark.asyncio
async def test_adjacent_window_allowed(session):
    session.add(_price(1, effective_from=T0, effective_to=T1))
    await session.commit()

    # Closing at T1 and starting at T1 is the intended "new price from now on".
    await assert_window_available(
        session,
        sku=SKU_TOKEN_PROMPT,
        model_name="Qwen3-8B",
        group_name=None,
        effective_from=T1,
        effective_to=None,
    )


@pytest.mark.asyncio
async def test_update_excludes_itself(session):
    session.add(_price(1, effective_from=T0, effective_to=None))
    await session.commit()

    await assert_window_available(
        session,
        sku=SKU_TOKEN_PROMPT,
        model_name="Qwen3-8B",
        group_name=None,
        effective_from=T0,
        effective_to=None,
        exclude_id=1,
    )


@pytest.mark.asyncio
async def test_different_model_does_not_collide(session):
    session.add(_price(1, model_name="Qwen3-8B", effective_from=T0))
    await session.commit()

    await assert_window_available(
        session,
        sku=SKU_TOKEN_PROMPT,
        model_name="Qwen3-30B",
        group_name=None,
        effective_from=T0,
        effective_to=None,
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_exact_model_beats_family_default(session):
    session.add(_price(1, model_name=None, price="0.010"))  # family default
    session.add(_price(2, model_name="Qwen3-8B", price="0.002"))
    await session.commit()

    entry = await resolve_price(session, sku=SKU_TOKEN_PROMPT, model_name="Qwen3-8B")
    assert entry.id == 2


@pytest.mark.asyncio
async def test_resolve_newest_window_wins(session):
    session.add(_price(1, effective_from=T0, effective_to=None, price="0.002"))
    session.add(_price(2, effective_from=T1, effective_to=None, price="0.001"))
    await session.commit()

    before = await resolve_price(
        session, sku=SKU_TOKEN_PROMPT, model_name="Qwen3-8B", at=T0 + timedelta(days=1)
    )
    after = await resolve_price(
        session, sku=SKU_TOKEN_PROMPT, model_name="Qwen3-8B", at=T1 + timedelta(days=1)
    )
    assert before.id == 1
    assert after.id == 2


@pytest.mark.asyncio
async def test_resolve_group_specific_beats_plain_model(session):
    session.add(_price(1, model_name="Qwen3-8B", price="0.002"))
    session.add(
        _price(2, model_name="Qwen3-8B", group_name="vip", price="0.0016")
    )
    await session.commit()

    vip = await resolve_price(
        session, sku=SKU_TOKEN_PROMPT, model_name="Qwen3-8B", group_name="vip"
    )
    plain = await resolve_price(
        session, sku=SKU_TOKEN_PROMPT, model_name="Qwen3-8B"
    )
    assert vip.id == 2
    assert plain.id == 1


@pytest.mark.asyncio
async def test_resolve_unpriced_returns_none(session):
    """No price is not a zero price: the rater must refuse to guess."""
    session.add(_price(1, model_name="Qwen3-8B"))
    await session.commit()

    assert (
        await resolve_price(session, sku=SKU_TOKEN_PROMPT, model_name="Unknown-Model")
        is None
    )


@pytest.mark.asyncio
async def test_resolve_ignores_inactive_and_deleted_rows(session):
    session.add(_price(1, is_active=False))
    session.add(_price(2, deleted_at=NOW))
    await session.commit()

    assert (
        await resolve_price(session, sku=SKU_TOKEN_PROMPT, model_name="Qwen3-8B")
        is None
    )


@pytest.mark.asyncio
async def test_resolve_outside_window_returns_none(session):
    session.add(_price(1, effective_from=T0, effective_to=T1))
    await session.commit()

    assert (
        await resolve_price(
            session, sku=SKU_TOKEN_PROMPT, model_name="Qwen3-8B", at=T2
        )
        is None
    )


@pytest.mark.asyncio
async def test_resolve_unknown_sku_raises(session):
    with pytest.raises(ValueError, match="unknown billing sku"):
        await resolve_price(session, sku="model.token.mystery")


@pytest.mark.asyncio
async def test_cache_invalidation_makes_writes_visible(session):
    session.add(_price(1, price="0.002"))
    await session.commit()
    first = await resolve_price(session, sku=SKU_TOKEN_PROMPT, model_name="Qwen3-8B")
    assert first.price == Decimal("0.002")

    # A write that does NOT invalidate is served stale — the documented TTL
    # bound, and the reason every route calls invalidate_price_cache(). The new
    # row starts a later window, which is how a real price change lands (two
    # rows claiming one window would be rejected by the overlap check).
    session.add(_price(2, price="0.001", effective_from=T1))
    await session.commit()
    stale = await resolve_price(
        session, sku=SKU_TOKEN_PROMPT, model_name="Qwen3-8B", at=T1 + timedelta(days=1)
    )
    assert stale.id == 1

    invalidate_price_cache()
    fresh = await resolve_price(
        session, sku=SKU_TOKEN_PROMPT, model_name="Qwen3-8B", at=T1 + timedelta(days=1)
    )
    assert fresh.id == 2


@pytest.mark.asyncio
async def test_caller_without_group_does_not_get_group_price(session):
    """A group discount must not leak to callers outside the group.

    Regression: the scope filter used to treat "caller named no group" as
    "any row matches", which billed ungrouped traffic at a group's rate.
    """
    session.add(_price(1, model_name="Qwen3-8B", price="0.002"))
    session.add(_price(2, model_name="Qwen3-8B", group_name="vip", price="0.0016"))
    await session.commit()

    ungrouped = await resolve_price(
        session, sku=SKU_TOKEN_PROMPT, model_name="Qwen3-8B"
    )
    assert ungrouped.id == 1
    other_group = await resolve_price(
        session, sku=SKU_TOKEN_PROMPT, model_name="Qwen3-8B", group_name="reseller"
    )
    assert other_group.id == 1


@pytest.mark.asyncio
async def test_resource_sku_does_not_inherit_a_model_price(session):
    """A caller with no model must not pick up some model's rate."""
    session.add(_price(1, model_name="Qwen3-8B", price="0.002"))
    await session.commit()

    assert (
        await resolve_price(session, sku=SKU_TOKEN_PROMPT, model_name=None) is None
    )


@pytest.mark.asyncio
async def test_resolve_accepts_a_naive_instant(session):
    """Naive and aware datetimes must be comparable.

    Regression: rows read back from the database carry aware datetimes
    (``UTCDateTime`` re-attaches UTC) while a caller's ``at=`` or ``_utcnow()``
    may be naive; comparing the two raised TypeError, which in the rater would
    look like a rating run dying mid-batch.
    """
    session.add(_price(1, effective_from=T0, effective_to=T1))
    await session.commit()

    naive_at = T0 + timedelta(days=1)  # no tzinfo
    aware_at = naive_at.replace(tzinfo=timezone.utc)
    assert (await resolve_price(session, sku=SKU_TOKEN_PROMPT,
                               model_name="Qwen3-8B", at=naive_at)).id == 1
    assert (await resolve_price(session, sku=SKU_TOKEN_PROMPT,
                               model_name="Qwen3-8B", at=aware_at)).id == 1


def test_windows_overlap_mixes_naive_and_aware():
    aware_from = T0.replace(tzinfo=timezone.utc)
    assert windows_overlap(aware_from, None, T1, None) is True
    assert windows_overlap(T0, T1, aware_from, None) is True


def test_identical_windows_resolve_deterministically():
    """Two rows claiming one window must still give one stable answer.

    The overlap check forbids this shape; determinism is defence in depth, so a
    bill never depends on the order rows came back in.
    """
    rows = [_price(1, price="0.002"), _price(2, price="0.001")]
    first = pick_price(rows, sku=SKU_TOKEN_PROMPT, at=T0, model_name="Qwen3-8B")
    second = pick_price(
        list(reversed(rows)), sku=SKU_TOKEN_PROMPT, at=T0, model_name="Qwen3-8B"
    )
    assert first.id == second.id == 2


def test_pick_price_is_pure():
    """The resolver's core is a pure function of the rows it is given."""
    rows = [
        _price(1, model_name=None, price="0.010"),
        _price(2, model_name="Qwen3-8B", price="0.002"),
    ]
    chosen = pick_price(rows, sku=SKU_TOKEN_PROMPT, at=T0, model_name="Qwen3-8B")
    assert chosen.id == 2
    assert (
        pick_price(rows, sku=SKU_TOKEN_CACHED, at=T0, model_name="Qwen3-8B") is None
    )


# ---------------------------------------------------------------------------
# Amount computation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "price,per_quantity,quantity,expected",
    [
        ("0.002", "1000", "1000", "0.00200000"),  # exactly one priced unit
        ("0.002", "1000", "1234", "0.00246800"),  # fractional unit
        ("0.002", "1000", "1", "0.00000200"),  # single token
        ("0.002", "1000", "1234.5678", "0.00246914"),  # rounds half away from zero
        ("12.5", "1", "3", "37.50000000"),  # gpu.hour style: per one unit
        ("0", "1000", "999999", "0E-8"),  # free model
    ],
)
def test_compute_amount(price, per_quantity, quantity, expected):
    entry = _price(
        1, price=price, per_quantity=per_quantity, unit=UNIT_TOKENS
    )
    assert compute_amount(entry, Decimal(quantity)) == Decimal(expected)


def test_compute_amount_rejects_non_positive_divisor():
    entry = _price(1)
    entry.per_quantity = Decimal(0)
    with pytest.raises(ValueError, match="non-positive per_quantity"):
        compute_amount(entry, Decimal(100))


def test_resource_sku_amounts_use_their_own_unit():
    """A gpu-hour price must not be quotable per token (schema-level guard)."""
    entry = _price(
        1,
        sku=SKU_GPU_HOUR_PREFIX + "910b",
        model_name=None,
        price="12.5",
        per_quantity="1",
        unit=UNIT_GPU_HOURS,
    )
    # 90 minutes of one card.
    assert compute_amount(entry, Decimal("1.5")) == Decimal("18.75000000")


# ---------------------------------------------------------------------------
# Identity immutability
# ---------------------------------------------------------------------------


def test_identity_change_rejected():
    existing = _price(1)
    incoming = _price(1)
    incoming.sku = SKU_TOKEN_CACHED
    with pytest.raises(InvalidException) as exc_info:
        assert_identity_unchanged(existing, incoming)
    assert "Cannot change sku" in exc_info.value.message


def test_price_change_is_allowed():
    existing = _price(1, price="0.002")
    incoming = _price(1, price="0.001")
    assert_identity_unchanged(existing, incoming)  # no raise


def test_identity_check_tolerates_naive_and_aware_instants():
    """Same instant, different tzinfo shape, must not read as a change.

    Regression: rows come back from ``UTCDateTime`` aware while a request body
    carries a naive instant; the raw comparison rejected every update.
    """
    existing = _price(1, effective_from=T0.replace(tzinfo=timezone.utc))
    incoming = _price(1, effective_from=T0)  # naive, same instant
    assert_identity_unchanged(existing, incoming)  # no raise

    moved = _price(1, effective_from=T0 + timedelta(days=1))
    with pytest.raises(InvalidException) as exc_info:
        assert_identity_unchanged(existing, moved)
    assert "effective_from" in exc_info.value.message


# ---------------------------------------------------------------------------
# Schema-level validation (the 422 boundary)
# ---------------------------------------------------------------------------


def _create_payload(**overrides):
    payload = {
        "sku": SKU_TOKEN_PROMPT,
        "model_name": "Qwen3-8B",
        "unit": UNIT_TOKENS,
        "price": Decimal("0.002"),
        "per_quantity": Decimal(1000),
        "effective_from": T0,
    }
    payload.update(overrides)
    return payload


def test_create_rejects_unit_mismatch():
    with pytest.raises(ValidationError, match="must be 'gpu_hours'"):
        PriceBookEntryCreate(**_create_payload(sku=SKU_GPU_HOUR_PREFIX + "910b"))


def test_create_rejects_unknown_sku():
    with pytest.raises(ValidationError, match="unknown billing sku"):
        PriceBookEntryCreate(**_create_payload(sku="model.token.mystery"))


def test_create_rejects_negative_price():
    with pytest.raises(ValidationError, match="must not be negative"):
        PriceBookEntryCreate(**_create_payload(price=Decimal("-1")))


def test_create_rejects_non_positive_per_quantity():
    with pytest.raises(ValidationError, match="greater than zero"):
        PriceBookEntryCreate(**_create_payload(per_quantity=Decimal(0)))


def test_create_rejects_inverted_window():
    with pytest.raises(ValidationError, match="later than effective_from"):
        PriceBookEntryCreate(**_create_payload(effective_to=T0))


def test_create_accepts_storage_sku():
    entry = PriceBookEntryCreate(
        **_create_payload(
            sku=SKU_STORAGE_GB_HOUR,
            model_name=None,
            unit="gb_hours",
            price=Decimal("0.0001"),
            per_quantity=Decimal(1),
        )
    )
    assert entry.sku == SKU_STORAGE_GB_HOUR
