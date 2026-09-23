"""Rating engine tests (WP3).

These pin the properties a bill depends on, against a real database:

* one request becomes exactly the token SKUs it should, with the cached subset
  billed once and not again inside the prompt count;
* rating twice writes nothing the second time — the sweep is a retry loop, so
  idempotency is what stands between a restart and a double charge;
* usage that cannot be priced becomes a VOID marker rather than disappearing,
  and is *promoted* to a real charge once a price exists;
* resource buckets are rated only once sealed, and the seconds → gpu-hours
  conversion keeps the fraction a sliced card contributes;
* ``off`` mode writes nothing at all.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.billing import (
    SKU_STORAGE_GB_HOUR,
    SKU_TOKEN_CACHED,
    SKU_TOKEN_COMPLETION,
    SKU_TOKEN_PROMPT,
    UNIT_GB_HOURS,
    UNIT_GPU_HOURS,
    UNIT_TOKENS,
    LedgerEntry,
    LedgerStatus,
    PriceBookEntry,
    Quota,
    QuotaLimitType,
    QuotaScope,
    SettleMode,
    Wallet,
)
from gpustack.schemas.metered_usage import (
    METER_INSTANCE_UPTIME,
    METER_STORAGE_CAPACITY,
    RESOURCE_TYPE_GPU_INSTANCE,
    RESOURCE_TYPE_PERSISTENT_VOLUME,
    UNIT_MIB_SECONDS,
    UNIT_SECONDS,
    MeteredUsage,
)
from gpustack.schemas.model_usage_details import ModelUsageDetails
from gpustack.server import billing_rater
from gpustack.server.billing_pricing import invalidate_price_cache
from gpustack.server.billing_quota import invalidate_quota_cache
from gpustack.server.billing_rater import (
    SKU_OUT_OF_SCOPE,
    BillingMode,
    BillingRater,
    billing_mode,
)

NOW = datetime(2026, 9, 22, 12, 0, 0)
DAY_START = datetime(2026, 9, 22, 0, 0, 0)
T0 = datetime(2026, 9, 1, 0, 0, 0)
ORG = 10
USER = 7
MODEL = "Qwen3-8B"


def _price(id_, sku, price, model_name=MODEL, unit=UNIT_TOKENS, per_quantity="1000"):
    return PriceBookEntry(
        id=id_,
        sku=sku,
        model_name=model_name,
        unit=unit,
        price=Decimal(price),
        per_quantity=Decimal(per_quantity),
        currency="CNY",
        version=1,
        effective_from=T0,
        is_active=True,
        created_at=NOW,
        updated_at=NOW,
    )


def _detail(
    id_,
    *,
    prompt=1000,
    cached=0,
    completion=500,
    completed=True,
    model_name=MODEL,
    consumer=ORG,
    owner=None,
):
    return ModelUsageDetails(
        id=id_,
        user_id=USER,
        user_name="user7",
        model_name=model_name,
        api_key_id=3,
        api_key_name="key3",
        consumer_principal_id=consumer,
        owner_principal_id=owner,
        date=NOW.date(),
        prompt_token_count=prompt,
        completion_token_count=completion,
        prompt_cached_token_count=cached,
        completed=completed,
        request_id=f"req-{id_}",
        created_at=NOW,
        updated_at=NOW,
        completed_at=NOW,
    )


def _bucket(
    id_,
    *,
    meter_key=METER_INSTANCE_UPTIME,
    resource_type=RESOURCE_TYPE_GPU_INSTANCE,
    seconds=3600,
    sku_count=Decimal(2),
    gpu_type="910b",
    sealed=True,
    consumer=ORG,
    unit=UNIT_SECONDS,
):
    dimensions = {"gpu_count": 2}
    if gpu_type:
        dimensions["gpu_type"] = gpu_type
    return MeteredUsage(
        id=id_,
        meter_key=meter_key,
        resource_type=resource_type,
        resource_id=id_,
        resource_name=f"inst-{id_}",
        consumer_principal_id=consumer,
        consumer_name="acme",
        sku=f"sha1:{id_}",
        sku_count=sku_count,
        dimensions=dimensions if gpu_type else {"gpu_count": 0},
        bucket_start=T0,
        quantity=seconds,
        unit=unit,
        sealed_at=NOW if sealed else None,
        created_at=NOW,
        updated_at=NOW,
    )


@pytest_asyncio.fixture
async def engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        for model in (
            PriceBookEntry,
            LedgerEntry,
            ModelUsageDetails,
            MeteredUsage,
            Wallet,
            # The sweep advances quota counters as well as writing ledger rows,
            # so the table has to exist for the sweep to complete quietly.
            Quota,
        ):
            await conn.run_sync(model.__table__.create)
    invalidate_price_cache()
    invalidate_quota_cache()
    yield engine
    invalidate_price_cache()
    invalidate_quota_cache()
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    """Stands in for the app-wide session factory the rater loop uses."""

    @asynccontextmanager
    async def _factory():
        async with AsyncSession(engine, expire_on_commit=False) as s:
            yield s

    with patch.object(billing_rater, "async_session", _factory):
        yield _factory


async def _seed(factory, *rows):
    async with factory() as s:
        for row in rows:
            s.add(row)
        await s.commit()


async def _ledger(engine, **filters):
    async with AsyncSession(engine) as s:
        rows = (await s.exec(select(LedgerEntry))).all()
    for key, value in filters.items():
        rows = [r for r in rows if getattr(r, key) == value]
    return rows


# ---------------------------------------------------------------------------
# Token rating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_request_becomes_three_token_entries(engine, session_factory):
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _price(2, SKU_TOKEN_CACHED, "0.0002"),
        _price(3, SKU_TOKEN_COMPLETION, "0.008"),
        _detail(100, prompt=1000, cached=200, completion=500),
    )

    report = await BillingRater(mode=BillingMode.SHADOW).rate_once()

    assert (report.token_sources, report.token_entries) == (1, 3)
    entries = {e.sku: e for e in await _ledger(engine)}
    # The cached subset is billed once, under its own SKU — not again as prompt.
    assert entries[SKU_TOKEN_PROMPT].quantity == Decimal(800)
    assert entries[SKU_TOKEN_CACHED].quantity == Decimal(200)
    assert entries[SKU_TOKEN_COMPLETION].quantity == Decimal(500)
    # 800 tokens at 0.002 per 1000.
    assert entries[SKU_TOKEN_PROMPT].amount == Decimal("0.00160000")
    assert entries[SKU_TOKEN_CACHED].amount == Decimal("0.00004000")
    assert entries[SKU_TOKEN_COMPLETION].amount == Decimal("0.00400000")
    for entry in entries.values():
        assert entry.settle_mode == SettleMode.REALTIME.value
        assert entry.status == LedgerStatus.PENDING.value
        assert entry.principal_id == ORG
        assert entry.source_table == "model_usage_details"
        assert entry.source_id == 100
        assert entry.unit == UNIT_TOKENS
        assert entry.price_book_version == 1
        assert entry.request_id == "req-100"


@pytest.mark.asyncio
async def test_rating_twice_writes_nothing_new(engine, session_factory):
    """The sweep is a retry loop; idempotency is what makes that safe."""
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _price(3, SKU_TOKEN_COMPLETION, "0.008"),
        _detail(100),
    )
    rater = BillingRater(mode=BillingMode.SHADOW)

    first = await rater.rate_once()
    second = await rater.rate_once()

    assert first.token_entries == 2
    assert (second.token_sources, second.token_entries) == (0, 0)
    assert len(await _ledger(engine)) == 2


@pytest.mark.asyncio
async def test_interrupted_request_is_not_charged(engine, session_factory):
    """``completed=False`` means the counts are estimates — the platform's rule
    is that an interrupted request is not billed."""
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _detail(100, completed=False),
    )

    report = await BillingRater(mode=BillingMode.SHADOW).rate_once()

    assert report.token_sources == 0
    assert await _ledger(engine) == []


@pytest.mark.asyncio
async def test_cached_above_prompt_clamps_instead_of_going_negative(
    engine, session_factory
):
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _price(2, SKU_TOKEN_CACHED, "0.0002"),
        _detail(100, prompt=100, cached=250, completion=0),
    )

    await BillingRater(mode=BillingMode.SHADOW).rate_once()

    entries = {e.sku: e for e in await _ledger(engine)}
    assert SKU_TOKEN_PROMPT not in entries, "a zero-quantity SKU writes no row"
    assert entries[SKU_TOKEN_CACHED].quantity == Decimal(250)
    assert all(e.amount >= 0 for e in entries.values())


@pytest.mark.asyncio
async def test_unpriced_model_is_voided_then_promoted(engine, session_factory):
    """An unpriced model must leave a visible gap, not a silent zero."""
    await _seed(session_factory, _detail(100))
    rater = BillingRater(mode=BillingMode.SHADOW)

    first = await rater.rate_once()
    assert first.token_unpriced == 1
    voided = await _ledger(engine, status=LedgerStatus.VOID.value)
    # One marker per SKU the request actually uses: prompt and completion.
    # The cached SKU is absent because this request had no cache hit, and a
    # zero-quantity SKU writes no row.
    assert len(voided) == 2
    assert {e.sku for e in voided} == {SKU_TOKEN_PROMPT, SKU_TOKEN_COMPLETION}
    assert all(e.amount == Decimal(0) for e in voided)
    # Idempotency depends on the marker carrying its source, so pin it.
    assert all(e.source_id == 100 for e in voided)
    assert "no active price" in first.unpriced_reasons[0]

    # Still unpriced: retried, but the markers are not duplicated.
    await rater.rate_once()
    assert len(await _ledger(engine, status=LedgerStatus.VOID.value)) == 2

    # A price appears — the markers are promoted in place.
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _price(3, SKU_TOKEN_COMPLETION, "0.008"),
    )
    third = await rater.rate_once()
    assert third.promoted == 2
    live = await _ledger(engine, status=LedgerStatus.PENDING.value)
    assert {e.sku for e in live} == {SKU_TOKEN_PROMPT, SKU_TOKEN_COMPLETION}
    assert all(e.amount > 0 for e in live)
    # Promotion fills the placeholder rather than adding a second row.
    assert len(await _ledger(engine)) == 2
    assert await _ledger(engine, status=LedgerStatus.VOID.value) == []


@pytest.mark.asyncio
async def test_usage_with_no_payer_is_voided(engine, session_factory):
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _detail(100, consumer=None, owner=None),
    )

    report = await BillingRater(mode=BillingMode.SHADOW).rate_once()

    assert report.token_unpriced == 1
    entries = await _ledger(engine, status=LedgerStatus.VOID.value)
    assert entries and entries[0].principal_id is None
    assert "no attributable payer" in report.unpriced_reasons[0]


@pytest.mark.asyncio
async def test_owner_principal_is_the_fallback_payer(engine, session_factory):
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _price(3, SKU_TOKEN_COMPLETION, "0.008"),
        _detail(100, consumer=None, owner=ORG),
    )

    await BillingRater(mode=BillingMode.SHADOW).rate_once()

    entry = (await _ledger(engine, sku=SKU_TOKEN_PROMPT))[0]
    assert entry.status == LedgerStatus.PENDING.value
    assert entry.principal_id == ORG


# ---------------------------------------------------------------------------
# Resource rating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sealed_gpu_bucket_becomes_gpu_hours(engine, session_factory):
    await _seed(
        session_factory,
        _price(
            1,
            "gpu.hour.910b",
            "12.5",
            model_name=None,
            unit=UNIT_GPU_HOURS,
            per_quantity="1",
        ),
        _bucket(200, seconds=3600, sku_count=Decimal(2)),
    )

    report = await BillingRater(mode=BillingMode.SHADOW).rate_once()

    assert (report.resource_sources, report.resource_entries) == (1, 1)
    entry = (await _ledger(engine))[0]
    assert entry.sku == "gpu.hour.910b"
    assert entry.quantity == Decimal(2)  # 3600s x 2 cards / 3600
    assert entry.unit == UNIT_GPU_HOURS
    assert entry.amount == Decimal("25.00000000")
    assert entry.settle_mode == SettleMode.DEFERRED.value
    assert entry.source_table == "metered_usage"
    assert entry.resource_id == 200


@pytest.mark.asyncio
async def test_sliced_card_keeps_its_fraction(engine, session_factory):
    await _seed(
        session_factory,
        _price(
            1,
            "gpu.hour.910b",
            "12.5",
            model_name=None,
            unit=UNIT_GPU_HOURS,
            per_quantity="1",
        ),
        _bucket(201, seconds=1800, sku_count=Decimal("0.5")),
    )

    await BillingRater(mode=BillingMode.SHADOW).rate_once()

    entry = (await _ledger(engine))[0]
    # 1800s x 0.5 card / 3600 = 0.25 gpu-hours.
    assert entry.quantity == Decimal("0.25")
    assert entry.amount == Decimal("3.12500000")


@pytest.mark.asyncio
async def test_unsealed_bucket_is_not_rated_yet(engine, session_factory):
    """An open bucket can still grow; rating it would bill an unfinished hour."""
    await _seed(
        session_factory,
        _price(
            1,
            "gpu.hour.910b",
            "12.5",
            model_name=None,
            unit=UNIT_GPU_HOURS,
            per_quantity="1",
        ),
        _bucket(202, sealed=False),
    )

    report = await BillingRater(mode=BillingMode.SHADOW).rate_once()

    assert report.resource_sources == 0
    assert await _ledger(engine) == []


@pytest.mark.asyncio
async def test_cpu_only_instance_is_marked_out_of_scope(engine, session_factory):
    await _seed(
        session_factory, _bucket(203, gpu_type=None, sku_count=Decimal(0))
    )

    report = await BillingRater(mode=BillingMode.SHADOW).rate_once()

    assert report.resource_unpriced == 1
    entries = await _ledger(engine, status=LedgerStatus.VOID.value)
    assert len(entries) == 1
    assert entries[0].sku == SKU_OUT_OF_SCOPE


@pytest.mark.asyncio
async def test_storage_bucket_becomes_gb_hours(engine, session_factory):
    await _seed(
        session_factory,
        _price(
            1,
            SKU_STORAGE_GB_HOUR,
            "0.0001",
            model_name=None,
            unit=UNIT_GB_HOURS,
            per_quantity="1",
        ),
        _bucket(
            204,
            meter_key=METER_STORAGE_CAPACITY,
            resource_type=RESOURCE_TYPE_PERSISTENT_VOLUME,
            seconds=1024 * 3600,  # 1 GiB held for one hour
            sku_count=Decimal(1),
            gpu_type=None,
            unit=UNIT_MIB_SECONDS,
        ),
    )

    await BillingRater(mode=BillingMode.SHADOW).rate_once()

    entry = (await _ledger(engine))[0]
    assert entry.sku == SKU_STORAGE_GB_HOUR
    assert entry.quantity == Decimal(1)
    assert entry.unit == UNIT_GB_HOURS


@pytest.mark.asyncio
async def test_resource_rating_is_idempotent(engine, session_factory):
    await _seed(
        session_factory,
        _price(
            1,
            "gpu.hour.910b",
            "12.5",
            model_name=None,
            unit=UNIT_GPU_HOURS,
            per_quantity="1",
        ),
        _bucket(205),
    )
    rater = BillingRater(mode=BillingMode.SHADOW)

    await rater.rate_once()
    second = await rater.rate_once()

    assert second.resource_sources == 0
    assert len(await _ledger(engine)) == 1


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_off_mode_writes_nothing(engine, session_factory):
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _detail(100),
        _bucket(200),
    )

    report = await BillingRater(mode=BillingMode.OFF).rate_once()

    assert report.entries == 0
    assert await _ledger(engine) == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("off", BillingMode.OFF),
        ("SHADOW", BillingMode.SHADOW),
        (" enforce ", BillingMode.ENFORCE),
    ],
)
def test_billing_mode_parsing(raw, expected):
    assert billing_mode(raw) is expected


def test_billing_mode_rejects_a_typo():
    """A typo must fail at startup, not silently disable billing."""
    with pytest.raises(ValueError, match="Invalid GPUSTACK_BILLING_MODE"):
        billing_mode("enfoce")


def test_default_mode_is_shadow():
    assert billing_mode() is BillingMode.SHADOW


def test_rater_rejects_non_positive_cadence():
    with pytest.raises(ValueError, match="interval must be positive"):
        BillingRater(interval_seconds=0)
    with pytest.raises(ValueError, match="batch size must be positive"):
        BillingRater(batch_size=0)


@pytest.mark.asyncio
async def test_batch_size_bounds_one_sweep(engine, session_factory):
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _price(3, SKU_TOKEN_COMPLETION, "0.008"),
        *[_detail(300 + i) for i in range(5)],
    )
    rater = BillingRater(mode=BillingMode.SHADOW, batch_size=2)

    first = await rater.rate_once()
    assert first.token_sources == 2

    await rater.rate_once()
    await rater.rate_once()
    # 5 requests x 2 priced SKUs (prompt + completion; cached is zero).
    assert len(await _ledger(engine)) == 10


@pytest.mark.asyncio
async def test_unpriced_gap_warns_once_then_reports_a_count(engine, session_factory, caplog):
    """A permanently unpriced model must not warn every sweep.

    The gap is retried until a price appears, so without dedup the same warning
    would land in the log every interval forever and bury the new gaps that
    actually need someone to act.
    """
    await _seed(session_factory, _detail(100))
    rater = BillingRater(mode=BillingMode.SHADOW)

    first = await rater.rate_once()
    with caplog.at_level("WARNING", logger="gpustack.server.billing_rater"):
        rater._log_unpriced(first)
    warnings_first = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings_first) == 1
    assert "no active price" in warnings_first[0].getMessage()

    caplog.clear()
    second = await rater.rate_once()
    assert second.unpriced_reasons == first.unpriced_reasons  # still unpriced
    with caplog.at_level("INFO", logger="gpustack.server.billing_rater"):
        rater._log_unpriced(second)
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []
    infos = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    assert any("previously reported unpriced" in m for m in infos)


@pytest.mark.asyncio
async def test_report_summary_is_loggable(engine, session_factory):
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _price(3, SKU_TOKEN_COMPLETION, "0.008"),
        _detail(100),
    )

    report = await BillingRater(mode=BillingMode.SHADOW).rate_once()

    text = report.summary()
    assert "shadow" in text
    assert "2 token entries" in text


@pytest.mark.asyncio
async def test_enforce_mode_rates_without_debiting_any_wallet(engine, session_factory):
    """Rating never moves money, whatever the mode.

    Settlement is a separate loop (``server.billing_settlement``), so enforce
    mode rating must produce exactly what shadow produces and leave wallets
    alone — the property that makes a shadow reconciliation trustworthy as a
    preview of what enforce will charge.
    """
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _price(3, SKU_TOKEN_COMPLETION, "0.008"),
        _detail(100),
    )

    shadow = await BillingRater(mode=BillingMode.SHADOW).rate_once()
    async with AsyncSession(engine) as s:
        await s.exec(delete(LedgerEntry))
        await s.commit()
    invalidate_price_cache()

    enforced = await BillingRater(mode=BillingMode.ENFORCE).rate_once()

    assert enforced.token_entries == shadow.token_entries == 2
    entries = await _ledger(engine)
    assert all(e.status == LedgerStatus.PENDING.value for e in entries)
    async with AsyncSession(engine) as s:
        wallets = (await s.exec(select(Wallet))).all()
    assert wallets == []


@pytest.mark.asyncio
async def test_rated_at_uses_the_request_completion_time(engine, session_factory):
    """Charges land in the period the usage happened in, not when it was rated."""
    late = NOW - timedelta(days=3)
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _price(3, SKU_TOKEN_COMPLETION, "0.008"),
        _detail(100),
    )
    async with session_factory() as s:
        row = (
            await s.exec(select(ModelUsageDetails).where(ModelUsageDetails.id == 100))
        ).one()
        row.completed_at = late
        s.add(row)
        await s.commit()
    invalidate_price_cache()

    await BillingRater(mode=BillingMode.SHADOW).rate_once()

    entry = (await _ledger(engine, sku=SKU_TOKEN_PROMPT))[0]
    assert entry.occurred_at.replace(tzinfo=None) == late


# ---------------------------------------------------------------------------
# Quota counters — the sweep advances them with the rows it writes
# ---------------------------------------------------------------------------


def _quota(id_=1, *, api_key_id=3, limit_value="100000", window_start=None, used="0"):
    return Quota(
        id=id_,
        scope=QuotaScope.API_KEY,
        api_key_id=api_key_id,
        limit_type=QuotaLimitType.DAILY_TOKENS,
        limit_value=Decimal(limit_value),
        enabled=True,
        window_start=window_start or DAY_START,
        used=Decimal(used),
        created_at=NOW,
        updated_at=NOW,
    )


async def _quota_row(engine, id_=1):
    async with AsyncSession(engine) as s:
        return (await s.exec(select(Quota).where(Quota.id == id_))).first()


@pytest.mark.asyncio
async def test_a_sweep_advances_the_counter_for_the_rows_it_wrote(
    engine, session_factory
):
    """Ledger and ceiling move together, in the sweep's one transaction.

    A counter advanced without its rows would not be reproducible from the
    ledger, and rows written without advancing a counter would leave the ceiling
    blind to them until the next window rolled.
    """
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _price(2, SKU_TOKEN_COMPLETION, "0.006"),
        _quota(),
        _detail(100, prompt=1000, completion=500),
    )

    report = await BillingRater(mode=BillingMode.SHADOW).rate_once()

    assert report.quotas_advanced == 1
    assert (await _quota_row(engine)).used == Decimal("1500")


@pytest.mark.asyncio
async def test_a_second_sweep_does_not_count_the_same_request_twice(
    engine, session_factory
):
    """The rater re-reads its sources every tick; a ceiling that advanced for a
    row it had already written would bite early and look like a mystery limit."""
    await _seed(
        session_factory,
        _price(1, SKU_TOKEN_PROMPT, "0.002"),
        _price(2, SKU_TOKEN_COMPLETION, "0.006"),
        _quota(),
        _detail(100, prompt=1000, completion=500),
    )

    await BillingRater(mode=BillingMode.SHADOW).rate_once()
    second = await BillingRater(mode=BillingMode.SHADOW).rate_once()

    assert second.quotas_advanced == 0
    assert (await _quota_row(engine)).used == Decimal("1500")


@pytest.mark.asyncio
async def test_unpriced_usage_does_not_advance_a_counter(engine, session_factory):
    """A VOID placeholder is not consumption; counting it would charge a ceiling
    for usage that has not been billed yet."""
    await _seed(
        session_factory,
        _quota(),
        _detail(100, prompt=5000, completion=500),
    )

    report = await BillingRater(mode=BillingMode.SHADOW).rate_once()

    assert report.token_unpriced == 1
    assert report.quotas_advanced == 0
    assert (await _quota_row(engine)).used == Decimal("0")


@pytest.mark.asyncio
async def test_a_promoted_placeholder_starts_counting(engine, session_factory):
    """When the missing price appears, the row it unblocks is real usage and the
    ceiling has to see it — the promotion is the moment it becomes billable."""
    await _seed(
        session_factory,
        _price(2, SKU_TOKEN_COMPLETION, "0.006"),
        _quota(),
        _detail(100, prompt=1000, completion=500),
    )
    first = await BillingRater(mode=BillingMode.SHADOW).rate_once()
    assert first.token_unpriced == 1
    assert (await _quota_row(engine)).used == Decimal("0")

    await _seed(session_factory, _price(1, SKU_TOKEN_PROMPT, "0.002"))
    second = await BillingRater(mode=BillingMode.SHADOW).rate_once()

    assert second.promoted == 2
    assert (await _quota_row(engine)).used == Decimal("1500")
