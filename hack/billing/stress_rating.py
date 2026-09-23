#!/usr/bin/env python3
"""Rating-pipeline stress run (WP8.1).

Answers one operational question: **can the rater keep up?** A rating sweep runs
every ``GPUSTACK_BILLING_RATE_INTERVAL_SECONDS`` (30s by default) and processes at
most ``GPUSTACK_BILLING_RATE_BATCH_SIZE`` rows per source per pass. If a sweep
takes longer than its interval, or sustains fewer rows per second than the
cluster produces, the ledger falls behind — and the way that shows up is a bill
that arrives late, which is the hardest kind of billing complaint to argue with.

What it measures
----------------
* per-sweep wall time (p50 / p95 / max) against the interval budget;
* sustained throughput in source rows per second, against an assumed peak;
* the backlog curve — it must fall monotonically to zero and a pass over an
  empty backlog must write nothing (which is also the double-counting check);
* the ledger row count against an independently computed expectation, and the
* absence of unpriced rows, so a slow pipeline is not mistaken for a cheap one.

Nothing here is a unit test: the correctness of what gets written is
``shadow_rating_drill.py``'s job, and its exit code is the release gate. This one
measures how long that takes at volume.

Usage
-----
    DATABASE_URL=postgresql://user@host:5432/gpustack \
        STRESS_REQUESTS=20000 STRESS_BUCKETS=5000 \
        uv run python hack/billing/stress_rating.py

Seeded rows live in reserved id ranges and are deleted on the way out, including
on failure, so a run against a shared database leaves nothing behind. Exits
non-zero when any threshold is missed.
"""

import asyncio
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Tuple

from sqlalchemy import delete, func
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
from gpustack.server import db as gpustack_db
from gpustack.server.billing_pricing import invalidate_price_cache
from gpustack.server.billing_quota import invalidate_quota_cache
from gpustack.server.billing_rater import BillingMode, BillingRater
from gpustack.server.init_db import init_db_engine

DEFAULT_DATABASE_URL = "postgresql://root@localhost:5432/gpustack"

# Reserved ranges, distinct from the drill's, so the two tools can share a
# database without clearing each other's rows.
REQUEST_IDS = range(9910001, 9910001 + 500000)
BUCKET_IDS = range(9920001, 9920001 + 200000)

NOW = datetime.now(timezone.utc).replace(tzinfo=None)
HOUR_AGO = (NOW - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

MODELS = ("Stress-7B", "Stress-30B")
ORG = 990199
USER = 990299
API_KEY = 990399

# Prices seeded by *this* run all carry this effective_from, which is what lets
# cleanup delete exactly its own rows. A pattern match on (sku, model_name) would
# also delete a real gpu.hour price on a shared database, and a stress tool that
# can quietly unprice a production accelerator is not a tool anybody will run
# twice.
SEED_EFFECTIVE_FROM = (NOW - timedelta(days=1)).replace(microsecond=0)

PRICES: List[Tuple[str, str, str, str, str]] = [
    (SKU_TOKEN_PROMPT, MODELS[0], "0.002", "1000", UNIT_TOKENS),
    (SKU_TOKEN_CACHED, MODELS[0], "0.0002", "1000", UNIT_TOKENS),
    (SKU_TOKEN_COMPLETION, MODELS[0], "0.006", "1000", UNIT_TOKENS),
    (SKU_TOKEN_PROMPT, MODELS[1], "0.004", "1000", UNIT_TOKENS),
    (SKU_TOKEN_CACHED, MODELS[1], "0.0004", "1000", UNIT_TOKENS),
    (SKU_TOKEN_COMPLETION, MODELS[1], "0.012", "1000", UNIT_TOKENS),
]
# Resource prices have no model: their specificity lives in the sku itself
# (``gpu.hour.<gpu_type>``), which is what the resolver treats as a family default.
RESOURCE_PRICES: List[Tuple[str, str, str, str]] = [
    ("gpu.hour.910b", "12.5", "1", UNIT_GPU_HOURS),
    (SKU_STORAGE_GB_HOUR, "0.0001", "1", UNIT_GB_HOURS),
]
RESOURCE_SKUS = [sku for sku, _p, _q, _u in RESOURCE_PRICES]


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, default))


REQUESTS = _int_env("STRESS_REQUESTS", 5000)
BUCKETS = _int_env("STRESS_BUCKETS", 2000)
# How many sweeps to allow before declaring the backlog undrainable. Generous on
# purpose: the interesting failure is a sweep that takes longer than its
# interval, not one that needs several passes to clear a large seed.
MAX_SWEEPS = _int_env("STRESS_MAX_SWEEPS", 200)
BUDGET_SECONDS = float(os.getenv("STRESS_BUDGET_SECONDS", "30"))
# The peak the pipeline has to keep up with, in source rows per second. A cluster
# producing one completed request per worker per second plus one sealed bucket per
# instance per hour is well under this; the number exists so "it finished" is not
# the only verdict.
PEAK_ROWS_PER_SECOND = float(os.getenv("STRESS_PEAK_ROWS_PER_SECOND", "200"))
BATCH_SIZE = _int_env("STRESS_BATCH_SIZE", 500)


def expected_ledger_rows(requests: int, buckets: int) -> int:
    """How many ledger rows the seed must produce, computed from the seed itself.

    Independent of the rater: a pipeline that wrote nothing at all would still
    "drain the backlog" if the backlog were measured by what the rater says it
    did, so the expectation is derived here from the row shapes below.
    """
    per_request = 0
    for index in range(requests):
        # Every tenth request hits a prefix cache, which adds the cached SKU.
        per_request += 3 if index % 10 == 0 else 2
    # Every bucket is priced: half are accelerator uptime, half are storage.
    return per_request + buckets


async def cleanup(session: AsyncSession) -> None:
    await session.exec(
        delete(LedgerEntry).where(
            LedgerEntry.source_table == "model_usage_details",
            LedgerEntry.source_id.in_(list(REQUEST_IDS)[:REQUESTS]),
        )
    )
    await session.exec(
        delete(LedgerEntry).where(
            LedgerEntry.source_table == "metered_usage",
            LedgerEntry.source_id.in_(list(BUCKET_IDS)[:BUCKETS]),
        )
    )
    await session.exec(
        delete(ModelUsageDetails).where(
            ModelUsageDetails.id.in_(list(REQUEST_IDS)[:REQUESTS])
        )
    )
    await session.exec(
        delete(MeteredUsage).where(MeteredUsage.id.in_(list(BUCKET_IDS)[:BUCKETS]))
    )
    # Token prices for the stress models are unambiguously ours.
    await session.exec(
        delete(PriceBookEntry).where(PriceBookEntry.model_name.in_(list(MODELS)))
    )
    # Resource prices only when they carry this run's effective_from, so a real
    # gpu.hour price on a shared database survives.
    await session.exec(
        delete(PriceBookEntry).where(
            PriceBookEntry.sku.in_(RESOURCE_SKUS),
            PriceBookEntry.model_name.is_(None),
            PriceBookEntry.effective_from == SEED_EFFECTIVE_FROM,
        )
    )
    await session.commit()


async def seed(session: AsyncSession) -> None:
    """Seed in chunks: one transaction holding 20k rows is a long lock, and the
    point of this tool is to measure the rater rather than the seeder."""
    chunk = 1000
    for sku, model, price, per, unit in PRICES:
        session.add(
            PriceBookEntry(
                sku=sku,
                model_name=model,
                unit=unit,
                price=Decimal(price),
                per_quantity=Decimal(per),
                currency="CNY",
                version=1,
                effective_from=SEED_EFFECTIVE_FROM,
                is_active=True,
                created_at=NOW,
                updated_at=NOW,
            )
        )
    for sku, price, per, unit in RESOURCE_PRICES:
        session.add(
            PriceBookEntry(
                sku=sku,
                model_name=None,
                unit=unit,
                price=Decimal(price),
                per_quantity=Decimal(per),
                currency="CNY",
                version=1,
                effective_from=SEED_EFFECTIVE_FROM,
                is_active=True,
                created_at=NOW,
                updated_at=NOW,
            )
        )
    await session.commit()

    ids = list(REQUEST_IDS)[:REQUESTS]
    for start in range(0, len(ids), chunk):
        batch = ids[start : start + chunk]
        for index, rid in enumerate(batch):
            cached = 512 if (start + index) % 10 == 0 else 0
            prompt = 1024 + cached
            session.add(
                ModelUsageDetails(
                    id=rid,
                    user_id=USER,
                    user_name="stress-user",
                    model_id=rid % 100,
                    model_name=MODELS[index % len(MODELS)],
                    api_key_id=API_KEY,
                    api_key_name="stress-key",
                    consumer_principal_id=ORG,
                    owner_principal_id=None,
                    date=NOW.date(),
                    prompt_token_count=prompt,
                    completion_token_count=256,
                    prompt_cached_token_count=cached,
                    completed=True,
                    request_id=f"stress-req-{rid}",
                    started_at=HOUR_AGO,
                    completed_at=HOUR_AGO + timedelta(seconds=5),
                    ttft_ms=320,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        await session.commit()

    bucket_ids = list(BUCKET_IDS)[:BUCKETS]
    for start in range(0, len(bucket_ids), chunk):
        batch = bucket_ids[start : start + chunk]
        for index, bid in enumerate(batch):
            # Half accelerator uptime (1 card-hour), half storage (100 GiB for an
            # hour), so both resource SKUs and both unit conversions are exercised.
            is_gpu = index % 2 == 0
            session.add(
                MeteredUsage(
                    id=bid,
                    meter_key=(
                        METER_INSTANCE_UPTIME if is_gpu else METER_STORAGE_CAPACITY
                    ),
                    resource_type=(
                        RESOURCE_TYPE_GPU_INSTANCE
                        if is_gpu
                        else RESOURCE_TYPE_PERSISTENT_VOLUME
                    ),
                    resource_id=bid % 50,
                    resource_name=f"stress-{'gpu' if is_gpu else 'pv'}-{bid % 50}",
                    consumer_principal_id=ORG,
                    consumer_name="stress-org",
                    sku=f"sha1:stress{bid}",
                    # Card count; fractional for a sliced accelerator, which is
                    # why the rater multiplies rather than rounds.
                    sku_count=Decimal(1),
                    dimensions={"gpu_type": "910b", "gpu_count": 1} if is_gpu else {},
                    bucket_start=HOUR_AGO,
                    quantity=Decimal(3600 if is_gpu else 102400),
                    unit=UNIT_SECONDS if is_gpu else UNIT_MIB_SECONDS,
                    settled_until=NOW,
                    sealed_at=NOW,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        await session.commit()


async def count_unrated(engine) -> Tuple[int, int]:
    """Seeded rows the rater has not produced a non-VOID entry for, per source.

    Scoped to this run's reserved id ranges, and that scoping is load bearing:
    counted across the whole ledger, rows left behind by another tool (the drill
    keeps its VOID placeholders, by design) make the difference go negative, and
    a negative backlog never reaches zero — so the drain loop runs to its sweep
    cap and the throughput figure ends up measuring empty passes.
    """
    request_ids = list(REQUEST_IDS)[:REQUESTS]
    bucket_ids = list(BUCKET_IDS)[:BUCKETS]
    async with AsyncSession(engine) as session:
        details = (
            await session.exec(
                select(func.count(ModelUsageDetails.id)).where(
                    ModelUsageDetails.id.in_(request_ids),
                    ModelUsageDetails.completed.is_(True),
                )
            )
        ).one()
        rated_details = (
            await session.exec(
                select(func.count(func.distinct(LedgerEntry.source_id))).where(
                    LedgerEntry.source_table == "model_usage_details",
                    LedgerEntry.status != LedgerStatus.VOID.value,
                    LedgerEntry.source_id >= request_ids[0],
                    LedgerEntry.source_id <= request_ids[-1],
                )
            )
        ).one()
        buckets = (
            await session.exec(
                select(func.count(MeteredUsage.id)).where(
                    MeteredUsage.id.in_(bucket_ids),
                    MeteredUsage.sealed_at.is_not(None),
                )
            )
        ).one()
        rated_buckets = (
            await session.exec(
                select(func.count(func.distinct(LedgerEntry.source_id))).where(
                    LedgerEntry.source_table == "metered_usage",
                    LedgerEntry.status != LedgerStatus.VOID.value,
                    LedgerEntry.source_id >= bucket_ids[0],
                    LedgerEntry.source_id <= bucket_ids[-1],
                )
            )
        ).one()
    return int(details) - int(rated_details), int(buckets) - int(rated_buckets)


async def run(database_url: str) -> int:
    # Awaited, and through the platform's own initializer: a URL carrying libpq's
    # ``sslmode`` is normalized here exactly as it is at server startup, and the
    # rater opens sessions through the same engine.
    engine = await init_db_engine(database_url)
    gpustack_db.engine = engine
    problems: List[str] = []
    bar = "=" * 78

    print(bar)
    print(
        f"计价管道压测：{REQUESTS} 请求明细 + {BUCKETS} 资源桶，"
        f"batch={BATCH_SIZE}，周期预算={BUDGET_SECONDS}s，"
        f"峰值要求={PEAK_ROWS_PER_SECOND} 行/秒"
    )
    print(bar)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        await cleanup(session)
        started = time.monotonic()
        await seed(session)
        seed_seconds = time.monotonic() - started
    print(f"[1] 种子写入完成，用时 {seed_seconds:.1f}s")

    invalidate_price_cache()
    invalidate_quota_cache()
    # Shadow: this measures the rating pipeline, and shadow is the mode a
    # deployment reconciles in. Settlement and invoicing have their own runs.
    rater = BillingRater(mode=BillingMode.SHADOW, batch_size=BATCH_SIZE)

    durations: List[float] = []
    written: List[int] = []
    backlog: List[Tuple[int, int]] = []
    total_sweeps = 0
    drain_started = time.monotonic()
    for sweep in range(1, MAX_SWEEPS + 1):
        before = time.monotonic()
        report = await rater.rate_once()
        durations.append(time.monotonic() - before)
        written.append(report.entries)
        total_sweeps = sweep
        pending_details, pending_buckets = await count_unrated(engine)
        backlog.append((pending_details, pending_buckets))
        print(
            f"    sweep {sweep:>3}: {durations[-1]:6.2f}s, "
            f"{report.entries:>6} entries, unpriced(全库)="
            f"{report.token_unpriced + report.resource_unpriced}, "
            f"本次积压={pending_details}/{pending_buckets}"
        )
        if pending_details == 0 and pending_buckets == 0:
            break
    drain_seconds = time.monotonic() - drain_started
    # The pipeline's own rate, excluding the time this tool spends counting the
    # backlog between sweeps — which is measurement, not rating.
    sweep_seconds = sum(durations)

    # One more pass over an empty backlog: it must write nothing, which is the
    # no-double-counting check at volume.
    idle_started = time.monotonic()
    idle_report = await rater.rate_once()
    idle_seconds = time.monotonic() - idle_started

    async with AsyncSession(engine) as session:
        ledger_rows = (
            await session.exec(
                select(func.count(LedgerEntry.id)).where(
                    LedgerEntry.principal_id == ORG,
                    LedgerEntry.deleted_at.is_(None),
                )
            )
        ).one()
        void_rows = (
            await session.exec(
                select(func.count(LedgerEntry.id)).where(
                    LedgerEntry.principal_id == ORG,
                    LedgerEntry.status == LedgerStatus.VOID.value,
                )
            )
        ).one()

    expected = expected_ledger_rows(REQUESTS, BUCKETS)
    throughput = (REQUESTS + BUCKETS) / sweep_seconds if sweep_seconds else 0.0
    slowest = max(durations) if durations else 0.0
    p50 = statistics.median(durations) if durations else 0.0
    p95 = (
        statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else slowest
    )

    print("\n[2] 结果")
    print(
        f"    sweep 次数 {total_sweeps}（+1 空转），排空壁钟 {drain_seconds:.1f}s / "
        f"sweep 累计 {sweep_seconds:.1f}s，吞吐 {throughput:.0f} 源行/秒"
    )
    print(
        f"    单次 sweep 耗时 p50={p50:.2f}s p95={p95:.2f}s max={slowest:.2f}s"
        f"（周期预算 {BUDGET_SECONDS}s），空转一次 {idle_seconds:.2f}s"
    )
    print(
        f"    ledger {int(ledger_rows)} 行（期望 {expected}），其中 VOID "
        f"{int(void_rows)} 行；末轮积压 {backlog[-1] if backlog else (0, 0)}"
    )

    # -- thresholds -------------------------------------------------------
    if slowest > BUDGET_SECONDS:
        problems.append(
            f"单次 sweep 最慢 {slowest:.2f}s 超过周期预算 {BUDGET_SECONDS}s：管道会落后于产出"
        )
    if throughput < PEAK_ROWS_PER_SECOND:
        problems.append(
            f"吞吐 {throughput:.0f} 行/秒 低于峰值要求 {PEAK_ROWS_PER_SECOND:.0f}"
        )
    if idle_report.entries != 0:
        problems.append(f"空转一轮仍写入 {idle_report.entries} 条：存在重复计价")
    if int(ledger_rows) != expected:
        problems.append(f"ledger 行数 {int(ledger_rows)} ≠ 期望 {expected}")
    if int(void_rows) != 0:
        problems.append(f"{int(void_rows)} 条未计价（VOID）：种子价格不完整")
    if backlog and backlog[-1] != (0, 0):
        problems.append(f"积压未排空：{backlog[-1]}")
    # Monotonic drain: a backlog that rises between sweeps means the sweep is
    # producing work faster than it retires it.
    for previous, current in zip(backlog, backlog[1:]):
        if sum(current) > sum(previous):
            problems.append(f"积压增长：{previous} → {current}")
            break

    print("\n" + bar)
    if problems:
        print(f"压测未通过（{len(problems)} 项）：")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print(
            f"压测通过：{REQUESTS + BUCKETS} 源行在 {sweep_seconds:.1f}s sweep 时间内排空"
            f"（{throughput:.0f} 行/秒 ≥ {PEAK_ROWS_PER_SECOND:.0f}），"
            f"最慢单次 sweep {slowest:.2f}s ≤ {BUDGET_SECONDS}s，"
            f"空转零写入，ledger {int(ledger_rows)} 行与期望一致且无未计价"
        )
    print(bar)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        await cleanup(session)
    print("清理完成（种子数据与本次 ledger 已删除）")

    await engine.dispose()
    return 1 if problems else 0


def main() -> int:
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    try:
        return asyncio.run(run(url))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
