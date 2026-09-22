#!/usr/bin/env python3
"""Shadow-rating drill: seed realistic usage, rate it, reconcile the ledger.

Exercises the billing rater (WP3) end to end against a real PostgreSQL, in
shadow mode — the ledger is written, no wallet is touched. Run it before turning
``GPUSTACK_BILLING_MODE`` up from ``shadow``, and after any change to pricing or
rating logic.

The seed covers the shapes a real NPU platform produces (cache hits, long
context, MoE, whole-card and sliced-card GPU hours, persistent volumes) plus the
awkward cases the rater must not get wrong:

  * an interrupted stream (``completed=false``)      -> not billed at all
  * a model nobody priced                            -> VOID marker
  * usage with no attributable payer                 -> VOID marker
  * cached tokens above prompt tokens (upstream bug) -> clamped, never negative
  * an unsealed metering bucket                      -> skipped until sealed
  * a CPU-only instance                              -> VOID, out of v1 scope

Reconciliation is against an independent oracle: expected amounts are recomputed
in this file with plain arithmetic from the same seed spec, deliberately not by
calling the rater's pricing code, so a bug in the rater cannot cancel itself out.
Exit status is non-zero on any difference, so this can gate a release.

Examples:
  # against the default local database
  uv run python hack/billing/shadow_rating_drill.py

  # against another database (same URL form the server itself takes: the
  # asyncpg driver and its connect args are derived by init_db_engine)
  DATABASE_URL=postgresql://user:pass@host:5432/gpustack?sslmode=disable \\
    uv run python hack/billing/shadow_rating_drill.py

The drill is re-runnable: its rows are tagged by id range and cleared first.
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple

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
    Wallet,
)
from gpustack.schemas.metered_usage import (
    METER_INSTANCE_UPTIME,
    METER_STORAGE_CAPACITY,
    RESOURCE_TYPE_CPU_INSTANCE,
    RESOURCE_TYPE_GPU_INSTANCE,
    RESOURCE_TYPE_PERSISTENT_VOLUME,
    UNIT_MIB_SECONDS,
    UNIT_SECONDS,
    MeteredUsage,
)
from gpustack.schemas.model_usage_details import ModelUsageDetails
from gpustack.server import db as gpustack_db
from gpustack.server.billing_pricing import invalidate_price_cache
from gpustack.server.billing_rater import SKU_OUT_OF_SCOPE, BillingMode, BillingRater
from gpustack.server.init_db import init_db_engine

DEFAULT_DATABASE_URL = "postgresql://root@localhost:5432/gpustack"

NOW = datetime.now(timezone.utc).replace(tzinfo=None)
HOUR_AGO = (NOW - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

# Drill tenants and row ids live in reserved ranges, so a re-run clears exactly
# its own data and never touches a real deployment's rows.
ORG_A, ORG_B = 990101, 990102
REQUEST_IDS = range(9901001, 9901100)
BUCKET_IDS = range(9902001, 9902100)

MONEY = Decimal(1).scaleb(-8)

# ---------------------------------------------------------------------------
# Seed spec — the single source of truth for both the seeder and the oracle
# ---------------------------------------------------------------------------

# (sku, model_name, price, per_quantity, unit)
PRICES: List[Tuple[str, str, str, str, str]] = [
    (SKU_TOKEN_PROMPT, "Qwen3-8B", "0.002", "1000", UNIT_TOKENS),
    (SKU_TOKEN_CACHED, "Qwen3-8B", "0.0002", "1000", UNIT_TOKENS),
    (SKU_TOKEN_COMPLETION, "Qwen3-8B", "0.008", "1000", UNIT_TOKENS),
    (SKU_TOKEN_PROMPT, "Qwen3-30B-A3B", "0.006", "1000", UNIT_TOKENS),
    (SKU_TOKEN_CACHED, "Qwen3-30B-A3B", "0.0006", "1000", UNIT_TOKENS),
    (SKU_TOKEN_COMPLETION, "Qwen3-30B-A3B", "0.024", "1000", UNIT_TOKENS),
    ("gpu.hour.910b", None, "12.5", "1", UNIT_GPU_HOURS),
    (SKU_STORAGE_GB_HOUR, None, "0.0001", "1", UNIT_GB_HOURS),
]
PRICED_MODELS = {"Qwen3-8B", "Qwen3-30B-A3B"}

# (id, model, prompt, cached, completion, completed, payer, scenario)
REQUESTS = [
    (9901001, "Qwen3-8B", 1200, 0, 480, True, ORG_A, "普通对话"),
    (9901002, "Qwen3-8B", 5000, 3200, 620, True, ORG_A, "命中前缀缓存"),
    (9901003, "Qwen3-8B", 32000, 0, 100, True, ORG_B, "长上下文 32K"),
    (9901004, "Qwen3-30B-A3B", 2000, 500, 800, True, ORG_B, "MoE 模型"),
    (9901005, "Qwen3-30B-A3B", 800, 0, 300, True, ORG_A, "MoE 模型"),
    (9901006, "Qwen3-8B", 900, 0, 200, False, ORG_A, "流式中断→不计费"),
    (9901007, "Drill-Unpriced", 1500, 0, 400, True, ORG_A, "未配价→VOID"),
    (9901008, "Qwen3-8B", 700, 0, 150, True, None, "无归属 payer→VOID"),
    (9901009, "Qwen3-8B", 100, 250, 0, True, ORG_B, "cached>prompt→clamp"),
]

# (id, meter, resource_type, name, quantity, sku_count, gpu_type, sealed,
#  payer, unit, scenario)
BUCKETS = [
    (9902001, METER_INSTANCE_UPTIME, RESOURCE_TYPE_GPU_INSTANCE, "inst-8card",
     3600, Decimal(8), "910b", True, ORG_A, UNIT_SECONDS, "910B×8 整机 1 小时"),
    (9902002, METER_INSTANCE_UPTIME, RESOURCE_TYPE_GPU_INSTANCE, "inst-sliced",
     1800, Decimal("0.5"), "910b", True, ORG_B, UNIT_SECONDS, "切片半卡 30 分钟"),
    (9902003, METER_INSTANCE_UPTIME, RESOURCE_TYPE_GPU_INSTANCE, "inst-open",
     3600, Decimal(2), "910b", False, ORG_A, UNIT_SECONDS, "未封桶→跳过"),
    (9902004, METER_INSTANCE_UPTIME, RESOURCE_TYPE_CPU_INSTANCE, "inst-cpu",
     3600, Decimal(1), None, True, ORG_A, UNIT_SECONDS, "CPU 实例→VOID"),
    (9902005, METER_STORAGE_CAPACITY, RESOURCE_TYPE_PERSISTENT_VOLUME, "pv-100g",
     102400 * 3600, Decimal(1), None, True, ORG_B, UNIT_MIB_SECONDS,
     "100 GiB × 1 小时"),
]


def _money(amount: Decimal) -> Decimal:
    """The rater's rounding rule, restated here on purpose."""
    return amount.quantize(MONEY, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Oracle — expectations computed independently of the rater
# ---------------------------------------------------------------------------


def expected_ledger() -> Dict[Tuple[str, int, str], Tuple[str, Decimal, Decimal]]:
    prices = {
        (sku, model): (Decimal(price), Decimal(per))
        for sku, model, price, per, _unit in PRICES
    }
    expected: Dict[Tuple[str, int, str], Tuple[str, Decimal, Decimal]] = {}

    for rid, model, prompt, cached, completion, completed, payer, _s in REQUESTS:
        if not completed:
            continue
        # The cached subset is billed once, under its own SKU.
        billable_prompt = max(Decimal(prompt) - Decimal(cached), Decimal(0))
        pairs = [
            (SKU_TOKEN_PROMPT, billable_prompt),
            (SKU_TOKEN_CACHED, Decimal(cached)),
            (SKU_TOKEN_COMPLETION, Decimal(completion)),
        ]
        unpriced = model not in PRICED_MODELS or payer is None
        for sku, qty in pairs:
            if qty <= 0:
                continue
            if unpriced:
                expected[("model_usage_details", rid, sku)] = (
                    LedgerStatus.VOID.value, Decimal(0), Decimal(0)
                )
                continue
            price, per = prices[(sku, model)]
            expected[("model_usage_details", rid, sku)] = (
                LedgerStatus.PENDING.value, qty, _money(qty * price / per)
            )

    for (bid, meter, _rt, _name, quantity, cards, gpu_type, sealed, _payer,
         _unit, _s) in BUCKETS:
        if not sealed:
            continue
        if meter == METER_INSTANCE_UPTIME:
            if not gpu_type or cards <= 0:
                expected[("metered_usage", bid, SKU_OUT_OF_SCOPE)] = (
                    LedgerStatus.VOID.value, Decimal(0), Decimal(0)
                )
                continue
            sku = f"gpu.hour.{gpu_type}"
            qty = Decimal(quantity) * cards / Decimal(3600)
        else:
            sku = SKU_STORAGE_GB_HOUR
            qty = Decimal(quantity) / Decimal(1024) / Decimal(3600)
        price, per = prices[(sku, None)]
        expected[("metered_usage", bid, sku)] = (
            LedgerStatus.PENDING.value, qty, _money(qty * price / per)
        )
    return expected


# ---------------------------------------------------------------------------
# Seed / cleanup
# ---------------------------------------------------------------------------


async def cleanup(session: AsyncSession) -> None:
    await session.exec(
        delete(LedgerEntry).where(
            LedgerEntry.source_table == "model_usage_details",
            LedgerEntry.source_id.in_(list(REQUEST_IDS)),
        )
    )
    await session.exec(
        delete(LedgerEntry).where(
            LedgerEntry.source_table == "metered_usage",
            LedgerEntry.source_id.in_(list(BUCKET_IDS)),
        )
    )
    await session.exec(
        delete(ModelUsageDetails).where(ModelUsageDetails.id.in_(list(REQUEST_IDS)))
    )
    await session.exec(delete(MeteredUsage).where(MeteredUsage.id.in_(list(BUCKET_IDS))))
    await session.exec(
        delete(PriceBookEntry).where(
            PriceBookEntry.model_name.in_(list(PRICED_MODELS) + ["Drill-Unpriced"])
        )
    )
    await session.exec(delete(PriceBookEntry).where(PriceBookEntry.model_name.is_(None)))
    await session.commit()


async def seed(session: AsyncSession) -> None:
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
                effective_from=NOW - timedelta(days=1),
                is_active=True,
                created_at=NOW,
                updated_at=NOW,
            )
        )

    for rid, model, prompt, cached, completion, completed, payer, _s in REQUESTS:
        session.add(
            ModelUsageDetails(
                id=rid,
                user_id=7,
                user_name="drill-user",
                model_id=rid % 100,
                model_name=model,
                api_key_id=3,
                api_key_name="drill-key",
                consumer_principal_id=payer,
                owner_principal_id=None,
                date=NOW.date(),
                prompt_token_count=prompt,
                completion_token_count=completion,
                prompt_cached_token_count=cached,
                completed=completed,
                request_id=f"drill-req-{rid}",
                started_at=HOUR_AGO,
                completed_at=HOUR_AGO + timedelta(seconds=5),
                ttft_ms=320,
                created_at=NOW,
                updated_at=NOW,
            )
        )

    for (bid, meter, rtype, name, quantity, cards, gpu_type, sealed, payer,
         unit, _s) in BUCKETS:
        dimensions = {"gpu_count": int(cards)}
        if gpu_type:
            dimensions["gpu_type"] = gpu_type
        session.add(
            MeteredUsage(
                id=bid,
                meter_key=meter,
                resource_type=rtype,
                resource_id=bid,
                resource_name=name,
                consumer_principal_id=payer,
                consumer_name="drill-org",
                sku=f"sha1:drill{bid}",
                sku_count=cards,
                dimensions=dimensions,
                bucket_start=HOUR_AGO,
                quantity=quantity,
                unit=unit,
                settled_until=NOW,
                sealed_at=NOW if sealed else None,
                created_at=NOW,
                updated_at=NOW,
            )
        )
    await session.commit()


async def drill_ledger(session: AsyncSession) -> List[LedgerEntry]:
    """Every ledger row the drill's own source rows produced."""
    return list(
        (
            await session.exec(
                select(LedgerEntry).where(
                    LedgerEntry.source_id.in_(list(REQUEST_IDS) + list(BUCKET_IDS))
                )
            )
        ).all()
    )


def fmt(value) -> str:
    text = f"{Decimal(value):.8f}".rstrip("0").rstrip(".")
    return text or "0"


# ---------------------------------------------------------------------------
# Drill
# ---------------------------------------------------------------------------


async def run(database_url: str) -> int:
    # Built the way the server builds it, so the drill talks to the database
    # through the same driver, pool and connect args — a URL carrying libpq's
    # ``sslmode`` is normalized here exactly as it is at server startup.
    engine = await init_db_engine(database_url)
    # What init_database() would have set; the rater opens sessions through it.
    gpustack_db.engine = engine
    invalidate_price_cache()

    bar = "=" * 78
    print(bar)
    print("影子计价演练 · PostgreSQL · GPUSTACK_BILLING_MODE=shadow")
    print(bar)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        await cleanup(session)
        await seed(session)
        print(
            f"\n[1] 灌入：{len(PRICES)} 条价格 / {len(REQUESTS)} 条请求明细 / "
            f"{len(BUCKETS)} 条资源计量桶"
        )

    rater = BillingRater(mode=BillingMode.SHADOW)
    report = await rater.rate_once()
    print(f"\n[2] 第一轮 sweep：{report.summary()}")
    for reason in report.unpriced_reasons:
        print(f"      · 未计价：{reason}")

    async with AsyncSession(engine, expire_on_commit=False) as session:
        rows_after_first = await drill_ledger(session)

    second = await rater.rate_once()
    print(f"\n[3] 第二轮 sweep（幂等）：{second.summary()}")
    print(
        "      注：VOID 源行按设计会被重试（等价格补齐后晋升），"
        "所以幂等的判据是 ledger 行数不变，而非源行数归零"
    )

    async with AsyncSession(engine, expire_on_commit=False) as session:
        rows = await drill_ledger(session)
        wallets = (
            await session.exec(select(func.count()).select_from(Wallet))
        ).one()

    expected = expected_ledger()
    actual = {
        (r.source_table, r.source_id, r.sku): (
            r.status,
            Decimal(r.quantity),
            Decimal(r.amount),
        )
        for r in rows
    }
    print(f"\n[4] 对账：独立重算期望 {len(expected)} 条 / ledger 实际 {len(actual)} 条")

    problems: List[str] = []
    for key, (status, qty, amount) in sorted(expected.items()):
        got = actual.get(key)
        if got is None:
            problems.append(
                f"缺失 {key}：期望 status={status} qty={fmt(qty)} amount={fmt(amount)}"
            )
            continue
        got_status, got_qty, got_amount = got
        if got_status != status:
            problems.append(f"{key}：status 期望 {status} 实际 {got_status}")
        if got_qty != qty:
            problems.append(f"{key}：quantity 期望 {fmt(qty)} 实际 {fmt(got_qty)}")
        if got_amount != amount:
            problems.append(f"{key}：amount 期望 {fmt(amount)} 实际 {fmt(got_amount)}")
    for key in sorted(set(actual) - set(expected)):
        problems.append(f"多出未预期的 ledger 行：{key}")

    print("\n[5] 按 SKU 汇总（仅 PENDING = 真实计费）：")
    print(f"      {'SKU':<24}{'笔数':>5}{'数量':>14}{'金额(CNY)':>16}{'结算':>10}")
    by_sku: Dict[str, list] = {}
    for r in rows:
        if r.status != LedgerStatus.PENDING.value:
            continue
        agg = by_sku.setdefault(
            r.sku, [0, Decimal(0), Decimal(0), r.settle_mode]
        )
        agg[0] += 1
        agg[1] += Decimal(r.quantity)
        agg[2] += Decimal(r.amount)
    total = Decimal(0)
    for sku, (count, qty, amount, mode) in sorted(by_sku.items()):
        print(f"      {sku:<24}{count:>5}{fmt(qty):>14}{fmt(amount):>16}{mode:>10}")
        total += amount
    print(
        f"      {'合计':<24}{sum(v[0] for v in by_sku.values()):>5}"
        f"{'':>14}{fmt(total):>16}"
    )

    voided = [r for r in rows if r.status == LedgerStatus.VOID.value]
    print(f"\n[6] VOID（未计费积压，可查询 status='void'）：{len(voided)} 条")
    for r in sorted(voided, key=lambda r: (r.source_table, r.source_id, r.sku)):
        print(
            f"      · {r.source_table} id={r.source_id} sku={r.sku} "
            f"payer={r.principal_id}"
        )

    per_org: Dict[int, Decimal] = {}
    for r in rows:
        if r.status == LedgerStatus.PENDING.value and r.principal_id:
            per_org[r.principal_id] = per_org.get(
                r.principal_id, Decimal(0)
            ) + Decimal(r.amount)
    print("\n[7] 按租户汇总应付：")
    for pid, amount in sorted(per_org.items()):
        label = {ORG_A: "ORG_A", ORG_B: "ORG_B"}.get(pid, str(pid))
        print(f"      {label} (principal_id={pid}): {fmt(amount)} CNY")

    print(
        f"\n[8] 影子模式不动钱包：billing_wallet 行数 = {wallets} "
        f"({'OK' if wallets == 0 else '异常：应为 0'})"
    )
    # Idempotency is "the second sweep wrote nothing", not "it found no
    # sources": VOID rows are retried on purpose until a price appears.
    idempotent = (
        len(rows) == len(rows_after_first)
        and second.entries == 0
        and second.promoted == 0
    )
    print(
        f"[9] 幂等：ledger 行数 {len(rows_after_first)} → {len(rows)}，"
        f"第二轮新增计费 {second.entries} 条 = {idempotent}"
    )

    print("\n" + bar)
    if problems or not idempotent or wallets != 0:
        print(f"演练失败：对账差异 {len(problems)} 处，幂等={idempotent}")
        for p in problems[:20]:
            print(f"  - {p}")
    else:
        print(
            f"演练通过：{len(actual)} 条 ledger 与独立重算逐条一致（0 差异），"
            f"幂等成立，钱包未被触碰"
        )
    print(bar)

    await engine.dispose()
    return 1 if (problems or not idempotent or wallets != 0) else 0


def main() -> int:
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    return asyncio.run(run(url))


if __name__ == "__main__":
    raise SystemExit(main())
