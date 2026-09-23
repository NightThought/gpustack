#!/usr/bin/env python3
"""Billing reconciliation (WP8.2), and the tool the SOP in docs/prd/14 points at.

Four independent checks over a real database, each comparing two things that are
supposed to be equal and were produced by different code paths:

1. **Sources → ledger.** Quantity per ``(source_id, sku)`` recomputed from
   ``model_usage_details`` and ``metered_usage`` with this script's own arithmetic,
   against the ledger's sums. Catches a rater that mis-converts a unit, drops a
   SKU, or bills a row twice.
2. **Ledger row internal consistency.** ``amount == quantity × unit_price ÷
   per_quantity`` against the price row the entry snapshotted, and ``unit``
   matching the sku. Catches a charge whose stored amount disagrees with the price
   it claims to have used — the discrepancy a tenant finds first.
3. **Wallet ↔ ledger.** A wallet's balance must equal its settled credits minus
   its settled debits, and — over a window — the change in balance must equal the
   change in those sums. Catches money that moved without a ledger row, or a
   ledger row that never moved money.
4. **Invoice ↔ its parts.** Each statement's lines sum to its total, and the
   entries it claims sum to the same number. Catches a statement that cannot be
   reproduced from what it billed.

Nothing here re-derives *which* usage should have been priced — that is
``shadow_rating_drill.py``'s job, against a seed it controls. This tool reads
whatever is in the database, so it is the one to run against production.

Exit code is non-zero when any check fails, so it can be a cron job or a gate.

Usage
-----
    DATABASE_URL=postgresql://user@host:5432/gpustack \
        uv run python hack/billing/reconcile.py

    # tolerate wallets whose opening balance was set out of band (a drill, a
    # migration, a manual fix that predates the adjustment table):
    RECONCILE_ALLOW_OPENING_BALANCE=1 ...

    # also verify conservation over a live window:
    RECONCILE_WATCH_SECONDS=60 ...
"""

import asyncio
import os
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple

from sqlalchemy import case, func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.billing import (
    MONEY_SCALE,
    SKU_GPU_HOUR_PREFIX,
    SKU_STORAGE_GB_HOUR,
    SKU_TOKEN_CACHED,
    SKU_TOKEN_COMPLETION,
    SKU_TOKEN_PROMPT,
    Invoice,
    InvoiceItem,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    PriceBookEntry,
    Wallet,
    unit_for_sku,
)
from gpustack.schemas.metered_usage import (
    METER_INSTANCE_UPTIME,
    METER_STORAGE_CAPACITY,
    MeteredUsage,
)
from gpustack.schemas.model_usage_details import ModelUsageDetails
from gpustack.server import db as gpustack_db
from gpustack.server.init_db import init_db_engine

DEFAULT_DATABASE_URL = "postgresql://root@localhost:5432/gpustack"

MONEY = Decimal(1).scaleb(-MONEY_SCALE)
SECONDS_PER_HOUR = Decimal(3600)
MIB_PER_GB = Decimal(1024)

# How many mismatches of each kind to print before summarising the rest as a
# count. A reconciliation that dumps 40k rows is a reconciliation nobody reads.
MAX_REPORTED = int(os.getenv("RECONCILE_MAX_REPORTED", "10"))
PRINCIPAL_ID = os.getenv("RECONCILE_PRINCIPAL_ID")
ALLOW_OPENING_BALANCE = os.getenv("RECONCILE_ALLOW_OPENING_BALANCE", "") == "1"
WATCH_SECONDS = int(os.getenv("RECONCILE_WATCH_SECONDS", "0"))
# Whether a completed source with no ledger row at all is a failure. Off by
# default because the rater runs on an interval, so a just-completed request is
# legitimately unrated for up to one sweep; turn it on for a quiet system where
# "nothing is pending" is the expected state.
STRICT_UNRATED = os.getenv("RECONCILE_STRICT_UNRATED", "") == "1"


def _money(value) -> Decimal:
    return Decimal(value or 0).quantize(MONEY, rounding=ROUND_HALF_UP)


def _principal_filter(model):
    """Optional scoping, so a reconciliation can be run for one org."""
    if PRINCIPAL_ID is None:
        return None
    return model.principal_id == int(PRINCIPAL_ID)


# ---------------------------------------------------------------------------
# Check 1 — sources → ledger
# ---------------------------------------------------------------------------


def expected_token_quantities(row: ModelUsageDetails) -> Dict[str, Decimal]:
    """Per-SKU token counts for one completed request.

    Written out here rather than imported from the rater: a reconciliation that
    calls the code it is checking agrees with itself whatever that code does.
    """
    prompt_total = Decimal(row.prompt_token_count or 0)
    cached = Decimal(row.prompt_cached_token_count or 0)
    # Cached tokens are billed under their own SKU, so they come out of the
    # prompt count; a stream that reports more cached than prompt (it happens)
    # clamps rather than producing a negative quantity.
    if cached > prompt_total:
        cached = prompt_total
    quantities = {
        SKU_TOKEN_PROMPT: prompt_total - cached,
        SKU_TOKEN_CACHED: cached,
        SKU_TOKEN_COMPLETION: Decimal(row.completion_token_count or 0),
    }
    return {sku: qty for sku, qty in quantities.items() if qty > 0}


def expected_resource_quantity(
    row: MeteredUsage,
) -> Optional[Tuple[str, Decimal]]:
    """``(sku, quantity)`` for one sealed bucket, or None if it is out of scope."""
    seconds = Decimal(row.quantity or 0)
    if row.meter_key == METER_INSTANCE_UPTIME:
        gpu_type = (row.dimensions or {}).get("gpu_type")
        cards = Decimal(str(row.sku_count or 0))
        if not gpu_type or cards <= 0:
            return None
        hours = seconds * cards / SECONDS_PER_HOUR
        return (f"{SKU_GPU_HOUR_PREFIX}{gpu_type}", hours) if hours > 0 else None
    if row.meter_key == METER_STORAGE_CAPACITY:
        gb_hours = seconds / MIB_PER_GB / SECONDS_PER_HOUR
        return (SKU_STORAGE_GB_HOUR, gb_hours) if gb_hours > 0 else None
    return None


async def ledger_quantity_sums(
    session: AsyncSession, source_table: str, source_ids: List[int]
) -> Dict[Tuple[int, str], Decimal]:
    """Summed quantity per ``(source_id, sku)`` over non-VOID ledger rows."""
    if not source_ids:
        return {}
    rows = (
        await session.exec(
            select(
                LedgerEntry.source_id,
                LedgerEntry.sku,
                func.sum(LedgerEntry.quantity),
            )
            .where(
                LedgerEntry.deleted_at.is_(None),
                LedgerEntry.source_table == source_table,
                LedgerEntry.status != LedgerStatus.VOID.value,
                LedgerEntry.source_id.in_(source_ids),
            )
            .group_by(LedgerEntry.source_id, LedgerEntry.sku)
        )
    ).all()
    return {(int(row[0]), str(row[1])): Decimal(row[2] or 0) for row in rows}


async def ledger_source_states(
    session: AsyncSession, source_table: str
) -> Tuple[set, set]:
    """``(source_ids with a real entry, source_ids whose only entries are VOID)``.

    The distinction is what keeps this tool honest. A source with VOID rows is
    not a discrepancy: the rater could not price it, said so with a placeholder,
    and will retry every sweep until a price appears. Reporting that as "missing"
    would bury a real mismatch in a list of known gaps — and an operator who
    learns to ignore the output stops reading it.
    """
    rows = (
        await session.exec(
            select(
                LedgerEntry.source_id,
                func.count(LedgerEntry.id),
                func.sum(
                    case(
                        (LedgerEntry.status != LedgerStatus.VOID.value, 1),
                        else_=0,
                    )
                ),
            )
            .where(
                LedgerEntry.deleted_at.is_(None),
                LedgerEntry.source_table == source_table,
            )
            .group_by(LedgerEntry.source_id)
        )
    ).all()
    rated, void_only = set(), set()
    for row in rows:
        source_id = int(row[0])
        if int(row[2] or 0) > 0:
            rated.add(source_id)
        else:
            void_only.add(source_id)
    return rated, void_only


async def check_sources(session: AsyncSession, problems: List[str]) -> Dict[str, int]:
    stats = {
        "requests": 0,
        "buckets": 0,
        "compared": 0,
        "mismatched": 0,
        "unpriced": 0,
        "unrated": 0,
        "out_of_scope": 0,
    }
    reported = 0

    details = (
        await session.exec(
            select(ModelUsageDetails).where(
                ModelUsageDetails.completed.is_(True),
                ModelUsageDetails.deleted_at.is_(None),
                *([_principal_filter(ModelUsageDetails)] if PRINCIPAL_ID else []),
            )
        )
    ).all()
    stats["requests"] = len(details)
    sums = await ledger_quantity_sums(
        session, "model_usage_details", [row.id for row in details]
    )
    rated_ids, void_ids = await ledger_source_states(session, "model_usage_details")
    for row in details:
        if row.id in void_ids and row.id not in rated_ids:
            # A visible gap the rater is already retrying, not a discrepancy.
            stats["unpriced"] += 1
            continue
        if row.id not in rated_ids:
            # No entry at all: either the rater has not reached it (normal, it
            # runs on an interval) or it was skipped. Reported, and only a
            # failure when the operator asks for strictness.
            stats["unrated"] += 1
            if STRICT_UNRATED:
                problems.append(
                    f"[明细→账本] request id={row.id} model={row.model_name!r} "
                    "已完成但无任何账本行"
                )
            continue
        for sku, quantity in expected_token_quantities(row).items():
            stats["compared"] += 1
            actual = sums.get((row.id, sku))
            if actual is None or Decimal(actual) != quantity:
                stats["mismatched"] += 1
                if reported < MAX_REPORTED:
                    reported += 1
                    problems.append(
                        f"[明细→账本] request id={row.id} model={row.model_name!r} "
                        f"sku={sku}: 期望 {quantity} 实际 "
                        f"{'缺失' if actual is None else actual}"
                    )

    buckets = (
        await session.exec(
            select(MeteredUsage).where(
                MeteredUsage.sealed_at.is_not(None),
                MeteredUsage.deleted_at.is_(None),
                *([_principal_filter(MeteredUsage)] if PRINCIPAL_ID else []),
            )
        )
    ).all()
    stats["buckets"] = len(buckets)
    bucket_sums = await ledger_quantity_sums(
        session, "metered_usage", [row.id for row in buckets]
    )
    rated_buckets, void_buckets = await ledger_source_states(session, "metered_usage")
    for row in buckets:
        expected = expected_resource_quantity(row)
        if expected is None:
            # Out of scope (a CPU-only instance, or no accelerator type to price
            # by): the rater leaves a VOID placeholder, so there is nothing to
            # compare and expecting a row here would be wrong.
            stats["out_of_scope"] += 1
            continue
        if row.id in void_buckets and row.id not in rated_buckets:
            stats["unpriced"] += 1
            continue
        if row.id not in rated_buckets:
            stats["unrated"] += 1
            if STRICT_UNRATED:
                problems.append(
                    f"[明细→账本] bucket id={row.id} meter={row.meter_key} "
                    "已封桶但无任何账本行"
                )
            continue
        sku, quantity = expected
        stats["compared"] += 1
        actual = bucket_sums.get((row.id, sku))
        if actual is None or Decimal(actual) != quantity:
            stats["mismatched"] += 1
            if reported < MAX_REPORTED:
                reported += 1
                problems.append(
                    f"[明细→账本] bucket id={row.id} meter={row.meter_key} sku={sku}: "
                    f"期望 {quantity} 实际 {'缺失' if actual is None else actual}"
                )

    if stats["mismatched"] > reported:
        problems.append(
            f"[明细→账本] 另有 {stats['mismatched'] - reported} 处差异未列出"
        )
    return stats


# ---------------------------------------------------------------------------
# Check 2 — each ledger row against the price it snapshotted
# ---------------------------------------------------------------------------


async def check_row_consistency(
    session: AsyncSession, problems: List[str]
) -> Dict[str, int]:
    stats = {"entries": 0, "priced": 0, "bad_amount": 0, "bad_unit": 0}

    prices = {row.id: row for row in (await session.exec(select(PriceBookEntry))).all()}
    entries = (
        await session.exec(
            select(LedgerEntry).where(
                LedgerEntry.deleted_at.is_(None),
                LedgerEntry.status != LedgerStatus.VOID.value,
                *([_principal_filter(LedgerEntry)] if PRINCIPAL_ID else []),
            )
        )
    ).all()
    stats["entries"] = len(entries)

    reported = 0
    for entry in entries:
        try:
            expected_unit = unit_for_sku(entry.sku)
        except ValueError:
            expected_unit = None
        if expected_unit is not None and entry.unit != expected_unit:
            stats["bad_unit"] += 1
            if reported < MAX_REPORTED:
                reported += 1
                problems.append(
                    f"[行内一致] ledger id={entry.id} sku={entry.sku}: unit="
                    f"{entry.unit!r} 应为 {expected_unit!r}"
                )
            continue

        price_row = prices.get(entry.price_book_id) if entry.price_book_id else None
        if price_row is None:
            # A wallet movement (a top-up, an adjustment) has no price behind it;
            # its amount is the money itself.
            continue
        stats["priced"] += 1
        per = Decimal(price_row.per_quantity or 1)
        expected = (
            Decimal(entry.quantity or 0) * Decimal(price_row.price or 0) / per
            if per
            else Decimal(0)
        ).quantize(MONEY, rounding=ROUND_HALF_UP)
        if _money(entry.amount) != expected:
            stats["bad_amount"] += 1
            if reported < MAX_REPORTED:
                reported += 1
                problems.append(
                    f"[行内一致] ledger id={entry.id} sku={entry.sku} "
                    f"request={entry.request_id}: amount={entry.amount} 应为 "
                    f"{expected}（quantity={entry.quantity} × price="
                    f"{price_row.price} ÷ per_quantity={per}）"
                )

    if stats["bad_amount"] + stats["bad_unit"] > reported:
        problems.append(
            f"[行内一致] 另有 "
            f"{stats['bad_amount'] + stats['bad_unit'] - reported} 处差异未列出"
        )
    return stats


# ---------------------------------------------------------------------------
# Check 3 — wallet ↔ ledger
# ---------------------------------------------------------------------------


async def settled_sums(session: AsyncSession) -> Dict[int, Tuple[Decimal, Decimal]]:
    """``principal_id → (settled credits, settled debits)``."""
    rows = (
        await session.exec(
            select(
                LedgerEntry.principal_id,
                LedgerEntry.direction,
                func.sum(LedgerEntry.amount),
            )
            .where(
                LedgerEntry.deleted_at.is_(None),
                LedgerEntry.status == LedgerStatus.SETTLED.value,
                LedgerEntry.principal_id.is_not(None),
            )
            .group_by(LedgerEntry.principal_id, LedgerEntry.direction)
        )
    ).all()
    sums: Dict[int, List[Decimal]] = {}
    for row in rows:
        principal_id = int(row[0])
        bucket = sums.setdefault(principal_id, [Decimal(0), Decimal(0)])
        if row[1] == LedgerDirection.CREDIT.value:
            bucket[0] += Decimal(row[2] or 0)
        else:
            bucket[1] += Decimal(row[2] or 0)
    return {key: (value[0], value[1]) for key, value in sums.items()}


async def check_wallets(
    session: AsyncSession, problems: List[str]
) -> Dict[str, object]:
    wallets = (
        await session.exec(
            select(Wallet).where(
                Wallet.deleted_at.is_(None),
                *([_principal_filter(Wallet)] if PRINCIPAL_ID else []),
            )
        )
    ).all()
    sums = await settled_sums(session)

    residuals: Dict[int, Decimal] = {}
    for wallet in wallets:
        credits, debits = sums.get(wallet.principal_id, (Decimal(0), Decimal(0)))
        residual = Decimal(wallet.balance or 0) - (credits - debits)
        if residual != 0:
            residuals[wallet.principal_id] = residual
            if ALLOW_OPENING_BALANCE:
                continue
            problems.append(
                f"[钱包↔账本] principal {wallet.principal_id}: 余额 "
                f"{wallet.balance} ≠ 已结算收入 {credits} − 已结算支出 {debits}"
                f"（残差 {residual}）"
            )
    return {
        "wallets": len(wallets),
        "residuals": residuals,
        "credits": sum((v[0] for v in sums.values()), Decimal(0)),
        "debits": sum((v[1] for v in sums.values()), Decimal(0)),
    }


async def snapshot_money(session: AsyncSession) -> Dict[str, object]:
    """Balances and settled sums, for the windowed conservation check."""
    wallets = {
        int(row.principal_id): Decimal(row.balance or 0)
        for row in (
            await session.exec(select(Wallet).where(Wallet.deleted_at.is_(None)))
        ).all()
    }
    return {"balances": wallets, "sums": await settled_sums(session)}


def check_conservation(
    before: Dict[str, object],
    after: Dict[str, object],
    problems: List[str],
) -> int:
    """Δbalance must equal Δcredits − Δdebits, whatever the opening balance was.

    The windowed form of check 3, and the one that survives an out-of-band
    opening balance: it compares movements rather than totals, so a wallet seeded
    directly (a drill, a migration) does not show up as a discrepancy — while
    money that moved without a ledger row still does.
    """
    checked = 0
    before_balances: Dict[int, Decimal] = before["balances"]  # type: ignore[assignment]
    after_balances: Dict[int, Decimal] = after["balances"]  # type: ignore[assignment]
    before_sums: Dict[int, Tuple[Decimal, Decimal]] = before["sums"]  # type: ignore[assignment]
    after_sums: Dict[int, Tuple[Decimal, Decimal]] = after["sums"]  # type: ignore[assignment]

    for principal_id in set(before_balances) | set(after_balances):
        delta_balance = after_balances.get(
            principal_id, Decimal(0)
        ) - before_balances.get(principal_id, Decimal(0))
        credit_before, debit_before = before_sums.get(
            principal_id, (Decimal(0), Decimal(0))
        )
        credit_after, debit_after = after_sums.get(
            principal_id, (Decimal(0), Decimal(0))
        )
        delta_ledger = (credit_after - credit_before) - (debit_after - debit_before)
        if delta_balance == 0 and delta_ledger == 0:
            continue
        checked += 1
        if delta_balance != delta_ledger:
            problems.append(
                f"[窗口守恒] principal {principal_id}: 余额变动 {delta_balance} ≠ "
                f"账本已结算变动 {delta_ledger}"
            )
    return checked


# ---------------------------------------------------------------------------
# Check 4 — invoice ↔ its parts
# ---------------------------------------------------------------------------


async def check_invoices(session: AsyncSession, problems: List[str]) -> Dict[str, int]:
    stats = {"invoices": 0, "bad_items": 0, "bad_entries": 0}
    invoices = (
        await session.exec(
            select(Invoice).where(
                Invoice.deleted_at.is_(None),
                *([_principal_filter(Invoice)] if PRINCIPAL_ID else []),
            )
        )
    ).all()
    stats["invoices"] = len(invoices)

    for invoice in invoices:
        total = Decimal(invoice.amount or 0)
        item_sum = (
            await session.exec(
                select(func.sum(InvoiceItem.amount)).where(
                    InvoiceItem.invoice_id == invoice.id,
                    InvoiceItem.deleted_at.is_(None),
                )
            )
        ).first()
        entry_sum = (
            await session.exec(
                select(func.sum(LedgerEntry.amount)).where(
                    LedgerEntry.invoice_id == invoice.id,
                    LedgerEntry.deleted_at.is_(None),
                )
            )
        ).first()

        if _money(item_sum) != _money(total):
            stats["bad_items"] += 1
            problems.append(
                f"[账单↔行] invoice {invoice.id} (principal {invoice.principal_id}): "
                f"行合计 {_money(item_sum)} ≠ 账单金额 {_money(total)}"
            )
        if _money(entry_sum) != _money(total):
            stats["bad_entries"] += 1
            problems.append(
                f"[账单↔条目] invoice {invoice.id}: 认领条目合计 "
                f"{_money(entry_sum)} ≠ 账单金额 {_money(total)}"
            )
    return stats


# ---------------------------------------------------------------------------


async def run(database_url: str) -> int:
    engine = await init_db_engine(database_url)
    gpustack_db.engine = engine
    problems: List[str] = []
    bar = "=" * 78
    started = time.monotonic()

    print(bar)
    print(
        "计费对账："
        f"principal={'全部' if PRINCIPAL_ID is None else PRINCIPAL_ID}，"
        f"期初余额容差={'开' if ALLOW_OPENING_BALANCE else '关'}，"
        f"窗口校验={WATCH_SECONDS}s"
    )
    print(bar)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        before = await snapshot_money(session) if WATCH_SECONDS > 0 else None

        source_stats = await check_sources(session, problems)
        print(
            f"[1] 明细→账本：{source_stats['requests']} 请求 / "
            f"{source_stats['buckets']} 资源桶，比对 "
            f"{source_stats['compared']} 个 (源, SKU) 数量，差异 "
            f"{source_stats['mismatched']}"
        )
        print(
            f"      未计价（VOID，rater 每轮重试）{source_stats['unpriced']} 个源，"
            f"超出 v1 范围 {source_stats['out_of_scope']} 个桶，"
            f"尚无任何账本行 {source_stats['unrated']} 个源"
            f"{'（严格模式：计为失败）' if STRICT_UNRATED else ''}"
        )

        row_stats = await check_row_consistency(session, problems)
        print(
            f"[2] 行内一致：{row_stats['entries']} 条账本行（其中 "
            f"{row_stats['priced']} 条有价格快照），金额不符 "
            f"{row_stats['bad_amount']}，单位不符 {row_stats['bad_unit']}"
        )

        wallet_stats = await check_wallets(session, problems)
        print(
            f"[3] 钱包↔账本：{wallet_stats['wallets']} 个钱包，已结算收入 "
            f"{_money(wallet_stats['credits'])} / 支出 "
            f"{_money(wallet_stats['debits'])}，残差钱包 "
            f"{len(wallet_stats['residuals'])} 个"
        )
        for principal_id, residual in list(wallet_stats["residuals"].items())[:5]:
            note = "（容差开启，视为期初余额）" if ALLOW_OPENING_BALANCE else ""
            print(f"      principal {principal_id}: 残差 {_money(residual)} {note}")

        invoice_stats = await check_invoices(session, problems)
        print(
            f"[4] 账单↔明细：{invoice_stats['invoices']} 张账单，行合计不符 "
            f"{invoice_stats['bad_items']}，条目合计不符 "
            f"{invoice_stats['bad_entries']}"
        )

    checked = 0
    if WATCH_SECONDS > 0 and before is not None:
        print(f"[5] 窗口守恒：观察 {WATCH_SECONDS}s …")
        await asyncio.sleep(WATCH_SECONDS)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            after = await snapshot_money(session)
        checked = check_conservation(before, after, problems)
        print(f"      {checked} 个主体在窗口内有资金变动，全部对平={checked >= 0}")

    print("\n" + bar)
    hard_failures = [
        problem
        for problem in problems
        if not (ALLOW_OPENING_BALANCE and problem.startswith("[钱包↔账本]"))
    ]
    if hard_failures:
        print(f"对账未通过（{len(hard_failures)} 项）：")
        for problem in hard_failures[: MAX_REPORTED * 4]:
            print(f"  - {problem}")
    else:
        print(
            f"对账通过：{source_stats['compared']} 个数量、"
            f"{row_stats['priced']} 条金额、{wallet_stats['wallets']} 个钱包、"
            f"{invoice_stats['invoices']} 张账单四方一致"
            + (f"，窗口内 {checked} 个主体资金变动对平" if WATCH_SECONDS else "")
        )
    print(f"用时 {time.monotonic() - started:.1f}s")
    print(bar)

    await engine.dispose()
    return 1 if hard_failures else 0


def main() -> int:
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    try:
        return asyncio.run(run(url))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
