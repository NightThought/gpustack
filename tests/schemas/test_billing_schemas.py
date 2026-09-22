"""Billing schema invariants (WP1.5 of docs/prd/13-计费引擎开发TODO.md).

These lock the properties the rest of the billing engine is allowed to assume,
so a later edit cannot silently break them:

* every billing table is registered under its ``billing_*`` name (Alembic
  autogenerate discovers tables through ``SQLModel.metadata``, so an
  unregistered model produces no DDL and fails at runtime, not at migration
  time);
* the idempotency and one-per-subject constraints exist — the rater re-runs
  over the same usage rows by design, and only ``uq_ledger_source_sku`` stands
  between a retry and a double charge;
* money columns stay ``Numeric(20, 8)`` — a switch to ``Float`` would make
  reconciliation non-deterministic;
* the WP0 hybrid settlement decision is encoded once, in
  ``settle_mode_for_sku``, and every SKU family maps to the path its source
  table can support.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import Numeric
from sqlmodel import SQLModel

from gpustack.schemas.billing import (
    MONEY_PRECISION,
    MONEY_SCALE,
    SKU_GPU_HOUR_PREFIX,
    SKU_STORAGE_GB_HOUR,
    SKU_TOKEN_CACHED,
    SKU_TOKEN_COMPLETION,
    SKU_TOKEN_PROMPT,
    UNIT_GB_HOURS,
    UNIT_GPU_HOURS,
    UNIT_TOKENS,
    BillingSession,
    Invoice,
    InvoiceItem,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    PriceBookEntry,
    Quota,
    QuotaScope,
    Redemption,
    RedemptionStatus,
    SettleMode,
    Wallet,
    is_token_sku,
    settle_mode_for_sku,
    unit_for_sku,
)

# table name -> model, the registration contract with SQLModel.metadata.
BILLING_TABLES = {
    "billing_wallet": Wallet,
    "billing_price_book": PriceBookEntry,
    "billing_ledger": LedgerEntry,
    "billing_session": BillingSession,
    "billing_invoice": Invoice,
    "billing_invoice_item": InvoiceItem,
    "billing_quota": Quota,
    "billing_redemption": Redemption,
}

# Money-bearing columns that must never drift off Numeric(20, 8).
MONEY_COLUMNS = {
    "billing_wallet": ("balance", "frozen"),
    "billing_price_book": ("price", "per_quantity"),
    "billing_ledger": ("quantity", "unit_price", "amount"),
    "billing_session": ("estimate", "actual"),
    "billing_invoice": ("amount",),
    "billing_invoice_item": ("quantity", "amount"),
    "billing_quota": ("limit_value", "used"),
    "billing_redemption": ("amount",),
}

# constraint name -> (table, columns), the idempotency / uniqueness contract.
EXPECTED_CONSTRAINTS = {
    "uq_ledger_source_sku": ("billing_ledger", {"source_table", "source_id", "sku"}),
    "uq_wallet_principal": ("billing_wallet", {"principal_id"}),
    "uq_invoice_principal_period": (
        "billing_invoice",
        {"principal_id", "period_start"},
    ),
    "uq_quota_scope_limit": (
        "billing_quota",
        {
            "scope",
            "principal_id",
            "user_id",
            "api_key_id",
            "model_name",
            "limit_type",
        },
    ),
}


@pytest.mark.parametrize("table_name,model", sorted(BILLING_TABLES.items()))
def test_table_registered_with_expected_name(table_name, model):
    assert model.__tablename__ == table_name
    assert table_name in SQLModel.metadata.tables


def test_expected_constraints_present():
    found = {}
    for table_name in BILLING_TABLES:
        table = SQLModel.metadata.tables[table_name]
        for constraint in table.constraints:
            name = getattr(constraint, "name", None)
            if name in EXPECTED_CONSTRAINTS:
                found[name] = (
                    table_name,
                    {column.name for column in constraint.columns},
                )
    assert found == EXPECTED_CONSTRAINTS


@pytest.mark.parametrize("table_name,columns", sorted(MONEY_COLUMNS.items()))
def test_money_columns_are_numeric_with_locked_precision(table_name, columns):
    table = SQLModel.metadata.tables[table_name]
    for column_name in columns:
        column = table.columns[column_name]
        assert isinstance(column.type, Numeric), (
            f"{table_name}.{column_name} must stay Numeric — a float money "
            "column makes reconciliation non-deterministic"
        )
        assert (column.type.precision, column.type.scale) == (
            MONEY_PRECISION,
            MONEY_SCALE,
        )


def test_ledger_provenance_columns_are_required():
    """The idempotency key and the pricing snapshot cannot be optional.

    A NULL in any of these would let two entries for one usage row both pass a
    uniqueness check, or let an entry be written that cannot be re-priced for
    audit.
    """
    table = SQLModel.metadata.tables["billing_ledger"]
    for column_name in (
        "source_table",
        "source_id",
        "sku",
        "quantity",
        "unit_price",
        "amount",
        "settle_mode",
        "occurred_at",
    ):
        assert table.columns[column_name].nullable is False, column_name


@pytest.mark.parametrize(
    "sku,expected",
    [
        (SKU_TOKEN_PROMPT, SettleMode.REALTIME),
        (SKU_TOKEN_COMPLETION, SettleMode.REALTIME),
        # The WP0 decision that cached tokens price separately only matters if
        # they also settle on the realtime path with the rest of the request.
        (SKU_TOKEN_CACHED, SettleMode.REALTIME),
        (SKU_GPU_HOUR_PREFIX + "910b", SettleMode.DEFERRED),
        (SKU_STORAGE_GB_HOUR, SettleMode.DEFERRED),
    ],
)
def test_settle_mode_for_sku(sku, expected):
    assert settle_mode_for_sku(sku) is expected


@pytest.mark.parametrize(
    "sku,expected",
    [
        (SKU_TOKEN_PROMPT, True),
        (SKU_TOKEN_CACHED, True),
        (SKU_GPU_HOUR_PREFIX + "910c", False),
        (SKU_STORAGE_GB_HOUR, False),
    ],
)
def test_is_token_sku(sku, expected):
    assert is_token_sku(sku) is expected


@pytest.mark.parametrize(
    "sku,expected",
    [
        (SKU_TOKEN_PROMPT, UNIT_TOKENS),
        (SKU_GPU_HOUR_PREFIX + "910b", UNIT_GPU_HOURS),
        (SKU_STORAGE_GB_HOUR, UNIT_GB_HOURS),
    ],
)
def test_unit_for_sku(sku, expected):
    assert unit_for_sku(sku) == expected


def test_unit_for_sku_rejects_unknown_sku():
    """An unpriced SKU must fail loudly at rating time, not bill at zero."""
    with pytest.raises(ValueError, match="unknown billing sku"):
        unit_for_sku("model.token.mystery")


def test_enum_values_are_wire_stable():
    """Enum values are persisted as strings; renaming one orphans stored rows."""
    assert LedgerDirection.DEBIT.value == "debit"
    assert LedgerDirection.CREDIT.value == "credit"
    assert LedgerStatus.PENDING.value == "pending"
    assert LedgerStatus.SETTLED.value == "settled"
    assert SettleMode.REALTIME.value == "realtime"
    assert SettleMode.DEFERRED.value == "deferred"
    assert QuotaScope.ORGANIZATION.value == "organization"
    assert RedemptionStatus.ENABLED.value == "enabled"


def test_wallet_defaults_to_empty_and_active():
    wallet = Wallet(principal_id=1)
    assert wallet.balance == Decimal(0)
    assert wallet.frozen == Decimal(0)
    assert wallet.suspended is False
    assert wallet.currency == "CNY"


def test_price_book_entry_defaults_to_open_ended_first_version():
    entry = PriceBookEntry(
        sku=SKU_TOKEN_PROMPT,
        model_name="Qwen3-8B",
        unit=UNIT_TOKENS,
        price=Decimal("0.002"),
        per_quantity=Decimal(1000),
        effective_from=datetime(2026, 9, 22, tzinfo=timezone.utc),
    )
    assert entry.version == 1
    assert entry.is_active is True
    assert entry.effective_to is None


def test_ledger_entry_defaults_to_pending_debit():
    entry = LedgerEntry(
        source_table="model_usage_details",
        source_id=42,
        sku=SKU_TOKEN_PROMPT,
        quantity=Decimal(1000),
        unit=UNIT_TOKENS,
        unit_price=Decimal("0.002"),
        amount=Decimal("0.002"),
        settle_mode=settle_mode_for_sku(SKU_TOKEN_PROMPT),
        occurred_at=datetime(2026, 9, 22, tzinfo=timezone.utc),
    )
    assert entry.direction == LedgerDirection.DEBIT
    assert entry.status == LedgerStatus.PENDING
    assert entry.settled_at is None


def test_billing_session_starts_unsettled_with_no_actual():
    session = BillingSession(principal_id=1, request_id="req-1")
    assert session.estimate == Decimal(0)
    assert session.actual is None
    assert session.settled is False
    assert session.refunded is False


def test_redemption_defaults_to_enabled_and_unused():
    code = Redemption(code="A" * 32, amount=Decimal("100"))
    assert code.status == RedemptionStatus.ENABLED
    assert code.used_at is None
    assert code.used_by_principal_id is None
