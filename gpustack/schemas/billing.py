"""Billing engine data model (WP1 of docs/prd/13-计费引擎开发TODO.md).

Design decisions frozen in WP0 (2026-09-22):

* **Two-level subject** — the wallet hangs off an ORG ``Principal`` (the
  settlement subject); quotas may be scoped down to a user, an api key or a
  model. Mirrors the tenant model already used by ``model_usage_details`` /
  ``metered_usage`` (``consumer_principal_id`` + kind snapshot).
* **Cached tokens are their own SKU** — ``model.token.cached`` prices
  separately from ``model.token.prompt``; the source detail rows already carry
  ``prompt_cached_token_count``.
* **Hybrid settlement** — token ledger entries settle in real time (the wallet
  is debited as the entry lands), resource entries (``gpu.hour.*`` /
  ``storage.*``, sourced from ``metered_usage`` hour buckets) settle deferred,
  at invoicing time. ``LedgerEntry.settle_mode`` records which side an entry
  belongs to so the two paths never guess.
* **Prepaid only** — balance is non-negative by construction; running out
  suspends the org's api keys rather than letting the wallet go below zero.

Conventions follow the neighbouring audit tables: FK-less id columns with
``*_name`` snapshots (billing rows must outlive the entities they charge for),
``Numeric`` for every money/quantity column, ``BaseModelMixin`` for
timestamps + soft delete.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import ClassVar, List, Optional

from pydantic import ConfigDict, model_validator
from sqlalchemy import (
    BigInteger,
    Column,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList, PublicFields, UTCDateTime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Money precision. 20 digits / 8 decimal places matches ``metered_usage``'s
# ``sku_count`` scale so a price × quantity product never loses the fractional
# part a sliced accelerator contributes. Amounts are rounded half-away-from-
# zero at the ledger boundary; the extra scale keeps intermediate products exact.
MONEY_PRECISION = 20
MONEY_SCALE = 8

# SKU families. Token SKUs settle in real time; resource SKUs settle deferred.
SKU_TOKEN_PROMPT = "model.token.prompt"
SKU_TOKEN_COMPLETION = "model.token.completion"
SKU_TOKEN_CACHED = "model.token.cached"
SKU_GPU_HOUR_PREFIX = "gpu.hour."  # + gpu_type, e.g. gpu.hour.910b
SKU_STORAGE_GB_HOUR = "storage.gb.hour"

# Wallet-movement SKUs. These price nothing — they label a credit or an
# adjustment in the ledger so every balance change has a row behind it — which
# is why ``unit_for_sku`` does not know them and rating never resolves a price
# for them.
SKU_WALLET_TOPUP = "wallet.topup"
SKU_WALLET_ADJUSTMENT = "wallet.adjustment"

# Units a price is quoted in, per SKU family.
UNIT_TOKENS = "tokens"
UNIT_GPU_HOURS = "gpu_hours"
UNIT_GB_HOURS = "gb_hours"
UNIT_REQUESTS = "requests"  # per-request priced operations (image / tts / stt)
UNIT_CURRENCY = "currency"  # wallet movements: quantity is an amount, not a count


def _money_column(**kwargs) -> Column:
    return Column(Numeric(MONEY_PRECISION, MONEY_SCALE), **kwargs)


# SKUs rated from ``model_usage_details`` (per request, settled immediately).
_TOKEN_SKUS = frozenset(
    {SKU_TOKEN_PROMPT, SKU_TOKEN_COMPLETION, SKU_TOKEN_CACHED}
)


def is_token_sku(sku: str) -> bool:
    """True for the per-request token SKUs.

    Per-request operations that carry no token counts (image / tts / stt, which
    ``model_usage_details`` bills per request) are priced under a token-family
    SKU with ``unit=requests`` rather than getting their own family, so this
    predicate — not the unit — is what decides the settlement path.
    """
    return sku in _TOKEN_SKUS


def settle_mode_for_sku(sku: str) -> "SettleMode":
    """Which settlement path a SKU belongs to (WP0 hybrid decision).

    Token SKUs settle in real time: the wallet is debited in the same
    transaction that writes the ledger entry. Everything else — ``gpu.hour.*``,
    ``storage.gb.hour`` and any future resource SKU — settles deferred, when the
    invoice covering its hour buckets is issued. Resource SKUs arrive from
    ``metered_usage`` buckets that are only final once ``sealed_at`` is set, so
    charging them on arrival would bill an hour that can still grow.
    """
    return SettleMode.REALTIME if is_token_sku(sku) else SettleMode.DEFERRED


def unit_for_sku(sku: str) -> str:
    """Canonical billing unit of a SKU, so a price row cannot disagree with it."""
    if is_token_sku(sku):
        return UNIT_TOKENS
    if sku.startswith(SKU_GPU_HOUR_PREFIX):
        return UNIT_GPU_HOURS
    if sku == SKU_STORAGE_GB_HOUR:
        return UNIT_GB_HOURS
    raise ValueError(f"unknown billing sku: {sku!r}")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LedgerDirection(str, Enum):
    """Which way money moves on a ledger entry."""

    DEBIT = "debit"  # charge the wallet
    CREDIT = "credit"  # top-up / refund / manual adjustment


class LedgerStatus(str, Enum):
    """Lifecycle of a single charge.

    ``PENDING`` only exists for deferred entries (resource SKUs waiting for the
    invoice that will settle them); realtime entries are written ``SETTLED`` in
    the same transaction that debits the wallet.
    """

    PENDING = "pending"
    SETTLED = "settled"
    REFUNDED = "refunded"
    VOID = "void"  # written in shadow mode / rejected by reconciliation


class SettleMode(str, Enum):
    """When the wallet is debited for this entry (WP0 hybrid decision)."""

    REALTIME = "realtime"  # token SKUs: debit as the entry lands
    DEFERRED = "deferred"  # resource SKUs: debit when the invoice is issued


class InvoiceStatus(str, Enum):
    DRAFT = "draft"
    ISSUED = "issued"
    SETTLED = "settled"  # wallet debited for the deferred entries it covers
    VOID = "void"


class RedemptionStatus(str, Enum):
    ENABLED = "enabled"
    USED = "used"
    DISABLED = "disabled"


class QuotaScope(str, Enum):
    """What a quota limit applies to (two-level subject decision)."""

    ORGANIZATION = "organization"
    USER = "user"
    API_KEY = "api_key"


class QuotaLimitType(str, Enum):
    DAILY_TOKENS = "daily_tokens"
    MONTHLY_AMOUNT = "monthly_amount"
    DAILY_AMOUNT = "daily_amount"


# ---------------------------------------------------------------------------
# Wallet — one per ORG principal (the settlement subject)
# ---------------------------------------------------------------------------


class Wallet(SQLModel, BaseModelMixin, table=True):
    """Prepaid balance of one organization.

    ``principal_id`` references the ORG ``Principal`` and is a real column (not
    FK-less): a wallet has no meaning once its subject is gone, and unlike the
    audit tables it is never read for history. Debits go through a single
    conditional UPDATE (``balance = balance - :n WHERE balance >= :n``) so
    concurrent charges cannot overdraw — the SQL-level equivalent of new-api's
    Lua reserve script.
    """

    __tablename__: ClassVar[str] = "billing_wallet"
    __table_args__ = (UniqueConstraint("principal_id", name="uq_wallet_principal"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    principal_id: int = Field(sa_column=Column(Integer, nullable=False, index=True))
    principal_name: Optional[str] = Field(default=None, max_length=255)
    currency: str = Field(default="CNY", sa_column=Column(String(8), nullable=False))
    balance: Decimal = Field(
        default=Decimal(0),
        sa_column=_money_column(nullable=False, default=Decimal(0), server_default="0"),
    )
    # Reserved by an open BillingSession (pre-authorization for a long-running
    # stream). Not part of ``balance``; available = balance - frozen.
    frozen: Decimal = Field(
        default=Decimal(0),
        sa_column=_money_column(nullable=False, default=Decimal(0), server_default="0"),
    )
    # Set when the balance runs dry; gateway_auth_reconciler suspends the org's
    # api keys while it is true and clears it on the next successful top-up.
    suspended: bool = Field(default=False)
    suspended_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )

    model_config = ConfigDict(protected_namespaces=())


# ---------------------------------------------------------------------------
# Price book — SKU × model × effective window
# ---------------------------------------------------------------------------


class PriceBookEntryBase(SQLModel):
    """Editable surface of a price row.

    ``sku`` + ``model_name`` (NULL for resource SKUs, which price per gpu_type
    embedded in the sku itself) + an effective window identify a price. Windows
    MUST NOT overlap for one (sku, model_name, group_name) — enforced by
    ``server.billing_pricing.assert_window_available``, not by a constraint,
    because "no overlap" is not expressible in portable DDL.

    The unit is derived from the sku rather than free-form: a price quoted per
    token against a gpu-hour SKU would silently bill a factor of 3600 off, so
    the pairing is checked at the API boundary instead of at rating time.
    """

    sku: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    # Model the price applies to (token SKUs). NULL = resource SKU or a
    # catch-all default for a sku family.
    model_name: Optional[str] = Field(default=None, max_length=255, index=True)
    # Optional discount dimension (mirrors new-api's groupRatio); NULL = no
    # group scoping. Reserved for the org-group pricing tier.
    group_name: Optional[str] = Field(default=None, max_length=128)
    unit: str = Field(sa_column=Column(String(32), nullable=False))
    # Price for ``per_quantity`` units (e.g. 0.002 per 1000 tokens) — the
    # divisor is explicit rather than implied by the unit string.
    price: Decimal = Field(sa_column=_money_column(nullable=False))
    per_quantity: Decimal = Field(
        default=Decimal(1),
        sa_column=_money_column(nullable=False, default=Decimal(1), server_default="1"),
    )
    currency: str = Field(default="CNY", sa_column=Column(String(8), nullable=False))
    effective_from: datetime = Field(sa_column=Column(UTCDateTime(), nullable=False))
    # NULL = open-ended (the current price).
    effective_to: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    is_active: bool = Field(default=True)

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="after")
    def _validate_pricing_shape(self):
        expected_unit = unit_for_sku(self.sku)  # raises on an unknown sku
        if self.unit != expected_unit:
            raise ValueError(
                f"unit for sku {self.sku!r} must be {expected_unit!r}, "
                f"got {self.unit!r}"
            )
        if self.price < 0:
            raise ValueError("price must not be negative")
        if self.per_quantity <= 0:
            raise ValueError("per_quantity must be greater than zero")
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be later than effective_from")
        return self


class PriceBookEntryUpdate(PriceBookEntryBase):
    """PUT payload: the whole editable surface, as with other resources here.

    ``sku`` / ``model_name`` / ``unit`` / ``currency`` / ``effective_from`` are
    the row's identity and history anchor — the route rejects changing them
    (edit the price or close the window instead), so a ledger entry's
    ``price_book_version`` always points at a row that means what it meant.
    """


class PriceBookEntryCreate(PriceBookEntryUpdate):
    pass


class PriceBookEntry(PriceBookEntryCreate, BaseModelMixin, table=True):
    """One price for one SKU.

    ``version`` is server-managed: it starts at 1 and is bumped by the route on
    every accepted price edit, and is stamped onto each ledger entry priced by
    this row, so a price change never rewrites history.
    """

    __tablename__: ClassVar[str] = "billing_price_book"

    id: Optional[int] = Field(default=None, primary_key=True)
    version: int = Field(default=1, sa_column=Column(Integer, nullable=False))


class PriceBookEntryListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "sku",
        "model_name",
        "price",
        "effective_from",
        "effective_to",
        "created_at",
        "updated_at",
    ]


class PriceBookEntryPublic(PriceBookEntryBase, PublicFields):
    version: int


PriceBookEntriesPublic = PaginatedList[PriceBookEntryPublic]


# ---------------------------------------------------------------------------
# Ledger — the immutable charge/refund record
# ---------------------------------------------------------------------------


class LedgerEntry(SQLModel, BaseModelMixin, table=True):
    """One priced charge (or credit), the system's single source of truth.

    Idempotency is the whole design: ``(source_table, source_id, sku)`` is
    unique, so re-running the rater over the same usage detail can never double
    charge. ``source_*`` points at the metering row that produced the entry
    (``model_usage_details.id`` for token SKUs, ``metered_usage.id`` for
    resource SKUs) and both ids are FK-less — a ledger row must survive
    archival or deletion of the usage row it was priced from.

    ``unit_price`` / ``price_book_version`` snapshot the price at rating time;
    the entry is never re-rated when the price book changes.
    """

    __tablename__: ClassVar[str] = "billing_ledger"
    __table_args__ = (
        UniqueConstraint(
            "source_table", "source_id", "sku", name="uq_ledger_source_sku"
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    # —— Provenance (idempotency key) ——
    source_table: str = Field(sa_column=Column(String(64), nullable=False))
    source_id: int = Field(sa_column=Column(BigInteger, nullable=False))

    # —— Payer + attribution snapshots (FK-less, audit style) ——
    principal_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    principal_name: Optional[str] = Field(default=None, max_length=255)
    user_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    user_name: Optional[str] = Field(default=None, max_length=255)
    api_key_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    api_key_name: Optional[str] = Field(default=None, max_length=255)
    model_name: Optional[str] = Field(default=None, max_length=255)
    cluster_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    # For resource entries: the metered resource (instance / volume) being paid
    # for. NULL for token entries.
    resource_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    resource_name: Optional[str] = Field(default=None, max_length=255)
    # Downstream request id, carried through so a user can quote one id and get
    # both the usage row and the charge (mirrors model_usage_details.request_id).
    request_id: Optional[str] = Field(default=None, index=True)

    # —— Pricing ——
    sku: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    quantity: Decimal = Field(sa_column=_money_column(nullable=False))
    unit: str = Field(sa_column=Column(String(32), nullable=False))
    unit_price: Decimal = Field(sa_column=_money_column(nullable=False))
    price_book_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    price_book_version: Optional[int] = Field(default=None)
    # Signed amount: positive for DEBIT (money leaving the wallet), negative
    # for CREDIT. Summing the column over a period is therefore the net charge.
    amount: Decimal = Field(sa_column=_money_column(nullable=False))
    currency: str = Field(default="CNY", sa_column=Column(String(8), nullable=False))
    direction: LedgerDirection = Field(
        default=LedgerDirection.DEBIT, sa_column=Column(String(16), nullable=False)
    )

    # —— Settlement ——
    settle_mode: SettleMode = Field(sa_column=Column(String(16), nullable=False))
    status: LedgerStatus = Field(
        default=LedgerStatus.PENDING, sa_column=Column(String(16), nullable=False)
    )
    billing_session_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    invoice_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    settled_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    # Time the underlying usage happened (not when it was rated) — invoices
    # group by this, and late-arriving details land in the period they belong
    # to rather than the period they were discovered in.
    occurred_at: datetime = Field(sa_column=Column(UTCDateTime(), nullable=False))

    model_config = ConfigDict(protected_namespaces=())


# ---------------------------------------------------------------------------
# Billing session — realtime side state machine (new-api §2 pattern)
# ---------------------------------------------------------------------------


class BillingSession(SQLModel, BaseModelMixin, table=True):
    """One request's pre-consume → settle / refund lifecycle.

    Only realtime (token) charges get a session; deferred resource charges are
    settled by the invoice. ``settled`` / ``refunded`` are mutually exclusive
    and each transition is one-way, which is what makes both operations safe to
    retry — the property new-api's BillingSession buys with the same two flags.
    """

    __tablename__: ClassVar[str] = "billing_session"

    id: Optional[int] = Field(default=None, primary_key=True)
    principal_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    request_id: Optional[str] = Field(default=None, index=True)
    # Pre-authorized estimate (frozen on the wallet while the request runs).
    estimate: Decimal = Field(
        default=Decimal(0),
        sa_column=_money_column(nullable=False, default=Decimal(0), server_default="0"),
    )
    # Actual charge once usage is known; NULL until settled.
    actual: Optional[Decimal] = Field(default=None, sa_column=_money_column())
    settled: bool = Field(default=False)
    refunded: bool = Field(default=False)
    settled_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )

    model_config = ConfigDict(protected_namespaces=())


# ---------------------------------------------------------------------------
# Invoice — deferred settlement + billing period document
# ---------------------------------------------------------------------------


class Invoice(SQLModel, BaseModelMixin, table=True):
    """One billing period for one organization.

    Covers the deferred (resource) ledger entries whose ``occurred_at`` falls in
    ``[period_start, period_end)``; issuing it debits the wallet once for the
    total and marks those entries ``SETTLED``. Token entries are already
    settled in real time, so they appear here as informational lines only.
    """

    __tablename__: ClassVar[str] = "billing_invoice"
    __table_args__ = (
        UniqueConstraint(
            "principal_id", "period_start", name="uq_invoice_principal_period"
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    principal_id: int = Field(sa_column=Column(Integer, nullable=False, index=True))
    principal_name: Optional[str] = Field(default=None, max_length=255)
    period_start: datetime = Field(sa_column=Column(UTCDateTime(), nullable=False))
    period_end: datetime = Field(sa_column=Column(UTCDateTime(), nullable=False))
    # Sum of the deferred entries this invoice settles.
    amount: Decimal = Field(sa_column=_money_column(nullable=False))
    currency: str = Field(default="CNY", sa_column=Column(String(8), nullable=False))
    status: InvoiceStatus = Field(
        default=InvoiceStatus.DRAFT, sa_column=Column(String(16), nullable=False)
    )
    issued_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    settled_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    # Set when the wallet could not cover the invoice; the org is suspended and
    # the invoice stays ISSUED until a top-up clears it.
    unpaid_reason: Optional[str] = Field(default=None, max_length=255)

    model_config = ConfigDict(protected_namespaces=())


class InvoiceItem(SQLModel, BaseModelMixin, table=True):
    """Per-SKU aggregation line of an invoice."""

    __tablename__: ClassVar[str] = "billing_invoice_item"

    id: Optional[int] = Field(default=None, primary_key=True)
    invoice_id: int = Field(sa_column=Column(Integer, nullable=False, index=True))
    sku: str = Field(sa_column=Column(String(64), nullable=False))
    model_name: Optional[str] = Field(default=None, max_length=255)
    quantity: Decimal = Field(sa_column=_money_column(nullable=False))
    unit: str = Field(sa_column=Column(String(32), nullable=False))
    amount: Decimal = Field(sa_column=_money_column(nullable=False))
    # Number of ledger entries rolled into this line (audit aid).
    entry_count: int = Field(default=0, sa_column=Column(Integer, nullable=False))

    model_config = ConfigDict(protected_namespaces=())


# There is deliberately no ``InvoiceCreate`` / ``InvoiceUpdate``: a statement is
# produced by the invoicer from the ledger and is not editable afterwards. An API
# that could rewrite an amount would make every invoice unverifiable against the
# entries it claims, which is the only thing that makes it auditable. Correction
# runs through the ledger (a refund, a re-rate) and the next statement.


class InvoiceItemPublic(SQLModel):
    id: int
    sku: str
    model_name: Optional[str] = None
    quantity: Decimal
    unit: str
    amount: Decimal
    entry_count: int = 0

    model_config = ConfigDict(protected_namespaces=())


class InvoicePublic(SQLModel):
    id: int
    principal_id: int
    principal_name: Optional[str] = None
    period_start: datetime
    period_end: datetime
    amount: Decimal
    currency: str
    status: InvoiceStatus
    issued_at: Optional[datetime] = None
    settled_at: Optional[datetime] = None
    unpaid_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(protected_namespaces=())


class InvoiceDetail(InvoicePublic):
    """One statement with the lines that make up its total."""

    items: List[InvoiceItemPublic] = []


class InvoiceListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "principal_id",
        "period_start",
        "period_end",
        "amount",
        "status",
        "issued_at",
        "settled_at",
        "created_at",
        "updated_at",
    ]


InvoicesPublic = PaginatedList[InvoicePublic]


# ---------------------------------------------------------------------------
# Quota — rate/spend limits consumed by the gateway (04 §3.1)
# ---------------------------------------------------------------------------


class QuotaBase(SQLModel):
    """Editable surface of a quota row.

    A limit is a triple of (subject, model, window): who it binds, what it
    covers, and how much of it may be consumed per window. ``model_name`` NULL
    means every model, which is what makes "this org may spend X a month"
    expressible without one row per model.

    Exactly one subject column is set and it must agree with ``scope`` — a row
    claiming API_KEY scope while carrying only a ``user_id`` would match nobody
    (the gate looks up by the column its scope names) and silently limit nothing,
    which is worse than an error at the API boundary.
    """

    scope: QuotaScope = Field(sa_column=Column(String(16), nullable=False))
    # Exactly one of these is set, matching ``scope``; the others stay NULL.
    # NOTE: SQL treats NULLs as distinct, so ``uq_quota_scope_limit`` only fully
    # guards scopes whose columns are all non-NULL — ``billing_quota.assert_no_duplicate``
    # rejects a duplicate (scope, subject, model, limit_type) whose subject
    # columns are NULL rather than relying on the constraint alone.
    principal_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    user_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    api_key_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    # NULL = all models.
    model_name: Optional[str] = Field(default=None, max_length=255)
    limit_type: QuotaLimitType = Field(sa_column=Column(String(32), nullable=False))
    # Tokens for DAILY_TOKENS, currency for the *_AMOUNT types. One column
    # rather than two because a row's ``limit_type`` decides the unit, exactly
    # as ``PriceBookEntry.unit`` is derived from its sku.
    limit_value: Decimal = Field(sa_column=_money_column(nullable=False))
    enabled: bool = Field(default=True)

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="after")
    def _validate_subject(self):
        expected = {
            QuotaScope.ORGANIZATION: "principal_id",
            QuotaScope.USER: "user_id",
            QuotaScope.API_KEY: "api_key_id",
        }[QuotaScope(self.scope)]
        present = [
            name
            for name in ("principal_id", "user_id", "api_key_id")
            if getattr(self, name) is not None
        ]
        if present != [expected]:
            raise ValueError(
                f"scope {str(self.scope)!r} requires exactly {expected} to be set, "
                f"got {present or 'none'}"
            )
        if self.limit_value <= 0:
            raise ValueError("limit_value must be greater than zero")
        return self


class QuotaUpdate(QuotaBase):
    """PUT payload: the whole editable surface.

    ``window_start`` / ``used`` are deliberately absent — they are the rater's
    state, and an API that let a caller zero them would let a tenant reset its
    own counter by editing the row.
    """


class QuotaCreate(QuotaUpdate):
    pass


class Quota(QuotaBase, BaseModelMixin, table=True):
    """A spend/usage ceiling at organization, user or api-key scope.

    Checked before a request is admitted (``server.billing_quota.check_quota``)
    and advanced after rating, so the hot path reads one row instead of summing
    the ledger. ``window_start`` marks the window ``used`` counts, which is what
    lets a daily counter be reset by comparison rather than by a cron job.
    """

    __tablename__: ClassVar[str] = "billing_quota"
    __table_args__ = (
        UniqueConstraint(
            "scope",
            "principal_id",
            "user_id",
            "api_key_id",
            "model_name",
            "limit_type",
            name="uq_quota_scope_limit",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    # Rolling window state, advanced by the rater.
    window_start: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    used: Decimal = Field(
        default=Decimal(0),
        sa_column=_money_column(nullable=False, default=Decimal(0), server_default="0"),
    )


class QuotaPublic(QuotaBase):
    id: int
    # Read-only window state, so a tenant can see how much of the ceiling is
    # left rather than only being told "429".
    window_start: Optional[datetime] = None
    used: Decimal = Decimal(0)
    created_at: datetime
    updated_at: datetime


class QuotaListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "scope",
        "limit_type",
        "limit_value",
        "created_at",
        "updated_at",
    ]


QuotasPublic = PaginatedList[QuotaPublic]


# ---------------------------------------------------------------------------
# Redemption — prepaid top-up codes (new-api §1.1 CAS pattern)
# ---------------------------------------------------------------------------


class Redemption(SQLModel, BaseModelMixin, table=True):
    """A single-use top-up code.

    Redeeming is one transaction: lock the row, flip ``status`` ENABLED → USED
    under a ``WHERE status = 'enabled'`` guard, then credit the wallet. The
    guard is what makes concurrent redemption of one code safe — the loser's
    UPDATE affects zero rows and the transaction aborts.
    """

    __tablename__: ClassVar[str] = "billing_redemption"

    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(sa_column=Column(String(64), nullable=False, unique=True))
    amount: Decimal = Field(sa_column=_money_column(nullable=False))
    currency: str = Field(default="CNY", sa_column=Column(String(8), nullable=False))
    status: RedemptionStatus = Field(
        default=RedemptionStatus.ENABLED, sa_column=Column(String(16), nullable=False)
    )
    # Batch/issuer label for operations (who created this batch, which campaign).
    batch: Optional[str] = Field(default=None, max_length=128)
    expires_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )
    used_by_principal_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer)
    )
    used_by_user_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    used_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime(), nullable=True)
    )

    model_config = ConfigDict(protected_namespaces=())


# ---------------------------------------------------------------------------
# Adjustment — operator-initiated wallet corrections (WP4.5)
# ---------------------------------------------------------------------------


class Adjustment(SQLModel, BaseModelMixin, table=True):
    """A manual correction to one wallet, keyed by an operator-supplied token.

    Redemptions cover money a tenant bought; this covers money an operator moved
    — a billing error reversed, a goodwill credit, a settlement that did not go
    through a code. Both write a wallet movement and a ledger row, and neither
    may be replayable by accident.

    The idempotency key is the caller's, not ours: a ticket number, a payment
    reference, whatever the operator's process guarantees is unique per
    correction. It is deliberately *not* the redemption ``code`` column reused
    under another name. A code identifies an instrument that existed before the
    money moved and can only be spent once; an adjustment has no instrument, so
    the only thing between a retried HTTP request and a double credit is a key
    the caller chose. Sharing one column between the two would let a ticket
    number collide with a code, and that collision fails closed — a legitimate
    correction rejected as a duplicate of an unrelated redemption.

    ``amount`` is signed: positive credits the wallet, negative debits it. One
    column rather than a direction plus a magnitude, because the sign is what
    both the ledger row and the wallet movement need, and a representation that
    has to be recombined is one that can be recombined wrongly.
    """

    __tablename__: ClassVar[str] = "billing_adjustment"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_adjustment_idempotency"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    # Both are real columns: ``principal_id`` is who the money belongs to (the
    # subject every other billing query is scoped by), ``wallet_id`` is the row
    # that moved. Either alone would leave a question answered only by a join
    # against history.
    principal_id: int = Field(sa_column=Column(Integer, nullable=False, index=True))
    principal_name: Optional[str] = Field(default=None, max_length=255)
    wallet_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    amount: Decimal = Field(sa_column=_money_column(nullable=False))
    currency: str = Field(default="CNY", sa_column=Column(String(8), nullable=False))
    # Why. An unexplained correction is indistinguishable from a mistake six
    # months later, and this is the column an auditor reads.
    reason: str = Field(sa_column=Column(String(512), nullable=False))
    operator_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    operator_name: Optional[str] = Field(default=None, max_length=255)
    # 191 rather than 255: the longest prefix MySQL can put a unique index on
    # under utf8mb4, so the constraint survives a deployment on that database.
    # No ``index=True`` here — the unique constraint below already creates one,
    # and a second index on the same column would only cost writes.
    idempotency_key: str = Field(sa_column=Column(String(191), nullable=False))
    # The ledger row this adjustment produced. A correction that cannot be tied
    # to its entry cannot be reconciled against the wallet movement, which is the
    # property that makes a manual money move auditable at all.
    ledger_entry_id: Optional[int] = Field(default=None, sa_column=Column(Integer))

    model_config = ConfigDict(protected_namespaces=())


class AdjustmentPublic(SQLModel):
    """Read-only: an adjustment records something that already happened.

    No update surface and no delete. Correcting a correction is another
    adjustment, so the trail stays complete — an edited or removed row would mean
    the ledger and the wallet no longer explain each other.
    """

    id: int
    principal_id: int
    principal_name: Optional[str] = None
    wallet_id: Optional[int] = None
    amount: Decimal
    currency: str
    reason: str
    operator_id: Optional[int] = None
    operator_name: Optional[str] = None
    idempotency_key: str
    ledger_entry_id: Optional[int] = None
    created_at: datetime

    model_config = ConfigDict(protected_namespaces=())


class AdjustmentListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "principal_id",
        "amount",
        "operator_id",
        "created_at",
    ]


AdjustmentsPublic = PaginatedList[AdjustmentPublic]


class AdjustmentCreate(SQLModel):
    """The request body for a manual correction.

    ``operator_id`` / ``operator_name`` are absent on purpose: they are taken
    from the authenticated caller, because an operator who could name somebody
    else as the author of a money movement could author one anonymously. Same
    reasoning for ``principal_name``, which is snapshotted server-side.
    """

    principal_id: int
    # Signed: positive credits the wallet, negative debits it. A separate
    # ``direction`` field would be one more thing to get inconsistent.
    amount: Decimal
    reason: str = Field(min_length=1, max_length=512)
    # The caller's key — a ticket number, a payment reference. Required, because
    # a correction with no idempotency key is one that a retried request can
    # apply twice, and the failure is silent.
    idempotency_key: str = Field(min_length=1, max_length=191)
    currency: str = Field(default="CNY", max_length=8)

    model_config = ConfigDict(protected_namespaces=())


class WalletListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "principal_id",
        "balance",
        "frozen",
        "suspended",
        "suspended_at",
        "created_at",
        "updated_at",
    ]


class WalletPublic(SQLModel):
    """A wallet as its owner or an operator needs to see it.

    ``available`` is derived rather than stored: balance minus frozen is what a
    caller can actually spend, and a client that computes it themselves will
    eventually compute it differently.
    """

    id: int
    principal_id: int
    principal_name: Optional[str] = None
    currency: str
    balance: Decimal
    frozen: Decimal
    available: Decimal = Decimal(0)
    suspended: bool = False
    suspended_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(protected_namespaces=())


class WalletState(WalletPublic):
    """A wallet plus what it owes — the numbers behind a 402.

    The outstanding split matters operationally: realtime arrears are collected
    by the next settlement sweep, invoice arrears by the next invoicing pass, so
    "how much do I need to top up" and "when will service come back" are two
    questions with different answers.
    """

    outstanding_realtime: Decimal = Decimal(0)
    outstanding_invoices: Decimal = Decimal(0)
    outstanding_total: Decimal = Decimal(0)


class LedgerEntryPublic(SQLModel):
    """One ledger row, as an auditor or a disputing tenant reads it.

    Carries the source identity (``source_table`` / ``source_id`` / ``request_id``)
    because that is what makes a charge traceable to the usage row it came from,
    and the price snapshot (``unit_price`` / ``price_book_id`` /
    ``price_book_version``) because a bill has to stay reproducible after the
    price that produced it has been superseded.
    """

    id: int
    source_table: str
    source_id: int
    principal_id: Optional[int] = None
    principal_name: Optional[str] = None
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    api_key_id: Optional[int] = None
    api_key_name: Optional[str] = None
    model_name: Optional[str] = None
    cluster_id: Optional[int] = None
    resource_id: Optional[int] = None
    resource_name: Optional[str] = None
    request_id: Optional[str] = None
    sku: str
    quantity: Decimal
    unit: str
    unit_price: Decimal
    price_book_id: Optional[int] = None
    price_book_version: Optional[int] = None
    amount: Decimal
    currency: str
    direction: str
    settle_mode: str
    status: str
    billing_session_id: Optional[int] = None
    invoice_id: Optional[int] = None
    settled_at: Optional[datetime] = None
    occurred_at: datetime
    created_at: datetime

    model_config = ConfigDict(protected_namespaces=())


class LedgerListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "occurred_at",
        "amount",
        "quantity",
        "principal_id",
        "sku",
        "created_at",
    ]


LedgerEntriesPublic = PaginatedList[LedgerEntryPublic]
