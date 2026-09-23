"""Billing administration API — the price book (WP2) and quotas (WP5.4).

Platform-scoped and admin-only (mounted under ``admin_routers``): prices are one
tenant-agnostic truth, so unlike the org-owned resources there is no
``owner_principal_id`` to scope by and no tenant filter to apply. Quotas sit here
for the same reason — a ceiling is granted by an operator, and a tenant that
could edit its own limit has no limit.

Writes go through the invariants in ``server.billing_pricing`` — window overlap
and identity stability — and invalidate that module's price cache, so the rater
never prices against a timeline where two rows claim the same instant.

A price row is append-only in spirit: ``sku`` / ``model_name`` / ``unit`` /
``currency`` / ``effective_from`` cannot change on an existing row, because
every ledger entry snapshots ``price_book_id`` + ``price_book_version`` and a
mutated identity would make those snapshots lie. Editing a price means either
changing ``price`` on the row (which bumps ``version``) or closing its window
and adding a new one.
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import Field
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlmodel import select

from gpustack.api.exceptions import (
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
)
from gpustack.schemas.billing import (
    Adjustment,
    AdjustmentCreate,
    AdjustmentListParams,
    AdjustmentPublic,
    AdjustmentsPublic,
    Invoice,
    InvoiceDetail,
    InvoiceItem,
    InvoiceItemPublic,
    InvoiceListParams,
    InvoicesPublic,
    LedgerEntriesPublic,
    LedgerEntry,
    LedgerListParams,
    PriceBookEntriesPublic,
    PriceBookEntry,
    PriceBookEntryCreate,
    PriceBookEntryListParams,
    PriceBookEntryPublic,
    PriceBookEntryUpdate,
    Quota,
    QuotaCreate,
    QuotaListParams,
    QuotaPublic,
    QuotasPublic,
    QuotaUpdate,
    Wallet,
    WalletListParams,
    WalletState,
    unit_for_sku,
)
from gpustack.server.billing_pricing import (
    assert_identity_unchanged,
    assert_window_available,
    compute_amount,
    invalidate_price_cache,
    resolve_price,
)
from gpustack.server.billing_settlement import (
    apply_adjustment,
    outstanding_invoices,
    outstanding_realtime_charges,
    redeem_code,
)
from gpustack.server.billing_quota import (
    assert_no_duplicate,
    enabled_quota_specs,
    invalidate_quota_cache,
    specs_for_caller,
    window_start_for,
    window_used,
)
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.schemas.common import PaginatedList
from gpustack.utils.export_delivery import XLSX_MEDIA_TYPE
from gpustack.utils.export_limits import attachment_headers
from gpustack.utils.tabular_export import build_xlsx

logger = logging.getLogger(__name__)

router = APIRouter()


class PriceResolveRequest(BaseModel):
    """What a rating would charge, asked before anything is charged.

    Exists for the admin UI's price-preview and for ops triage ("why did this
    request cost that"): it runs the same resolver the rater uses, so an answer
    from here is the answer rating will give.
    """

    sku: str
    model_name: Optional[str] = None
    group_name: Optional[str] = None
    at: Optional[datetime] = None
    # Sample quantity to price, so the response carries an amount and not just
    # a rate. Defaults to one billing unit's worth of the price row.
    quantity: Optional[float] = None

    model_config = ConfigDict(protected_namespaces=())


class PriceResolveResponse(BaseModel):
    priced: bool
    entry: Optional[PriceBookEntryPublic] = None
    quantity: Optional[float] = None
    amount: Optional[float] = None
    reason: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())


@router.get("", response_model=PriceBookEntriesPublic)
async def list_price_book(
    params: PriceBookEntryListParams = Depends(),
    sku: Optional[str] = None,
    model_name: Optional[str] = None,
    search: Optional[str] = None,
):
    fuzzy_fields = {"sku": search} if search else {}
    fields = {"deleted_at": None}
    if sku:
        fields["sku"] = sku
    if model_name:
        fields["model_name"] = model_name

    if params.watch:
        return StreamingResponse(
            PriceBookEntry.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await PriceBookEntry.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            page=params.page,
            per_page=params.perPage,
            order_by=params.sort_by,
        )


@router.get("/{id}", response_model=PriceBookEntryPublic)
async def get_price(session: SessionDep, id: int):
    return await _one_price_or_404(session, id)


@router.post("", response_model=PriceBookEntryPublic)
async def create_price(session: SessionDep, input: PriceBookEntryCreate):
    # ``unit_for_sku`` raises on an unknown SKU; the schema validator already
    # ran, so this is the belt to its braces — and it is what turns "no price
    # found" at rating time into a 422 at configuration time.
    unit_for_sku(input.sku)
    await assert_window_available(
        session,
        sku=input.sku,
        model_name=input.model_name,
        group_name=input.group_name,
        effective_from=input.effective_from,
        effective_to=input.effective_to,
    )

    try:
        created = await PriceBookEntry.create(session, input)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create price: {e}")

    invalidate_price_cache()
    return created


@router.put("/{id}", response_model=PriceBookEntryPublic)
async def update_price(session: SessionDep, id: int, input: PriceBookEntryUpdate):
    existing = await _one_price_or_404(session, id)
    # Identity is immutable: build the candidate row only to compare it, so the
    # rejection names the fields instead of silently ignoring them.
    candidate = PriceBookEntry.model_validate(
        {**input.model_dump(), "id": id, "version": existing.version}
    )
    assert_identity_unchanged(existing, candidate)
    await assert_window_available(
        session,
        sku=existing.sku,
        model_name=existing.model_name,
        group_name=input.group_name,
        effective_from=existing.effective_from,
        effective_to=input.effective_to,
        exclude_id=id,
    )

    payload = input.model_dump()
    # Bump the version only when the priced surface actually changed, so a
    # no-op PUT does not orphan ledger snapshots on a version that looks
    # different from the one that produced them.
    priced_surface = ("price", "per_quantity", "group_name", "effective_to")
    if any(payload.get(name) != getattr(existing, name) for name in priced_surface):
        payload["version"] = existing.version + 1

    try:
        await existing.update(session, payload)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update price: {e}")

    invalidate_price_cache()
    return await _one_price_or_404(session, id)


@router.delete("/{id}")
async def delete_price(session: SessionDep, id: int):
    existing = await _one_price_or_404(session, id)
    try:
        await existing.delete(session=session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete price: {e}")

    invalidate_price_cache()


@router.post("/resolve", response_model=PriceResolveResponse)
async def resolve(session: SessionDep, input: PriceResolveRequest):
    """Preview the price that rating would apply — nothing is written."""
    entry = await resolve_price(
        session,
        sku=input.sku,
        model_name=input.model_name,
        group_name=input.group_name,
        at=input.at,
    )
    if entry is None:
        return PriceResolveResponse(
            priced=False,
            reason=(
                f"no active price covers sku '{input.sku}'"
                + (f" model '{input.model_name}'" if input.model_name else "")
                + " at the requested instant"
            ),
        )

    quantity = (
        Decimal(str(input.quantity))
        if input.quantity is not None
        else entry.per_quantity
    )
    return PriceResolveResponse(
        priced=True,
        entry=PriceBookEntryPublic.model_validate(entry, from_attributes=True),
        quantity=float(quantity),
        amount=float(compute_amount(entry, quantity)),
    )


async def _one_price_or_404(session, id: int) -> PriceBookEntry:
    existing = await PriceBookEntry.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"price {id} not found")
    return existing


# ---------------------------------------------------------------------------
# Quotas (WP5.4) — ceilings on a subject's consumption per window
# ---------------------------------------------------------------------------

quota_router = APIRouter()


class QuotaCheckRequest(BaseModel):
    """Would this caller be refused right now?

    Runs the same gate the request paths run, so an operator can answer "why is
    this key getting 429s" — or verify a new ceiling bites where intended —
    without sending inference traffic. Nothing is written except a window
    rollover, which the gate performs on a real request too.
    """

    api_key_id: Optional[int] = None
    user_id: Optional[int] = None
    principal_id: Optional[int] = None
    model_name: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())


class QuotaCheckVerdict(BaseModel):
    quota_id: int
    scope: str
    limit_type: str
    model_name: Optional[str] = None
    limit_value: Decimal
    used: Decimal
    remaining: Decimal
    window_start: Optional[datetime] = None
    exhausted: bool

    model_config = ConfigDict(protected_namespaces=())


class QuotaCheckResponse(BaseModel):
    admitted: bool
    # The ceiling that refused, when one did — a caller can be under several.
    refused_by: Optional[QuotaCheckVerdict] = None
    evaluated: List[QuotaCheckVerdict] = []

    model_config = ConfigDict(protected_namespaces=())


@quota_router.get("", response_model=QuotasPublic)
async def list_quotas(
    params: QuotaListParams = Depends(),
    scope: Optional[str] = None,
    api_key_id: Optional[int] = None,
    user_id: Optional[int] = None,
    principal_id: Optional[int] = None,
    model_name: Optional[str] = None,
    enabled: Optional[bool] = None,
    search: Optional[str] = None,
):
    fuzzy_fields = {"model_name": search} if search else {}
    fields = {"deleted_at": None}
    if scope:
        fields["scope"] = scope
    if api_key_id is not None:
        fields["api_key_id"] = api_key_id
    if user_id is not None:
        fields["user_id"] = user_id
    if principal_id is not None:
        fields["principal_id"] = principal_id
    if model_name:
        fields["model_name"] = model_name
    if enabled is not None:
        fields["enabled"] = enabled

    if params.watch:
        return StreamingResponse(
            Quota.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await Quota.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            page=params.page,
            per_page=params.perPage,
            order_by=params.sort_by,
        )


@quota_router.get("/{id}", response_model=QuotaPublic)
async def get_quota(session: SessionDep, id: int):
    return await _one_quota_or_404(session, id)


@quota_router.post("", response_model=QuotaPublic)
async def create_quota(session: SessionDep, input: QuotaCreate):
    # The unique constraint cannot catch this one: every subject column but the
    # scope's own is NULL, and SQL treats NULLs as distinct, so two "all models"
    # ceilings for one key would both insert and both count the same window.
    await assert_no_duplicate(
        session,
        scope=input.scope,
        principal_id=input.principal_id,
        user_id=input.user_id,
        api_key_id=input.api_key_id,
        model_name=input.model_name,
        limit_type=input.limit_type,
    )

    try:
        created = await Quota.create(session, input)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create quota: {e}")

    invalidate_quota_cache()
    return created


@quota_router.put("/{id}", response_model=QuotaPublic)
async def update_quota(session: SessionDep, id: int, input: QuotaUpdate):
    existing = await _one_quota_or_404(session, id)
    # Identity is immutable, and for a reason specific to quotas: ``used`` counts
    # one subject's consumption of one window, so moving a row to a different
    # subject or model would leave it carrying a total that means something else
    # entirely — a ceiling that bites at the wrong threshold with no trace of why.
    # Delete and recreate to re-point a limit.
    identity = ("scope", "principal_id", "user_id", "api_key_id", "model_name",
                "limit_type")
    changed = [
        name
        for name in identity
        if _quota_field(input, name) != _quota_field(existing, name)
    ]
    if changed:
        raise InvalidException(
            message=(
                f"cannot change {', '.join(changed)} on an existing quota — its "
                "counter would no longer describe the subject it counts; delete "
                "it and create a new one"
            )
        )

    try:
        await existing.update(session, input.model_dump())
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update quota: {e}")

    invalidate_quota_cache()
    return await _one_quota_or_404(session, id)


@quota_router.delete("/{id}")
async def delete_quota(session: SessionDep, id: int):
    existing = await _one_quota_or_404(session, id)
    try:
        await existing.delete(session=session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete quota: {e}")

    invalidate_quota_cache()


@quota_router.post("/check", response_model=QuotaCheckResponse)
async def check(session: SessionDep, input: QuotaCheckRequest):
    """Evaluate the gate for a hypothetical caller and report every ceiling."""
    specs = specs_for_caller(
        await enabled_quota_specs(session),
        api_key_id=input.api_key_id,
        user_id=input.user_id,
        principal_id=input.principal_id,
        model_name=input.model_name,
    )
    now = datetime.now(timezone.utc)
    verdicts: List[QuotaCheckVerdict] = []
    refused_by = None
    for spec in specs:
        used = await window_used(session, spec, now)
        verdict = QuotaCheckVerdict(
            quota_id=spec.id,
            scope=spec.scope.value,
            limit_type=spec.limit_type.value,
            model_name=spec.model_name,
            limit_value=spec.limit_value,
            used=used,
            remaining=max(Decimal(0), spec.limit_value - used),
            window_start=window_start_for(spec.limit_type, now),
            exhausted=used >= spec.limit_value,
        )
        verdicts.append(verdict)
        if verdict.exhausted and refused_by is None:
            refused_by = verdict
    return QuotaCheckResponse(
        admitted=refused_by is None, refused_by=refused_by, evaluated=verdicts
    )


def _quota_field(row, name: str):
    """One identity field of a quota, with enums compared by value.

    A request body carries the enum member while a row read back from the
    database carries whatever the column holds, and ``QuotaScope.ORGANIZATION !=
    "organization"`` is False only by luck of the str mixin — comparing values
    makes the immutability check mean what it says instead of depending on it.
    """
    value = getattr(row, name, None)
    return value.value if isinstance(value, Enum) else value


async def _one_quota_or_404(session, id: int) -> Quota:
    existing = await Quota.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"quota {id} not found")
    return existing


# ---------------------------------------------------------------------------
# Invoices (WP6) — read-only statements over the deferred ledger
# ---------------------------------------------------------------------------

invoice_router = APIRouter()

# What an entry on a statement has to carry for a tenant to recognise it. The
# point of ``request_id`` and the source columns is that a disputed line can be
# followed back to the row the rater read, which is the only way "why was I
# charged this" has an answer that is not "trust us".
INVOICE_ENTRY_COLUMNS = [
    "ledger_id",
    "request_id",
    "occurred_at",
    "sku",
    "model_name",
    "resource_name",
    "quantity",
    "unit",
    "unit_price",
    "amount",
    "currency",
    "settle_mode",
    "status",
    "source_table",
    "source_id",
    "user_name",
    "api_key_name",
]

INVOICE_ITEM_COLUMNS = [
    "sku",
    "model_name",
    "quantity",
    "unit",
    "amount",
    "entry_count",
]


class InvoiceEntryPublic(BaseModel):
    """One ledger entry behind a statement line."""

    id: int
    request_id: Optional[str] = None
    occurred_at: datetime
    sku: str
    model_name: Optional[str] = None
    resource_name: Optional[str] = None
    quantity: Decimal
    unit: str
    unit_price: Decimal
    amount: Decimal
    currency: str
    settle_mode: str
    status: str
    source_table: str
    source_id: int
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    api_key_id: Optional[int] = None
    api_key_name: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())


class InvoiceEntriesPublic(BaseModel):
    items: List[InvoiceEntryPublic] = []
    total: int = 0
    # The statement's own total, so a page of entries can be read against it
    # without a second call — and so a mismatch is visible at a glance.
    invoice_amount: Decimal = Decimal(0)

    model_config = ConfigDict(protected_namespaces=())


@invoice_router.get("", response_model=InvoicesPublic)
async def list_invoices(
    params: InvoiceListParams = Depends(),
    principal_id: Optional[int] = None,
    status: Optional[str] = None,
    period_start: Optional[datetime] = None,
    period_end: Optional[datetime] = None,
    search: Optional[str] = None,
):
    """Statements, newest period first.

    ``period_start`` / ``period_end`` filter on overlap with the requested range,
    so a month's worth of statements is one call rather than one per org.
    """
    fuzzy_fields = {"principal_name": search} if search else {}
    fields = {"deleted_at": None}
    if principal_id is not None:
        fields["principal_id"] = principal_id
    if status:
        fields["status"] = status

    extra_conditions = []
    if period_start is not None:
        extra_conditions.append(Invoice.period_end > period_start)
    if period_end is not None:
        extra_conditions.append(Invoice.period_start < period_end)

    if params.watch:
        return StreamingResponse(
            Invoice.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await Invoice.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            extra_conditions=extra_conditions or None,
            page=params.page,
            per_page=params.perPage,
            order_by=params.sort_by,
        )


@invoice_router.get("/{id}", response_model=InvoiceDetail)
async def get_invoice(session: SessionDep, id: int):
    invoice = await _one_invoice_or_404(session, id)
    items = (
        await session.exec(
            select(InvoiceItem)
            .where(InvoiceItem.invoice_id == id, InvoiceItem.deleted_at.is_(None))
            .order_by(InvoiceItem.sku, InvoiceItem.model_name)
        )
    ).all()
    detail = InvoiceDetail.model_validate(invoice, from_attributes=True)
    detail.items = [
        InvoiceItemPublic.model_validate(item, from_attributes=True) for item in items
    ]
    return detail


@invoice_router.get("/{id}/entries", response_model=InvoiceEntriesPublic)
async def get_invoice_entries(
    session: SessionDep, id: int, page: int = 1, per_page: int = 100
):
    """The ledger rows a statement was built from, ``request_id`` included."""
    invoice = await _one_invoice_or_404(session, id)
    statement = (
        select(LedgerEntry)
        .where(
            LedgerEntry.invoice_id == id,
            LedgerEntry.deleted_at.is_(None),
        )
        .order_by(LedgerEntry.occurred_at, LedgerEntry.id)
    )
    total = len((await session.exec(select(LedgerEntry.id).where(
        LedgerEntry.invoice_id == id, LedgerEntry.deleted_at.is_(None)
    ))).all())
    rows = (
        await session.exec(
            statement.offset(max(0, (page - 1) * per_page)).limit(per_page)
        )
    ).all()
    return InvoiceEntriesPublic(
        items=[
            InvoiceEntryPublic.model_validate(row, from_attributes=True) for row in rows
        ],
        total=total,
        invoice_amount=Decimal(invoice.amount or 0),
    )


@invoice_router.get("/{id}/export")
async def export_invoice(session: SessionDep, id: int):
    """The statement as a workbook: one sheet of lines, one of entries.

    Two sheets rather than one because they answer different questions — the
    lines are what the total is made of, the entries are what a disputed line is
    made of — and flattening them would either repeat the entry detail on every
    line or lose the ``request_id`` that makes a charge traceable.
    """
    invoice = await _one_invoice_or_404(session, id)
    items = (
        await session.exec(
            select(InvoiceItem)
            .where(InvoiceItem.invoice_id == id, InvoiceItem.deleted_at.is_(None))
            .order_by(InvoiceItem.sku, InvoiceItem.model_name)
        )
    ).all()
    entries = (
        await session.exec(
            select(LedgerEntry)
            .where(LedgerEntry.invoice_id == id, LedgerEntry.deleted_at.is_(None))
            .order_by(LedgerEntry.occurred_at, LedgerEntry.id)
        )
    ).all()

    stamp = f"{invoice.period_start:%Y%m%d}-{invoice.period_end:%Y%m%d}"
    payload = await build_xlsx(
        [
            (
                "Invoice",
                [
                    "invoice_id",
                    "principal_id",
                    "principal_name",
                    "period_start",
                    "period_end",
                    "amount",
                    "currency",
                    "status",
                    "issued_at",
                    "settled_at",
                    "unpaid_reason",
                ],
                _one_row(
                    [
                        invoice.id,
                        invoice.principal_id,
                        invoice.principal_name,
                        _naive(invoice.period_start),
                        _naive(invoice.period_end),
                        float(invoice.amount or 0),
                        invoice.currency,
                        _value(invoice.status),
                        _naive(invoice.issued_at),
                        _naive(invoice.settled_at),
                        invoice.unpaid_reason,
                    ]
                ),
            ),
            (
                "Items",
                INVOICE_ITEM_COLUMNS,
                _rows(
                    [
                        item.sku,
                        item.model_name,
                        float(item.quantity or 0),
                        item.unit,
                        float(item.amount or 0),
                        item.entry_count,
                    ]
                    for item in items
                ),
            ),
            (
                "Entries",
                INVOICE_ENTRY_COLUMNS,
                _rows(
                    [
                        entry.id,
                        entry.request_id,
                        _naive(entry.occurred_at),
                        entry.sku,
                        entry.model_name,
                        entry.resource_name,
                        float(entry.quantity or 0),
                        entry.unit,
                        float(entry.unit_price or 0),
                        float(entry.amount or 0),
                        entry.currency,
                        _value(entry.settle_mode),
                        _value(entry.status),
                        entry.source_table,
                        entry.source_id,
                        entry.user_name,
                        entry.api_key_name,
                    ]
                    for entry in entries
                ),
            ),
        ]
    )
    return Response(
        content=payload,
        media_type=XLSX_MEDIA_TYPE,
        headers=attachment_headers(
            f"invoice-{invoice.principal_id}-{stamp}.xlsx"
        ),
    )


def _value(field):
    """Enum members render as their value; everything else passes through."""
    return field.value if isinstance(field, Enum) else field


def _naive(moment: Optional[datetime]) -> Optional[datetime]:
    """Drop the offset before a spreadsheet sees it.

    xlsxwriter is configured with ``remove_timezone`` as a backstop, but an
    instant written as UTC-naive is the one an operator compares against the
    ledger's own ``occurred_at``, which is stored naive.
    """
    if moment is None or moment.tzinfo is None:
        return moment
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


async def _one_row(row):
    yield row


async def _rows(iterator):
    for row in iterator:
        yield row


async def _one_invoice_or_404(session, id: int) -> Invoice:
    existing = await Invoice.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"invoice {id} not found")
    return existing


# ---------------------------------------------------------------------------
# Wallets (WP7.1) — balances, and the numbers behind a 402
# ---------------------------------------------------------------------------

wallet_router = APIRouter()
ledger_router = APIRouter()
adjustment_router = APIRouter()
# Tenant-facing: an org reads its own balance and redeems its own codes. Mounted
# under tenant_routers, so the caller's principal is resolved by the tenant
# context rather than passed in — a tenant that could name whose wallet to read
# could read anybody's.
tenant_router = APIRouter()


class RedeemRequest(BaseModel):
    code: str = Field(min_length=1, max_length=64)

    model_config = ConfigDict(protected_namespaces=())


class RedeemResponse(BaseModel):
    amount: Decimal
    currency: str
    # The balance after the credit, so the tenant sees the effect of the code in
    # the same response rather than having to ask again.
    balance: Decimal
    suspended: bool
    used_at: Optional[datetime] = None

    model_config = ConfigDict(protected_namespaces=())


async def _wallet_state(session, wallet) -> WalletState:
    """One wallet with what it owes, split the way the two pipelines collect it."""
    state = WalletState.model_validate(wallet, from_attributes=True)
    state.available = Decimal(wallet.balance or 0) - Decimal(wallet.frozen or 0)
    state.outstanding_realtime = await outstanding_realtime_charges(
        session, wallet.principal_id
    )
    state.outstanding_invoices = await outstanding_invoices(session, wallet.principal_id)
    state.outstanding_total = (
        state.outstanding_realtime + state.outstanding_invoices
    )
    return state


@wallet_router.get("", response_model=PaginatedList[WalletState])
async def list_wallets(
    params: WalletListParams = Depends(),
    principal_id: Optional[int] = None,
    suspended: Optional[bool] = None,
    search: Optional[str] = None,
):
    """Every wallet, for the operator view: who is in arrears, who is suspended."""
    fields = {"deleted_at": None}
    if principal_id is not None:
        fields["principal_id"] = principal_id
    if suspended is not None:
        fields["suspended"] = suspended
    fuzzy_fields = {"principal_name": search} if search else {}

    if params.watch:
        return StreamingResponse(
            Wallet.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        page = await Wallet.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            page=params.page,
            per_page=params.perPage,
            order_by=params.sort_by,
        )
    states = [_wallet_state(session, w) for w in page.items]
    return PaginatedList[WalletState](
        items=await _gather(states), pagination=page.pagination
    )


async def _gather(awaitables):
    return list(await asyncio.gather(*awaitables))


@wallet_router.get("/{principal_id}", response_model=WalletState)
async def get_wallet(session: SessionDep, principal_id: int):
    wallet = await _wallet_or_404(session, principal_id)
    return await _wallet_state(session, wallet)


@tenant_router.get("/wallet", response_model=WalletState)
async def get_my_wallet(session: SessionDep, ctx: TenantContextDep):
    """The caller's own wallet, and what a top-up has to cover.

    A wallet is created on first use rather than at signup, so a principal that
    has never been charged has no row. Answering 404 for that would make "you
    have no billing yet" look like an error, so the state is synthesised instead:
    zero balance, nothing owed, not suspended.
    """
    principal_id = ctx.current_principal_id or ctx.user.id
    wallet = (
        await session.exec(
            select(Wallet).where(
                Wallet.principal_id == principal_id, Wallet.deleted_at.is_(None)
            )
        )
    ).first()
    if wallet is None:
        return WalletState(
            id=0,
            principal_id=principal_id,
            principal_name=getattr(ctx.user, "name", None),
            currency="CNY",
            balance=Decimal(0),
            frozen=Decimal(0),
            available=Decimal(0),
            suspended=False,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            outstanding_realtime=await outstanding_realtime_charges(
                session, principal_id
            ),
            outstanding_invoices=await outstanding_invoices(session, principal_id),
            outstanding_total=Decimal(0),
        )
    return await _wallet_state(session, wallet)


@tenant_router.post("/redeem", response_model=RedeemResponse)
async def redeem(session: SessionDep, ctx: TenantContextDep, input: RedeemRequest):
    """Spend a top-up code on the caller's own wallet.

    The code is the idempotency key: it can only be used once, and a concurrent
    second attempt loses the compare-and-set inside ``redeem_code`` rather than
    crediting twice. Errors are the service's — an unknown, spent or expired code
    is a 422 or 409, not a 500.
    """
    principal_id = ctx.current_principal_id or ctx.user.id
    redemption = await redeem_code(
        session,
        code=input.code,
        principal_id=principal_id,
        user_id=ctx.user.id,
    )
    wallet = (
        await session.exec(
            select(Wallet).where(
                Wallet.principal_id == principal_id, Wallet.deleted_at.is_(None)
            )
        )
    ).first()
    return RedeemResponse(
        amount=Decimal(redemption.amount),
        currency=redemption.currency,
        balance=Decimal(wallet.balance) if wallet else Decimal(0),
        suspended=bool(wallet.suspended) if wallet else False,
        used_at=redemption.used_at,
    )


async def _wallet_or_404(session, principal_id: int) -> Wallet:
    wallet = (
        await session.exec(
            select(Wallet).where(
                Wallet.principal_id == principal_id, Wallet.deleted_at.is_(None)
            )
        )
    ).first()
    if wallet is None:
        raise NotFoundException(message=f"wallet for principal {principal_id} not found")
    return wallet


# ---------------------------------------------------------------------------
# Ledger (WP7.1) — the audit trail behind every charge
# ---------------------------------------------------------------------------


@ledger_router.get("", response_model=LedgerEntriesPublic)
async def list_ledger(
    params: LedgerListParams = Depends(),
    principal_id: Optional[int] = None,
    user_id: Optional[int] = None,
    api_key_id: Optional[int] = None,
    sku: Optional[str] = None,
    model_name: Optional[str] = None,
    settle_mode: Optional[str] = None,
    status: Optional[str] = None,
    direction: Optional[str] = None,
    invoice_id: Optional[int] = None,
    request_id: Optional[str] = None,
    occurred_after: Optional[datetime] = None,
    occurred_before: Optional[datetime] = None,
    search: Optional[str] = None,
):
    """Ledger rows, filterable the ways a reconciliation actually asks.

    ``request_id`` is an exact filter rather than a fuzzy one: it is the handle a
    dispute arrives with, and a partial match would return rows from somebody
    else's request. The time range is half-open (``>= after``, ``< before``) so
    paging a period in chunks neither drops nor duplicates a row on a boundary.
    """
    fields = {"deleted_at": None}
    for name, value in (
        ("principal_id", principal_id),
        ("user_id", user_id),
        ("api_key_id", api_key_id),
        ("sku", sku),
        ("model_name", model_name),
        ("settle_mode", settle_mode),
        ("status", status),
        ("direction", direction),
        ("invoice_id", invoice_id),
        ("request_id", request_id),
    ):
        if value is not None:
            fields[name] = value
    fuzzy_fields = {"resource_name": search} if search else {}

    extra_conditions = []
    if occurred_after is not None:
        extra_conditions.append(LedgerEntry.occurred_at >= occurred_after)
    if occurred_before is not None:
        extra_conditions.append(LedgerEntry.occurred_at < occurred_before)

    if params.watch:
        return StreamingResponse(
            LedgerEntry.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await LedgerEntry.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            extra_conditions=extra_conditions or None,
            page=params.page,
            per_page=params.perPage,
            order_by=params.sort_by,
        )


# ---------------------------------------------------------------------------
# Adjustments (WP4.5) — money an operator moves by hand
# ---------------------------------------------------------------------------


@adjustment_router.post("", response_model=AdjustmentPublic)
async def create_adjustment(
    session: SessionDep, ctx: TenantContextDep, input: AdjustmentCreate
):
    """Credit or debit a wallet by hand, once per idempotency key.

    The author is the authenticated caller and cannot be named in the body: an
    adjustment whose operator could be set by whoever sent the request would be
    an anonymous money movement with an audit trail that proves nothing.
    """
    principal_name = await _principal_name(session, input.principal_id)
    adjustment = await apply_adjustment(
        session,
        principal_id=input.principal_id,
        amount=Decimal(input.amount),
        reason=input.reason,
        idempotency_key=input.idempotency_key,
        operator_id=ctx.user.id,
        operator_name=getattr(ctx.user, "name", None),
        principal_name=principal_name,
        currency=input.currency,
    )
    return AdjustmentPublic.model_validate(adjustment, from_attributes=True)


@adjustment_router.get("", response_model=AdjustmentsPublic)
async def list_adjustments(
    params: AdjustmentListParams = Depends(),
    principal_id: Optional[int] = None,
    operator_id: Optional[int] = None,
    idempotency_key: Optional[str] = None,
    search: Optional[str] = None,
):
    """The correction trail. Read-only: a correction of a correction is another
    adjustment, so this list stays complete."""
    fields = {"deleted_at": None}
    if principal_id is not None:
        fields["principal_id"] = principal_id
    if operator_id is not None:
        fields["operator_id"] = operator_id
    if idempotency_key:
        fields["idempotency_key"] = idempotency_key
    fuzzy_fields = {"reason": search} if search else {}

    if params.watch:
        return StreamingResponse(
            Adjustment.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await Adjustment.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            page=params.page,
            per_page=params.perPage,
            order_by=params.sort_by,
        )


async def _principal_name(session, principal_id: int) -> Optional[str]:
    """Snapshot the subject's name onto the adjustment.

    Read from the principals table rather than trusted from the request, and
    optional: a correction against a principal that has since been deleted still
    has to be recordable, and its row is what says who it was for.
    """
    from gpustack.schemas.principals import Principal

    row = (
        await session.exec(select(Principal.name).where(Principal.id == principal_id))
    ).first()
    return row if isinstance(row, str) else None
