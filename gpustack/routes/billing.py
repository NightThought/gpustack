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

import logging
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from gpustack.api.exceptions import (
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
)
from gpustack.schemas.billing import (
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
    unit_for_sku,
)
from gpustack.server.billing_pricing import (
    assert_identity_unchanged,
    assert_window_available,
    compute_amount,
    invalidate_price_cache,
    resolve_price,
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
from gpustack.server.deps import SessionDep

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
