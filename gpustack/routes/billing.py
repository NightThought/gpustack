"""Billing administration API — the price book (WP2).

Platform-scoped and admin-only (mounted under ``admin_routers``): prices are one
tenant-agnostic truth, so unlike the org-owned resources there is no
``owner_principal_id`` to scope by and no tenant filter to apply.

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
from datetime import datetime
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas.billing import (
    PriceBookEntriesPublic,
    PriceBookEntry,
    PriceBookEntryCreate,
    PriceBookEntryListParams,
    PriceBookEntryPublic,
    PriceBookEntryUpdate,
    unit_for_sku,
)
from gpustack.server.billing_pricing import (
    assert_identity_unchanged,
    assert_window_available,
    compute_amount,
    invalidate_price_cache,
    resolve_price,
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
