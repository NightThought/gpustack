"""Invoice query and export endpoints (WP6.3).

Read-only by design — there is no create or update here, because a statement
that could be rewritten through the API would stop being evidence of what was
charged. What these tests pin down is the other half: that a statement can be
found, that its lines sum to its total, that a disputed line can be followed to
the ``request_id`` it came from, and that the exported workbook says the same
things the database does.
"""

import io
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import register_handlers
from gpustack.routes import billing as billing_routes
from gpustack.schemas.billing import (
    SKU_GPU_HOUR_PREFIX,
    SKU_STORAGE_GB_HOUR,
    UNIT_GB_HOURS,
    UNIT_GPU_HOURS,
    Invoice,
    InvoiceItem,
    InvoiceStatus,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    SettleMode,
)
from gpustack.server.deps import get_session
from gpustack.utils.export_delivery import XLSX_MEDIA_TYPE

NOW = datetime(2026, 9, 23, 14, 0, 0)
AUGUST = datetime(2026, 8, 1)
SEPTEMBER = datetime(2026, 9, 1)
JULY = datetime(2026, 7, 1)
PREFIX = "/billing/invoices"

ORG_A = 990101
ORG_B = 990102
SKU_910B = SKU_GPU_HOUR_PREFIX + "910b"


@pytest_asyncio.fixture
async def app_and_engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        for model in (Invoice, InvoiceItem, LedgerEntry):
            await conn.run_sync(model.__table__.create)

    app = FastAPI()
    register_handlers(app)
    app.include_router(billing_routes.invoice_router, prefix=PREFIX)

    async def _session_dep():
        async with AsyncSession(engine) as session:
            yield session

    app.dependency_overrides[get_session] = _session_dep

    @asynccontextmanager
    async def _list_session():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session

    with patch.object(billing_routes, "async_session", _list_session):
        yield app, engine
    await engine.dispose()


@pytest_asyncio.fixture
async def client(app_and_engine):
    app, _ = app_and_engine
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


def _invoice(
    id_,
    *,
    principal_id=ORG_A,
    period_start=AUGUST,
    amount="250.00",
    status=InvoiceStatus.SETTLED,
    name="org-a",
):
    return Invoice(
        id=id_,
        principal_id=principal_id,
        principal_name=name,
        period_start=period_start,
        period_end=period_start + timedelta(days=31),
        amount=Decimal(amount),
        currency="CNY",
        status=status,
        issued_at=NOW,
        settled_at=NOW if status == InvoiceStatus.SETTLED else None,
        unpaid_reason=None if status == InvoiceStatus.SETTLED else "billing:unpaid",
        created_at=NOW,
        updated_at=NOW,
    )


def _item(
    id_,
    invoice_id,
    *,
    sku=SKU_910B,
    unit=UNIT_GPU_HOURS,
    quantity="16",
    amount="200.00",
    model_name=None,
    entry_count=2,
):
    return InvoiceItem(
        id=id_,
        invoice_id=invoice_id,
        sku=sku,
        model_name=model_name,
        quantity=Decimal(quantity),
        unit=unit,
        amount=Decimal(amount),
        entry_count=entry_count,
        created_at=NOW,
        updated_at=NOW,
    )


def _entry(
    id_,
    invoice_id,
    *,
    sku=SKU_910B,
    quantity="12",
    amount="150.00",
    request_id="req-abc-123",
    occurred_at=AUGUST + timedelta(hours=1),
    resource_name="worker-1",
):
    return LedgerEntry(
        id=id_,
        source_table="metered_usage",
        source_id=id_,
        principal_id=ORG_A,
        principal_name="org-a",
        resource_name=resource_name,
        request_id=request_id,
        model_name=None,
        sku=sku,
        quantity=Decimal(quantity),
        unit=UNIT_GPU_HOURS,
        unit_price=Decimal("12.5"),
        amount=Decimal(amount),
        currency="CNY",
        direction=LedgerDirection.DEBIT,
        settle_mode=SettleMode.DEFERRED,
        status=LedgerStatus.SETTLED,
        invoice_id=invoice_id,
        occurred_at=occurred_at,
        created_at=NOW,
        updated_at=NOW,
    )


async def _seed(engine, *rows):
    async with AsyncSession(engine) as s:
        for row in rows:
            s.add(row)
        await s.commit()


async def _seed_one_statement(engine):
    """August for ORG_A: two lines, three entries, one of them storage."""
    await _seed(
        engine,
        _invoice(1, amount="250.01"),
        _item(1, 1, sku=SKU_910B, quantity="16", amount="200.00", entry_count=2),
        _item(
            2,
            1,
            sku=SKU_STORAGE_GB_HOUR,
            unit=UNIT_GB_HOURS,
            quantity="100",
            amount="50.01",
            entry_count=1,
        ),
        _entry(1, 1, request_id="req-gpu-1", amount="150.00"),
        _entry(
            2,
            1,
            request_id="req-gpu-2",
            amount="50.00",
            occurred_at=AUGUST + timedelta(hours=2),
        ),
        _entry(
            3,
            1,
            sku=SKU_STORAGE_GB_HOUR,
            request_id=None,
            amount="50.01",
            resource_name="pv-data",
            occurred_at=AUGUST + timedelta(hours=3),
        ),
    )


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_is_paginated(client, app_and_engine):
    _, engine = app_and_engine
    await _seed(
        engine,
        _invoice(1, principal_id=ORG_A, period_start=AUGUST),
        _invoice(
            2, principal_id=ORG_B, period_start=AUGUST, name="org-b", amount="80.00"
        ),
        _invoice(3, principal_id=ORG_A, period_start=JULY, amount="120.00"),
    )

    response = await client.get(PREFIX)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pagination"]["total"] == 3
    assert len(body["items"]) == 3
    # Never the whole editable surface: an invoice has no write API at all.
    assert "amount" in body["items"][0]
    assert body["items"][0]["currency"] == "CNY"


@pytest.mark.asyncio
async def test_list_filters_by_org_and_status(client, app_and_engine):
    _, engine = app_and_engine
    await _seed(
        engine,
        _invoice(1, principal_id=ORG_A, status=InvoiceStatus.SETTLED),
        _invoice(
            2,
            principal_id=ORG_B,
            name="org-b",
            status=InvoiceStatus.ISSUED,
            amount="80.00",
        ),
    )

    only_a = await client.get(PREFIX, params={"principal_id": ORG_A})
    assert only_a.json()["pagination"]["total"] == 1

    only_unpaid = await client.get(
        PREFIX, params={"status": InvoiceStatus.ISSUED.value}
    )
    assert only_unpaid.json()["pagination"]["total"] == 1
    assert only_unpaid.json()["items"][0]["principal_id"] == ORG_B
    assert only_unpaid.json()["items"][0]["unpaid_reason"] == "billing:unpaid"


@pytest.mark.asyncio
async def test_list_filters_by_period_overlap(client, app_and_engine):
    """A month's worth of statements is one call, not one per org."""
    _, engine = app_and_engine
    await _seed(
        engine,
        _invoice(1, period_start=JULY, amount="10.00"),
        _invoice(2, period_start=AUGUST, amount="20.00"),
        _invoice(3, period_start=SEPTEMBER, amount="30.00"),
    )

    response = await client.get(
        PREFIX,
        params={
            "period_start": AUGUST.isoformat(),
            "period_end": (SEPTEMBER + timedelta(days=1)).isoformat(),
        },
    )

    periods = sorted(item["period_start"] for item in response.json()["items"])
    assert len(periods) == 2
    assert all("2026-08" in p or "2026-09" in p for p in periods)


@pytest.mark.asyncio
async def test_list_searches_by_org_name(client, app_and_engine):
    _, engine = app_and_engine
    await _seed(
        engine,
        _invoice(1, name="acme-corp"),
        _invoice(2, principal_id=ORG_B, name="globex", amount="10.00"),
    )

    response = await client.get(PREFIX, params={"search": "glob"})

    assert response.json()["pagination"]["total"] == 1
    assert response.json()["items"][0]["principal_name"] == "globex"


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detail_carries_the_lines_that_make_up_its_total(client, app_and_engine):
    _, engine = app_and_engine
    await _seed_one_statement(engine)

    response = await client.get(f"{PREFIX}/1")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == 1
    assert body["status"] == InvoiceStatus.SETTLED.value
    assert len(body["items"]) == 2
    by_sku = {item["sku"]: item for item in body["items"]}
    assert by_sku[SKU_910B]["entry_count"] == 2
    assert by_sku[SKU_STORAGE_GB_HOUR]["unit"] == UNIT_GB_HOURS
    # The lines sum to the statement, which is what makes it checkable by eye.
    assert sum(Decimal(i["amount"]) for i in body["items"]) == Decimal(body["amount"])


@pytest.mark.asyncio
async def test_an_unknown_statement_is_a_404(client):
    response = await client.get(f"{PREFIX}/4242")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Entries — traceability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entries_expose_the_request_id_behind_each_charge(client, app_and_engine):
    """The answer to "why was I charged this" has to reach the source row."""
    _, engine = app_and_engine
    await _seed_one_statement(engine)

    response = await client.get(f"{PREFIX}/1/entries")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 3
    assert Decimal(body["invoice_amount"]) == Decimal("250.01")
    ids = {item["request_id"] for item in body["items"]}
    assert "req-gpu-1" in ids and "req-gpu-2" in ids
    # A resource charge has no request behind it; the source columns say where it
    # came from instead.
    storage = [i for i in body["items"] if i["sku"] == SKU_STORAGE_GB_HOUR][0]
    assert storage["request_id"] is None
    assert storage["source_table"] == "metered_usage"
    assert storage["resource_name"] == "pv-data"
    # And the entries sum to the statement.
    assert sum(Decimal(i["amount"]) for i in body["items"]) == Decimal(
        body["invoice_amount"]
    )


@pytest.mark.asyncio
async def test_entries_are_paginated(client, app_and_engine):
    _, engine = app_and_engine
    await _seed_one_statement(engine)

    page = await client.get(f"{PREFIX}/1/entries", params={"page": 1, "per_page": 2})

    body = page.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2


@pytest.mark.asyncio
async def test_entries_of_an_unknown_statement_is_a_404(client):
    assert (await client.get(f"{PREFIX}/4242/entries")).status_code == 404


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_is_an_attachment_named_for_the_period(client, app_and_engine):
    _, engine = app_and_engine
    await _seed_one_statement(engine)

    response = await client.get(f"{PREFIX}/1/export")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == XLSX_MEDIA_TYPE
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert "invoice-990101-20260801-20260901.xlsx" in disposition


@pytest.mark.asyncio
async def test_exported_workbook_agrees_with_the_database(client, app_and_engine):
    """Consistency, checked against the file rather than the code that wrote it.

    An xlsx is a zip of XML, and xlsxwriter writes inline strings, so the values
    can be read back out of the sheets without a spreadsheet library — which is
    the point: asserting on the payload rather than on our own serializer is what
    makes this a check and not a restatement.
    """
    _, engine = app_and_engine
    await _seed_one_statement(engine)

    response = await client.get(f"{PREFIX}/1/export")
    workbook = zipfile.ZipFile(io.BytesIO(response.content))

    names = workbook.namelist()
    assert "xl/workbook.xml" in names
    sheets = [n for n in names if n.startswith("xl/worksheets/sheet")]
    assert len(sheets) == 3

    declared = workbook.read("xl/workbook.xml").decode()
    for sheet_name in ("Invoice", "Items", "Entries"):
        assert sheet_name in declared

    def text_of(index: int) -> str:
        return workbook.read(f"xl/worksheets/sheet{index}.xml").decode()

    invoice_sheet, items_sheet, entries_sheet = (text_of(i) for i in (1, 2, 3))

    # Header: the statement's own fields.
    assert "250.01" in invoice_sheet
    assert InvoiceStatus.SETTLED.value in invoice_sheet
    assert "org-a" in invoice_sheet
    # Items: both lines, with the counts that make them up.
    assert SKU_910B in items_sheet
    assert SKU_STORAGE_GB_HOUR in items_sheet
    assert "200" in items_sheet
    # Entries: the traceability columns, including a request id and the source.
    assert "req-gpu-1" in entries_sheet
    assert "req-gpu-2" in entries_sheet
    assert "metered_usage" in entries_sheet
    assert "pv-data" in entries_sheet
    # Column headers are written, so a reader knows what a column is.
    assert "request_id" in entries_sheet
    assert "entry_count" in items_sheet


@pytest.mark.asyncio
async def test_export_of_an_unknown_statement_is_a_404(client):
    assert (await client.get(f"{PREFIX}/4242/export")).status_code == 404


@pytest.mark.asyncio
async def test_an_unpaid_statement_exports_with_its_reason(client, app_and_engine):
    """The document an org in arrears is given has to say why it is unpaid."""
    _, engine = app_and_engine
    await _seed(
        engine,
        _invoice(7, amount="150.00", status=InvoiceStatus.ISSUED),
        _item(1, 7, amount="150.00", entry_count=1),
        _entry(1, 7, amount="150.00"),
    )

    response = await client.get(f"{PREFIX}/7/export")

    sheet = (
        zipfile.ZipFile(io.BytesIO(response.content))
        .read("xl/worksheets/sheet1.xml")
        .decode()
    )
    assert InvoiceStatus.ISSUED.value in sheet
    assert "billing:unpaid" in sheet
