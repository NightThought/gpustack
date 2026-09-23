"""Quota ceilings and the pre-check that enforces them (WP5.4).

Four things have to hold, and each has a test below:

* a caller with no ceilings costs the request path nothing — no query at all,
  which is the property that keeps this off the critical path of every inference
  call;
* a ceiling bites at ``>=`` its limit, in the window it names, counted from the
  ledger rather than from a counter a tenant could reset;
* the counter advances with the ledger rows it counts and never twice for one
  row, and a window rolls by comparison so a daily limit needs no cron job;
* 429 is what a ceiling raises, never 402 — the two tell a client to do
  different things.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import TooManyRequestsException
from gpustack.schemas.api_keys import ApiKey, PermissionScope
from gpustack.schemas.billing import (
    SKU_GPU_HOUR_PREFIX,
    SKU_TOKEN_PROMPT,
    SKU_WALLET_TOPUP,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    Quota,
    QuotaLimitType,
    QuotaScope,
    SettleMode,
    UNIT_TOKENS,
)
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.security import new_secret_key_digest
from gpustack.server import billing_quota
from gpustack.server.billing_quota import (
    apply_usage,
    assert_no_duplicate,
    check_quota,
    enabled_quota_specs,
    invalidate_quota_cache,
    specs_for_caller,
    usage_in_window,
    window_start_for,
)
from gpustack.server.gateway_auth_reconciler import build_local_auth_tables

NOW = datetime(2026, 9, 23, 14, 30, 0, tzinfo=timezone.utc)
DAY_START = datetime(2026, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
MONTH_START = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)

ORG = 990101
USER = 990201
KEY = 58
MODEL = "Qwen3-8B"
SECRET_KEY = "31c7e2c1d0a447a1b0f0e6d5c4b3a291"


@pytest_asyncio.fixture
async def engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        for model in (Quota, LedgerEntry, ApiKey, Principal):
            await conn.run_sync(model.__table__.create)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s


@pytest_asyncio.fixture
async def session_factory(engine):
    """Stands in for the app-wide factory the rollover write opens its own of."""

    @asynccontextmanager
    async def _factory():
        async with AsyncSession(engine, expire_on_commit=False) as s:
            yield s

    invalidate_quota_cache()
    with patch.object(billing_quota, "async_session", _factory, create=True):
        with patch("gpustack.server.db.async_session", _factory):
            yield _factory
    invalidate_quota_cache()


def _quota(
    id_=1,
    scope=QuotaScope.API_KEY,
    limit_type=QuotaLimitType.DAILY_TOKENS,
    limit_value="1000",
    model_name=None,
    api_key_id=KEY,
    user_id=None,
    principal_id=None,
    enabled=True,
    window_start=None,
    used="0",
):
    if scope == QuotaScope.USER:
        api_key_id, user_id = None, user_id or USER
    elif scope == QuotaScope.ORGANIZATION:
        api_key_id, principal_id = None, principal_id or ORG
    return Quota(
        id=id_,
        scope=scope,
        api_key_id=api_key_id,
        user_id=user_id,
        principal_id=principal_id,
        model_name=model_name,
        limit_type=limit_type,
        limit_value=Decimal(str(limit_value)),
        enabled=enabled,
        window_start=window_start,
        used=Decimal(str(used)),
        created_at=NOW,
        updated_at=NOW,
    )


def _ledger(
    id_,
    *,
    sku=SKU_TOKEN_PROMPT,
    quantity="100",
    amount="0.20",
    occurred_at=NOW,
    api_key_id=KEY,
    user_id=USER,
    principal_id=ORG,
    model_name=MODEL,
    direction=LedgerDirection.DEBIT,
    status=LedgerStatus.PENDING,
):
    return LedgerEntry(
        id=id_,
        source_table="model_usage_details",
        source_id=id_,
        api_key_id=api_key_id,
        user_id=user_id,
        principal_id=principal_id,
        model_name=model_name,
        sku=sku,
        quantity=Decimal(str(quantity)),
        unit=UNIT_TOKENS,
        unit_price=Decimal("0.002"),
        amount=Decimal(str(amount)),
        currency="CNY",
        direction=direction,
        settle_mode=SettleMode.REALTIME,
        status=status,
        occurred_at=occurred_at,
        created_at=NOW,
        updated_at=NOW,
    )


async def _seed(session, *rows):
    for row in rows:
        session.add(row)
    await session.commit()


async def _quota_row(engine, id_=1):
    async with AsyncSession(engine) as s:
        return (await s.exec(select(Quota).where(Quota.id == id_))).first()


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def test_daily_windows_start_at_utc_midnight():
    assert window_start_for(QuotaLimitType.DAILY_TOKENS, NOW) == DAY_START
    assert window_start_for(QuotaLimitType.DAILY_AMOUNT, NOW) == DAY_START


def test_monthly_windows_start_on_the_first():
    assert window_start_for(QuotaLimitType.MONTHLY_AMOUNT, NOW) == MONTH_START


def test_a_naive_instant_is_read_as_utc():
    """UTCDateTime hands back aware values; callers may not."""
    assert window_start_for(QuotaLimitType.DAILY_TOKENS, NOW.replace(tzinfo=None)) == (
        DAY_START
    )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_matching_is_by_the_column_the_scope_names():
    specs = [
        _spec(1, QuotaScope.API_KEY, api_key_id=KEY),
        _spec(2, QuotaScope.USER, user_id=USER),
        _spec(3, QuotaScope.ORGANIZATION, principal_id=ORG),
        # Same subject id under a different scope must not match: user 58 and
        # key 58 are different things that happen to share a number.
        _spec(4, QuotaScope.USER, user_id=KEY),
    ]
    matched = specs_for_caller(
        specs, api_key_id=KEY, user_id=USER, principal_id=ORG, model_name=MODEL
    )
    assert sorted(s.id for s in matched) == [1, 2, 3]


def test_a_null_model_matches_every_model_and_a_named_one_only_itself():
    specs = [
        _spec(1, QuotaScope.API_KEY, api_key_id=KEY, model_name=None),
        _spec(2, QuotaScope.API_KEY, api_key_id=KEY, model_name=MODEL),
        _spec(3, QuotaScope.API_KEY, api_key_id=KEY, model_name="Qwen3-30B-A3B"),
    ]
    assert sorted(
        s.id
        for s in specs_for_caller(specs, api_key_id=KEY, model_name=MODEL)
    ) == [1, 2]


def test_a_caller_with_no_ids_matches_nothing():
    specs = [_spec(1, QuotaScope.API_KEY, api_key_id=KEY)]
    assert specs_for_caller(specs, model_name=MODEL) == []


def _spec(id_, scope, *, api_key_id=None, user_id=None, principal_id=None,
          model_name=None, limit_type=QuotaLimitType.DAILY_TOKENS,
          limit_value="1000"):
    from gpustack.server.billing_quota import QuotaSpec

    return QuotaSpec(
        id=id_,
        scope=scope,
        api_key_id=api_key_id,
        user_id=user_id,
        principal_id=principal_id,
        model_name=model_name,
        limit_type=limit_type,
        limit_value=Decimal(str(limit_value)),
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_caller_with_no_quotas_costs_no_query():
    """The property that keeps this off the hot path of every inference call."""

    class _NoQueries:
        def __getattr__(self, name):
            raise AssertionError(f"check_quota must not query for this caller ({name})")

    invalidate_quota_cache()
    with patch.object(billing_quota, "enabled_quota_specs", _async_return([])):
        await check_quota(
            _NoQueries(), api_key_id=KEY, user_id=USER, principal_id=ORG,
            model_name=MODEL,
        )


def _async_return(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


@pytest.mark.asyncio
async def test_under_the_ceiling_is_admitted(session_factory):
    async with session_factory() as s:
        await _seed(
            s,
            _quota(limit_value="1000", window_start=DAY_START, used="999"),
        )
    invalidate_quota_cache()
    async with session_factory() as s:
        await check_quota(
            s, api_key_id=KEY, user_id=USER, principal_id=ORG, model_name=MODEL
        )


@pytest.mark.asyncio
async def test_reaching_the_ceiling_is_refused_with_429(session_factory):
    """>= and not >: the ceiling is the last unit admitted."""
    async with session_factory() as s:
        await _seed(s, _quota(limit_value="1000", window_start=DAY_START, used="1000"))
    invalidate_quota_cache()
    async with session_factory() as s:
        with pytest.raises(TooManyRequestsException) as exc_info:
            await check_quota(
                s, api_key_id=KEY, user_id=USER, principal_id=ORG, model_name=MODEL
            )

    assert exc_info.value.status_code == 429
    assert "daily_tokens" in exc_info.value.message
    assert MODEL in exc_info.value.message


@pytest.mark.asyncio
async def test_a_monthly_spend_ceiling_is_refused_with_429(session_factory):
    async with session_factory() as s:
        await _seed(
            s,
            _quota(
                scope=QuotaScope.ORGANIZATION,
                limit_type=QuotaLimitType.MONTHLY_AMOUNT,
                limit_value="500",
                window_start=MONTH_START,
                used="500.00000001",
            ),
        )
    invalidate_quota_cache()
    async with session_factory() as s:
        with pytest.raises(TooManyRequestsException) as exc_info:
            await check_quota(
                s, api_key_id=KEY, user_id=USER, principal_id=ORG, model_name=MODEL
            )
    assert exc_info.value.status_code == 429
    assert "monthly_amount" in exc_info.value.message


@pytest.mark.asyncio
async def test_a_disabled_ceiling_does_not_refuse(session_factory):
    async with session_factory() as s:
        await _seed(
            s, _quota(limit_value="1", enabled=False, window_start=DAY_START, used="99")
        )
    invalidate_quota_cache()
    async with session_factory() as s:
        await check_quota(
            s, api_key_id=KEY, user_id=USER, principal_id=ORG, model_name=MODEL
        )


@pytest.mark.asyncio
async def test_a_stricter_key_ceiling_bites_before_a_looser_org_one(
    session_factory,
):
    """Several ceilings can bind one request; the first exhausted one refuses."""
    async with session_factory() as s:
        await _seed(
            s,
            _quota(
                id_=1, limit_value="1000000", window_start=DAY_START, used="10"
            ),
            _quota(
                id_=2,
                scope=QuotaScope.ORGANIZATION,
                limit_type=QuotaLimitType.DAILY_AMOUNT,
                limit_value="100",
                window_start=DAY_START,
                used="100",
            ),
        )
    invalidate_quota_cache()
    async with session_factory() as s:
        with pytest.raises(TooManyRequestsException) as exc_info:
            await check_quota(
                s, api_key_id=KEY, user_id=USER, principal_id=ORG, model_name=MODEL
            )
    assert "daily_amount" in exc_info.value.message


@pytest.mark.asyncio
async def test_an_unreadable_counter_is_allowed_through(session_factory, caplog):
    """Refusing inference because billing could not be read turns a billing
    problem into an outage; the exposure is bounded by the window."""
    invalidate_quota_cache()
    with patch.object(
        billing_quota, "enabled_quota_specs", _async_return([_spec(1, QuotaScope.API_KEY, api_key_id=KEY)])
    ):
        with patch.object(billing_quota, "window_used", _raising):
            with caplog.at_level("WARNING", logger="gpustack.server.billing_quota"):
                await check_quota(
                    session_factory, api_key_id=KEY, model_name=MODEL
                )
    assert "could not evaluate quota 1" in caplog.text


async def _raising(*args, **kwargs):
    raise RuntimeError("billing is unreachable")


# ---------------------------------------------------------------------------
# Window rollover, recomputed from the ledger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_new_window_is_recomputed_from_the_ledger(session_factory, engine):
    """Yesterday's counter must not refuse today's first request."""
    yesterday = DAY_START - timedelta(days=1)
    async with session_factory() as s:
        await _seed(
            s,
            _quota(limit_value="1000", window_start=yesterday, used="5000"),
            # Yesterday, over the ceiling then; today, one small request.
            _ledger(1, quantity="5000", occurred_at=yesterday + timedelta(hours=1)),
            _ledger(2, quantity="10", occurred_at=NOW),
        )
    invalidate_quota_cache()
    async with session_factory() as s:
        await check_quota(
            s, api_key_id=KEY, user_id=USER, principal_id=ORG, model_name=MODEL
        )

    row = await _quota_row(engine)
    assert row.window_start.replace(tzinfo=timezone.utc) == DAY_START
    # Only today's usage counts in today's window.
    assert row.used == Decimal("10")


@pytest.mark.asyncio
async def test_a_rollover_can_still_refuse(session_factory):
    async with session_factory() as s:
        await _seed(
            s,
            _quota(limit_value="100", window_start=DAY_START - timedelta(days=1),
                   used="0"),
            _ledger(1, quantity="250", occurred_at=NOW),
        )
    invalidate_quota_cache()
    async with session_factory() as s:
        with pytest.raises(TooManyRequestsException):
            await check_quota(
                s, api_key_id=KEY, user_id=USER, principal_id=ORG, model_name=MODEL
            )


@pytest.mark.asyncio
async def test_a_ceiling_created_mid_window_counts_usage_already_in_it(
    session_factory, engine
):
    """``window_start`` NULL means "never established", not "nothing used"."""
    async with session_factory() as s:
        await _seed(
            s,
            _quota(limit_value="100", window_start=None, used="0"),
            _ledger(1, quantity="60", occurred_at=DAY_START + timedelta(hours=1)),
            _ledger(2, quantity="60", occurred_at=DAY_START + timedelta(hours=2)),
        )
    invalidate_quota_cache()
    async with session_factory() as s:
        with pytest.raises(TooManyRequestsException):
            await check_quota(
                s, api_key_id=KEY, user_id=USER, principal_id=ORG, model_name=MODEL
            )


@pytest.mark.asyncio
async def test_a_top_up_does_not_raise_a_spend_ceiling(session_factory):
    """A CREDIT row is money arriving, not allowance being consumed."""
    async with session_factory() as s:
        await _seed(
            s,
            _quota(
                limit_type=QuotaLimitType.DAILY_AMOUNT,
                limit_value="10",
                window_start=DAY_START,
                used="0",
            ),
            _ledger(
                1,
                sku=SKU_WALLET_TOPUP,
                quantity="1",
                amount="1000",
                direction=LedgerDirection.CREDIT,
            ),
        )
    invalidate_quota_cache()
    async with session_factory() as s:
        total = await usage_in_window(
            s,
            spec=(await enabled_quota_specs(s))[0],
            window_start=DAY_START,
        )
    assert total == Decimal("0")


@pytest.mark.asyncio
async def test_an_unpriced_placeholder_does_not_count(session_factory):
    async with session_factory() as s:
        await _seed(
            s,
            _quota(limit_value="100", window_start=DAY_START, used="0"),
            _ledger(1, quantity="9999", status=LedgerStatus.VOID, amount="0"),
        )
    invalidate_quota_cache()
    async with session_factory() as s:
        await check_quota(
            s, api_key_id=KEY, user_id=USER, principal_id=ORG, model_name=MODEL
        )


@pytest.mark.asyncio
async def test_a_model_scoped_ceiling_only_counts_that_model(session_factory):
    async with session_factory() as s:
        await _seed(
            s,
            _quota(limit_value="100", model_name=MODEL, window_start=DAY_START),
            _ledger(1, quantity="5000", model_name="Qwen3-30B-A3B"),
        )
    invalidate_quota_cache()
    async with session_factory() as s:
        await check_quota(
            s, api_key_id=KEY, user_id=USER, principal_id=ORG, model_name=MODEL
        )


# ---------------------------------------------------------------------------
# Counter advancement (the rater's half)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_usage_advances_the_matching_counter(session_factory, engine):
    async with session_factory() as s:
        await _seed(s, _quota(limit_value="1000", window_start=DAY_START, used="0"))
    invalidate_quota_cache()

    async with session_factory() as s:
        advanced = await apply_usage(s, [_ledger(1, quantity="250")])
        await s.commit()

    assert advanced == 1
    assert (await _quota_row(engine)).used == Decimal("250")


@pytest.mark.asyncio
async def test_apply_usage_advances_every_ceiling_the_entry_binds(
    session_factory, engine
):
    """One request can count against a key cap, a user cap and an org cap."""
    async with session_factory() as s:
        await _seed(
            s,
            _quota(id_=1, limit_value="1000", window_start=DAY_START),
            _quota(id_=2, scope=QuotaScope.USER, limit_type=QuotaLimitType.DAILY_AMOUNT,
                   limit_value="100", window_start=DAY_START),
            _quota(id_=3, scope=QuotaScope.ORGANIZATION,
                   limit_type=QuotaLimitType.MONTHLY_AMOUNT, limit_value="500",
                   window_start=MONTH_START),
        )
    invalidate_quota_cache()

    async with session_factory() as s:
        advanced = await apply_usage(s, [_ledger(1, quantity="250", amount="0.50")])
        await s.commit()

    assert advanced == 3
    async with AsyncSession(engine) as s:
        rows = {row.id: row for row in (await s.exec(select(Quota))).all()}
    assert rows[1].used == Decimal("250")  # tokens
    assert rows[2].used == Decimal("0.50")  # currency
    assert rows[3].used == Decimal("0.50")


@pytest.mark.asyncio
async def test_apply_usage_keeps_windows_apart(session_factory, engine):
    """A backfill spanning midnight must not pour yesterday into today."""
    yesterday = DAY_START - timedelta(hours=2)
    async with session_factory() as s:
        await _seed(
            s,
            _quota(limit_value="1000", window_start=DAY_START, used="0"),
        )
    invalidate_quota_cache()

    async with session_factory() as s:
        await apply_usage(
            s,
            [
                _ledger(1, quantity="400", occurred_at=yesterday),
                _ledger(2, quantity="50", occurred_at=NOW),
            ],
        )
        await s.commit()

    # The row's window is today, so only today's delta lands on it.
    row = await _quota_row(engine)
    assert row.window_start.replace(tzinfo=timezone.utc) == DAY_START
    assert row.used == Decimal("50")


@pytest.mark.asyncio
async def test_apply_usage_skips_placeholders_and_unrelated_entries(
    session_factory, engine
):
    async with session_factory() as s:
        await _seed(s, _quota(limit_value="1000", window_start=DAY_START, used="0"))
    invalidate_quota_cache()

    async with session_factory() as s:
        advanced = await apply_usage(
            s,
            [
                _ledger(1, quantity="250", status=LedgerStatus.VOID, amount="0"),
                # Another key entirely.
                _ledger(2, quantity="999", api_key_id=KEY + 1, user_id=USER + 1),
                # A resource charge: not tokens, so it must not feed a token cap.
                _ledger(
                    3,
                    sku=SKU_GPU_HOUR_PREFIX + "910b",
                    quantity="12",
                    api_key_id=None,
                    user_id=None,
                ),
            ],
        )
        await s.commit()

    assert advanced == 0
    assert (await _quota_row(engine)).used == Decimal("0")


@pytest.mark.asyncio
async def test_apply_usage_establishes_a_window_it_finds_unstarted(
    session_factory, engine
):
    """The recompute path counts the applied entries, which the session sees.

    Staged here exactly as the rater stages them: ``apply_usage`` recomputes
    from the ledger rather than adding a delta to a counter that does not exist
    yet, so the rows it is handed must already be in the session or the new
    window would start short by precisely those rows.
    """
    async with session_factory() as s:
        await _seed(s, _quota(limit_value="1000", window_start=None, used="0"))
        await _seed(s, _ledger(1, quantity="100", occurred_at=NOW))
    invalidate_quota_cache()

    staged = _ledger(2, quantity="50")
    async with session_factory() as s:
        s.add(staged)
        await s.flush()
        advanced = await apply_usage(s, [staged])
        await s.commit()

    assert advanced == 1
    row = await _quota_row(engine)
    assert row.window_start.replace(tzinfo=timezone.utc) == DAY_START
    assert row.used == Decimal("150")


@pytest.mark.asyncio
async def test_apply_usage_with_no_ceilings_configured(session_factory):
    async with session_factory() as s:
        assert await apply_usage(s, [_ledger(1)]) == 0


# ---------------------------------------------------------------------------
# The duplicate guard the unique constraint cannot provide
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_second_all_model_ceiling_for_one_key_is_rejected(session_factory):
    """NULLs are distinct in SQL, so the constraint alone would allow both."""
    async with session_factory() as s:
        await _seed(s, _quota(model_name=None, limit_value="1000"))

    from gpustack.api.exceptions import InvalidException

    async with session_factory() as s:
        with pytest.raises(InvalidException) as exc_info:
            await assert_no_duplicate(
                s,
                scope=QuotaScope.API_KEY,
                api_key_id=KEY,
                model_name=None,
                limit_type=QuotaLimitType.DAILY_TOKENS,
            )
    assert "already exists" in exc_info.value.message


@pytest.mark.asyncio
async def test_a_different_limit_type_or_model_is_not_a_duplicate(session_factory):
    async with session_factory() as s:
        await _seed(
            s,
            _quota(id_=1, model_name=None, limit_value="1000"),
            _quota(id_=2, model_name=MODEL, limit_value="10"),
        )

    async with session_factory() as s:
        # Same subject and model as row 1, different limit type.
        await assert_no_duplicate(
            s,
            scope=QuotaScope.API_KEY,
            api_key_id=KEY,
            limit_type=QuotaLimitType.DAILY_AMOUNT,
        )
        # Same limit type, different model.
        await assert_no_duplicate(
            s,
            scope=QuotaScope.API_KEY,
            api_key_id=KEY,
            model_name="Qwen3-30B-A3B",
            limit_type=QuotaLimitType.DAILY_TOKENS,
        )


@pytest.mark.asyncio
async def test_updating_a_row_is_not_a_duplicate_of_itself(session_factory):
    async with session_factory() as s:
        await _seed(s, _quota(id_=7, limit_value="1000"))

    async with session_factory() as s:
        await assert_no_duplicate(
            s,
            scope=QuotaScope.API_KEY,
            api_key_id=KEY,
            limit_type=QuotaLimitType.DAILY_TOKENS,
            exclude_id=7,
        )


# ---------------------------------------------------------------------------
# Gateway path: a ceiling has to make the key ask the server
# ---------------------------------------------------------------------------


def _principal(id_, kind=PrincipalType.ORG, name="acme"):
    return Principal(
        id=id_, kind=kind, name=name, display_name=name, source="local",
        is_admin=False, is_active=True, created_at=NOW, updated_at=NOW,
    )


def _api_key(id_=KEY, access_key="ak-1", user_id=USER, owner_principal_id=ORG):
    return ApiKey(
        id=id_,
        name=f"key{id_}",
        access_key=access_key,
        hashed_secret_key=f"argon2-{id_}",
        secret_key_digest=new_secret_key_digest(
            secret_key=SECRET_KEY, is_custom=False, access_key=access_key
        ),
        scope=[PermissionScope.ALL],
        user_id=user_id,
        owner_principal_id=owner_principal_id,
        is_custom=False,
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.mark.asyncio
async def test_an_unlimited_key_stays_verifiable_locally(session_factory):
    async with session_factory() as s:
        await _seed(s, _principal(ORG), _principal(USER, PrincipalType.USER, "u"),
                    _api_key())
    keys, _ = await _tables(session_factory)
    assert keys["ak-1"].get("unrestricted") is True


@pytest.mark.asyncio
async def test_a_ceiling_on_the_key_withholds_the_local_shortcut(session_factory):
    """Without this the plugin answers locally and the ceiling is never asked
    about — enforcement would look configured and do nothing."""
    async with session_factory() as s:
        await _seed(s, _principal(ORG), _principal(USER, PrincipalType.USER, "u"),
                    _api_key(), _quota(limit_value="1000"))
    keys, _ = await _tables(session_factory)
    assert "unrestricted" not in keys["ak-1"]
    # Still published, so the gateway can authenticate it; it just has to ask.
    assert "digest" in keys["ak-1"]


@pytest.mark.asyncio
async def test_an_org_ceiling_withholds_the_shortcut_for_its_keys(session_factory):
    async with session_factory() as s:
        await _seed(
            s, _principal(ORG), _principal(USER, PrincipalType.USER, "u"),
            _api_key(),
            _quota(scope=QuotaScope.ORGANIZATION,
                   limit_type=QuotaLimitType.MONTHLY_AMOUNT, limit_value="500"),
        )
    keys, _ = await _tables(session_factory)
    assert "unrestricted" not in keys["ak-1"]


@pytest.mark.asyncio
async def test_a_disabled_ceiling_does_not_cost_the_shortcut(session_factory):
    async with session_factory() as s:
        await _seed(s, _principal(ORG), _principal(USER, PrincipalType.USER, "u"),
                    _api_key(), _quota(limit_value="1000", enabled=False))
    keys, _ = await _tables(session_factory)
    assert keys["ak-1"].get("unrestricted") is True


async def _tables(session_factory):
    async with session_factory() as s:
        return await build_local_auth_tables(s)


@pytest.mark.asyncio
async def test_gateway_auth_survives_billing_being_unreadable(
    session_factory, caplog
):
    """Publishing credentials must not depend on the billing tables being there.

    An exception escaping the quota read would not degrade quota enforcement, it
    would stop gateway authentication for every key in the deployment.
    """
    async with session_factory() as s:
        await _seed(s, _principal(ORG), _principal(USER, PrincipalType.USER, "u"),
                    _api_key())

    class _Broken:
        async def exec(self, *args, **kwargs):
            raise RuntimeError("no such table: billing_quota")

    from gpustack.server.gateway_auth_reconciler import _quota_subjects

    with caplog.at_level(
        "WARNING", logger="gpustack.server.gateway_auth_reconciler"
    ):
        subjects = await _quota_subjects(_Broken())

    assert (subjects.api_key_ids, subjects.user_ids, subjects.principal_ids) == (
        frozenset(), frozenset(), frozenset(),
    )
    assert "could not read billing quotas" in caplog.text


def test_a_stub_credential_does_not_look_limited():
    """The reconciler's quota subjects come from the table, never from a stub."""
    subjects = SimpleNamespace(
        api_key_ids=frozenset(), user_ids=frozenset(), principal_ids=frozenset()
    )
    from gpustack.server.gateway_auth_reconciler import gateway_key_unrestricted

    assert gateway_key_unrestricted(
        [PermissionScope.ALL], None, PrincipalType.USER,
        has_active_quota=(KEY in subjects.api_key_ids),
    )
    assert not gateway_key_unrestricted(
        [PermissionScope.ALL], None, PrincipalType.USER, has_active_quota=True
    )
