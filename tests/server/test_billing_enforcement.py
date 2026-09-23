"""Billing enforcement tests (WP5).

What has to hold for a suspension to actually stop traffic:

* flagging a wallet flags the keys that spend from it — matched by owning Org
  *and* by direct principal, so a key with no Org cannot slip through;
* resuming clears only what billing suspended, never an admin's suspension;
* both request paths refuse with 402 — the key flag on the gateway path, the
  wallet flag as the backstop for a key created after the suspension;
* the gateway's local auth table excludes a suspended key, which is the whole
  mechanism: a key the plugin can verify locally never reaches the server, so
  exclusion is what makes the 402 reachable at all.
"""

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
import pytest_asyncio
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from unittest.mock import patch

from gpustack.api.exceptions import PaymentRequiredException
from gpustack.schemas.api_keys import ApiKey, PermissionScope
from gpustack.schemas.billing import (
    Invoice,
    SKU_TOKEN_PROMPT,
    BillingSession,
    LedgerDirection,
    LedgerEntry,
    LedgerStatus,
    SettleMode,
    Wallet,
)
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.security import new_secret_key_digest
from gpustack.server import billing_settlement
from gpustack.server.billing_enforcement import (
    BILLING_REASON_PREFIX,
    REASON_ARREARS,
    assert_billing_active,
    inherited_suspension,
    resume_keys_for_wallet,
    suspend_keys_for_wallet,
)
from gpustack.server.billing_rater import BillingMode
from gpustack.server.billing_settlement import (
    BillingSettler,
    credit_wallet,
)
from gpustack.server.gateway_auth_reconciler import build_local_auth_tables

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
SECRET_KEY = "31c7e2c1d0a447a1b0f0e6d5c4b3a291"
ORG = 990101
USER_IN_ORG = 990201
USER_OWN_KEY = 990202
OTHER_ORG = 990102


def _principal(id_, kind=PrincipalType.ORG, name="acme"):
    return Principal(
        id=id_,
        kind=kind,
        name=name,
        display_name=name,
        source="local",
        is_admin=False,
        is_active=True,
        created_at=NOW,
        updated_at=NOW,
    )


def _key(
    id_,
    access_key,
    user_id=USER_IN_ORG,
    owner_principal_id=ORG,
    suspended=False,
    reason=None,
    digest=None,
):
    return ApiKey(
        id=id_,
        name=f"key{id_}",
        access_key=access_key,
        hashed_secret_key=f"argon2-{id_}",
        # A real digest, not a stand-in string: ``build_local_auth_tables`` skips
        # a digest it cannot parse, so a fake one would make the gateway-exclusion
        # test pass for the wrong reason (nothing published at all).
        secret_key_digest=digest
        or new_secret_key_digest(
            secret_key=SECRET_KEY, is_custom=False, access_key=access_key
        ),
        scope=[PermissionScope.ALL],
        user_id=user_id,
        owner_principal_id=owner_principal_id,
        is_custom=False,
        suspended=suspended,
        suspension_reason=reason,
        created_at=NOW,
        updated_at=NOW,
    )


def _charge(id_, principal_id=ORG, amount="5.00"):
    return LedgerEntry(
        id=id_,
        source_table="model_usage_details",
        source_id=id_,
        principal_id=principal_id,
        sku=SKU_TOKEN_PROMPT,
        quantity=Decimal(1000),
        unit="tokens",
        unit_price=Decimal("0.005"),
        amount=Decimal(str(amount)),
        currency="CNY",
        direction=LedgerDirection.DEBIT,
        settle_mode=SettleMode.REALTIME,
        status=LedgerStatus.PENDING,
        occurred_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


@pytest_asyncio.fixture
async def engine():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        for model in (
            Principal,
            ApiKey,
            Wallet,
            LedgerEntry,
            BillingSession,
            # Resuming a suspension weighs unpaid invoices too (see
            # billing_settlement.outstanding_charges).
            Invoice,
        ):
            await conn.run_sync(model.__table__.create)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    @asynccontextmanager
    async def _factory():
        async with AsyncSession(engine, expire_on_commit=False) as s:
            yield s

    with patch.object(billing_settlement, "async_session", _factory):
        yield _factory


@pytest_asyncio.fixture
async def session(engine):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s


async def _seed(factory, *rows):
    async with factory() as s:
        for row in rows:
            s.add(row)
        await s.commit()


async def _keys(engine):
    async with AsyncSession(engine) as s:
        return {k.access_key: k for k in (await s.exec(select(ApiKey))).all()}


# ---------------------------------------------------------------------------
# Suspension propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suspend_flags_keys_by_org_and_by_principal(engine, session_factory):
    await _seed(
        session_factory,
        _principal(ORG),
        _principal(USER_IN_ORG, PrincipalType.USER, "user-in-org"),
        _principal(USER_OWN_KEY, PrincipalType.USER, "user-own-key"),
        _key(1, "ak-org", user_id=USER_IN_ORG, owner_principal_id=ORG),
        # No Org owner: reachable only through the user_id arm. Without it a
        # legacy key would keep working after its owner ran out of money.
        _key(2, "ak-no-org", user_id=ORG, owner_principal_id=None),
        _key(3, "ak-other", user_id=USER_OWN_KEY, owner_principal_id=OTHER_ORG),
    )

    async with session_factory() as s:
        changed = await suspend_keys_for_wallet(s, ORG)
        await s.commit()

    assert changed == 2
    keys = await _keys(engine)
    assert keys["ak-org"].suspended is True
    assert keys["ak-org"].suspension_reason == REASON_ARREARS
    assert keys["ak-no-org"].suspended is True
    # Another org's key is untouched.
    assert keys["ak-other"].suspended is False


@pytest.mark.asyncio
async def test_suspend_is_idempotent(engine, session_factory):
    await _seed(session_factory, _principal(ORG), _key(1, "ak-org"))

    async with session_factory() as s:
        first = await suspend_keys_for_wallet(s, ORG)
        await s.commit()
    async with session_factory() as s:
        second = await suspend_keys_for_wallet(s, ORG)
        await s.commit()

    assert (first, second) == (1, 0)


@pytest.mark.asyncio
async def test_resume_clears_only_billing_suspensions(engine, session_factory):
    """An admin suspension must survive a top-up."""
    await _seed(
        session_factory,
        _principal(ORG),
        _key(1, "ak-billing", suspended=True, reason=REASON_ARREARS),
        _key(2, "ak-admin", suspended=True, reason="abuse: rate limit violation"),
    )

    async with session_factory() as s:
        cleared = await resume_keys_for_wallet(s, ORG)
        await s.commit()

    assert cleared == 1
    keys = await _keys(engine)
    assert keys["ak-billing"].suspended is False
    assert keys["ak-billing"].suspension_reason is None
    assert keys["ak-admin"].suspended is True
    assert keys["ak-admin"].suspension_reason.startswith("abuse:")


def test_billing_reason_is_namespaced():
    """The prefix is what makes a targeted resume safe; pin it."""
    assert REASON_ARREARS.startswith(BILLING_REASON_PREFIX)


# ---------------------------------------------------------------------------
# Request-path refusal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suspended_key_is_refused_with_402(session):
    key = _key(1, "ak-org", suspended=True, reason=REASON_ARREARS)

    with pytest.raises(PaymentRequiredException) as exc_info:
        await assert_billing_active(session, api_key=key, user=None)
    assert exc_info.value.status_code == 402
    assert REASON_ARREARS in exc_info.value.message


@pytest.mark.asyncio
async def test_a_key_minted_during_arrears_is_born_suspended(engine, session_factory):
    """Creation-time inheritance closes the gap the flag-only check leaves.

    The suspension pass already ran over the keys that existed; without
    inheriting at creation, an org in arrears could keep serving traffic simply
    by issuing itself a fresh credential.
    """
    await _seed(
        session_factory,
        Wallet(
            principal_id=ORG,
            balance=Decimal(0),
            suspended=True,
            created_at=NOW,
            updated_at=NOW,
        ),
    )

    async with session_factory() as s:
        reason = await inherited_suspension(
            s, owner_principal_id=ORG, user_id=USER_IN_ORG
        )
    assert reason == REASON_ARREARS

    # A healthy org's new key is born clean.
    async with session_factory() as s:
        await credit_wallet(s, OTHER_ORG, Decimal("10"))
        await s.commit()
    async with session_factory() as s:
        assert (
            await inherited_suspension(
                s, owner_principal_id=OTHER_ORG, user_id=USER_OWN_KEY
            )
            is None
        )


@pytest.mark.asyncio
async def test_inherited_suspension_fails_open_when_billing_is_unreachable(caplog):
    """Key creation must not 500 because the wallet could not be read.

    The gap is self-healing — while arrears remain, every settlement sweep
    re-propagates the suspension to all of the principal's keys — so refusing to
    create the key would trade a covered window for an outage.
    """
    with caplog.at_level("WARNING", logger="gpustack.server.billing_enforcement"):
        reason = await inherited_suspension(
            object(), owner_principal_id=ORG, user_id=USER_IN_ORG
        )

    assert reason is None
    assert "could not check wallet suspension" in caplog.text


@pytest.mark.asyncio
async def test_a_clean_key_passes_without_touching_the_database():
    """The request path must not issue a query to decide this.

    A stub session that would raise on any use is the assertion: the check reads
    a column off the credential it was handed. Putting a wallet read here would
    add a round trip to the authorization path of every inference call — and
    would break callers (and tests) that have no billing tables at all.
    """

    class _NoQueries:
        def __getattr__(self, name):
            raise AssertionError(
                f"assert_billing_active must not use the session ({name})"
            )

    await assert_billing_active(
        _NoQueries(), api_key=_key(1, "ak-org"), user=None, model_name="Qwen3-8B"
    )


@pytest.mark.asyncio
async def test_a_credential_with_no_suspension_field_passes(session):
    """Callers that predate the column (or stub it) are not refused."""
    legacy_key = SimpleNamespace(access_key="ak-legacy", owner_principal_id=ORG)
    await assert_billing_active(session, api_key=legacy_key, user=None)


@pytest.mark.asyncio
async def test_healthy_caller_passes(engine, session_factory):
    await _seed(
        session_factory,
        Wallet(
            principal_id=ORG,
            balance=Decimal(10),
            suspended=False,
            created_at=NOW,
            updated_at=NOW,
        ),
    )

    async with session_factory() as s:
        await assert_billing_active(s, api_key=_key(1, "ak-org"), user=None)


@pytest.mark.asyncio
async def test_no_wallet_means_no_suspension(session):
    """A deployment that has not enabled billing must not start refusing."""
    await assert_billing_active(session, api_key=_key(1, "ak-org"), user=None)


@pytest.mark.asyncio
async def test_openai_shaped_error_for_the_proxy_path(session):
    """Both paths refuse with 402 and the same reason.

    The envelope cannot be asserted by type here: ``http_exception_factory``
    implements ``is_openai_exception`` by reassigning the *class*'s
    ``__bases__``, so the first openai-shaped raise in a process permanently
    makes every ``PaymentRequiredException`` an ``OpenAIAPIException``. That is
    pre-existing behaviour of the factory, not of this module, so the assertion
    is on what both paths actually guarantee — status and message.
    """
    key = _key(1, "ak-org", suspended=True, reason=REASON_ARREARS)

    with pytest.raises(PaymentRequiredException) as shaped:
        await assert_billing_active(session, api_key=key, user=None, openai_shaped=True)
    with pytest.raises(PaymentRequiredException) as plain:
        await assert_billing_active(session, api_key=key, user=None)

    assert shaped.value.status_code == plain.value.status_code == 402
    assert REASON_ARREARS in shaped.value.message
    assert REASON_ARREARS in plain.value.message


@pytest.mark.asyncio
async def test_model_name_appears_in_the_refusal(session):
    key = _key(1, "ak-org", suspended=True, reason=REASON_ARREARS)

    with pytest.raises(PaymentRequiredException) as exc_info:
        await assert_billing_active(
            session, api_key=key, user=None, model_name="Qwen3-8B"
        )
    assert "Qwen3-8B" in exc_info.value.message


# ---------------------------------------------------------------------------
# Gateway local auth table
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_table_excludes_suspended_keys(engine, session_factory):
    """Exclusion is the mechanism: a locally verified key never reaches 402."""
    await _seed(
        session_factory,
        _principal(ORG),
        _principal(USER_IN_ORG, PrincipalType.USER, "user-in-org"),
        _key(1, "ak-live", user_id=USER_IN_ORG),
        _key(
            2,
            "ak-suspended",
            user_id=USER_IN_ORG,
            suspended=True,
            reason=REASON_ARREARS,
        ),
    )

    async with session_factory() as s:
        keys, _refs = await build_local_auth_tables(s)

    assert "ak-live" in keys
    assert "ak-suspended" not in keys


@pytest.mark.asyncio
async def test_gateway_table_readmits_a_resumed_key(engine, session_factory):
    await _seed(
        session_factory,
        _principal(ORG),
        _principal(USER_IN_ORG, PrincipalType.USER, "user-in-org"),
        _key(1, "ak-org", user_id=USER_IN_ORG),
    )
    async with session_factory() as s:
        await suspend_keys_for_wallet(s, ORG)
        await s.commit()
    async with session_factory() as s:
        keys, _ = await build_local_auth_tables(s)
    assert keys == {}

    async with session_factory() as s:
        await resume_keys_for_wallet(s, ORG)
        await s.commit()
    async with session_factory() as s:
        keys, _ = await build_local_auth_tables(s)
    assert "ak-org" in keys


# ---------------------------------------------------------------------------
# End to end: settle -> suspend -> top up -> resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settling_into_arrears_suspends_the_keys(engine, session_factory):
    await _seed(
        session_factory,
        _principal(ORG),
        _principal(USER_IN_ORG, PrincipalType.USER, "user-in-org"),
        _key(1, "ak-org", user_id=USER_IN_ORG),
        Wallet(
            principal_id=ORG,
            balance=Decimal("1.00"),
            created_at=NOW,
            updated_at=NOW,
        ),
        _charge(1, amount="5.00"),
    )

    report = await BillingSettler(mode=BillingMode.ENFORCE).settle_once()

    assert report.suspended == [ORG]
    assert report.entries_settled == 0
    keys = await _keys(engine)
    assert keys["ak-org"].suspended is True

    # And the request path now refuses it.
    async with session_factory() as s:
        with pytest.raises(PaymentRequiredException):
            await assert_billing_active(s, api_key=keys["ak-org"], user=None)


@pytest.mark.asyncio
async def test_top_up_resumes_keys_and_collects_the_arrears(engine, session_factory):
    await _seed(
        session_factory,
        _principal(ORG),
        _principal(USER_IN_ORG, PrincipalType.USER, "user-in-org"),
        _key(1, "ak-org", user_id=USER_IN_ORG),
        Wallet(
            principal_id=ORG,
            balance=Decimal("1.00"),
            suspended=True,
            created_at=NOW,
            updated_at=NOW,
        ),
        _charge(1, amount="5.00"),
    )
    async with session_factory() as s:
        await suspend_keys_for_wallet(s, ORG)
        await s.commit()

    async with session_factory() as s:
        await credit_wallet(s, ORG, Decimal("50"))
        await s.commit()

    report = await BillingSettler(mode=BillingMode.ENFORCE).settle_once()

    assert report.entries_settled == 1
    assert report.resumed == [ORG]
    keys = await _keys(engine)
    assert keys["ak-org"].suspended is False
    assert keys["ak-org"].suspension_reason is None
    wallet = (await _wallets(engine))[0]
    assert wallet.suspended is False
    assert wallet.balance == Decimal("46.00")


async def _wallets(engine):
    async with AsyncSession(engine) as s:
        return list((await s.exec(select(Wallet))).all())
