"""Billing enforcement: from an unpaid wallet to a refused request (WP5).

The settler decides that an org owes money it does not have; this module is what
makes that decision reach the request path. Two hops, and both have to happen:

1. **Flag the keys.** ``api_keys.suspended`` is the signal the gateway
   understands — a suspended key is filtered out of ``build_local_auth_tables``,
   so the plugin cannot verify it locally and forwards the request to
   ``/token-auth``. Excluding rather than flagging needs no plugin change:
   absence from the table is the only thing the plugin reads.
2. **Refuse on the server.** ``/token-auth`` and the in-process proxy path both
   answer 402 for a suspended credential, so the two paths refuse the same key
   for the same reason. A gateway that forwarded to a server which then let the
   request through would be worse than either alone.

Flipping the column is not enough on its own: ``APIKeyService.get_by_access_key``
is cached process-globally for ``SERVER_CACHE_TTL_SECONDS``, so without an
invalidation the fallback path keeps authenticating the pre-suspension row for
the rest of the TTL — the same trap ``ApiKey.delete()`` documents for revocation.
Every write here drops the affected keys' cache entries.

The key flag is authoritative, and that is what keeps the request path cheap
---------------------------------------------------------------------------
``assert_billing_active`` reads a column off the credential it was handed and
issues no query. That is only sound because the flag cannot go stale in the
direction that matters:

* suspending a wallet flags every key that spends from it, in the same
  transaction (``billing_settlement.suspend_wallet``);
* a key created *after* a suspension inherits it
  (:func:`inherited_suspension`, called from key creation);
* resuming clears the billing flags again.

The alternative — re-reading the wallet on every request — would put a database
round trip on the authorization path of every inference call to cover a window
that creation-time inheritance already closes.

Suspension reasons are prefixed (``billing:``) and only that prefix is ever
cleared by this module, so an admin suspension imposed for another reason
survives a top-up.
"""

import logging
from typing import List, Optional

from sqlalchemy import or_, update
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import PaymentRequiredException
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.billing import Wallet

logger = logging.getLogger(__name__)

# Namespace for reasons this module sets, and the only ones it clears.
BILLING_REASON_PREFIX = "billing:"
REASON_ARREARS = f"{BILLING_REASON_PREFIX}wallet balance exhausted"


async def _affected_keys(session: AsyncSession, principal_id: int) -> List[ApiKey]:
    """The keys a wallet's state speaks for.

    An org's wallet covers the keys it owns (``owner_principal_id``) and, for
    deployments that predate org ownership or where a user *is* the billing
    subject, the keys belonging to that principal directly (``user_id``). Both
    arms are matched because a key with no owner would otherwise be unreachable
    by any suspension — the failure mode being a tenant who keeps consuming
    after their money ran out.
    """
    rows = (
        await session.exec(
            select(ApiKey).where(
                ApiKey.deleted_at.is_(None),
                or_(
                    ApiKey.owner_principal_id == principal_id,
                    ApiKey.user_id == principal_id,
                ),
            )
        )
    ).all()
    return list(rows)


async def _drop_auth_cache(keys: List[ApiKey]) -> None:
    """Make a suspension take effect now rather than after the auth-cache TTL."""
    # Imported here because the service layer imports the schemas module, and a
    # module-level import would close that loop.
    from gpustack.server.cache import build_cache_key, delete_cache_by_key
    from gpustack.server.services import APIKeyService

    for key in keys:
        if key.access_key:
            await delete_cache_by_key(
                _key=build_cache_key(APIKeyService.get_by_access_key, key.access_key)
            )


async def suspend_keys_for_wallet(
    session: AsyncSession,
    principal_id: int,
    *,
    reason: str = REASON_ARREARS,
    commit: bool = False,
) -> int:
    """Suspend every key the wallet covers. Returns how many changed."""
    keys = [k for k in await _affected_keys(session, principal_id) if not k.suspended]
    if not keys:
        return 0

    await session.exec(
        update(ApiKey)
        .where(ApiKey.id.in_([k.id for k in keys]))
        .values(suspended=True, suspension_reason=reason)
    )
    await session.flush()
    if commit:
        await session.commit()
    await _drop_auth_cache(keys)
    logger.warning(
        f"billing: suspended {len(keys)} api key(s) for principal {principal_id} "
        f"({reason})"
    )
    return len(keys)


async def resume_keys_for_wallet(
    session: AsyncSession, principal_id: int, *, commit: bool = False
) -> int:
    """Clear billing suspensions once the wallet can cover its charges.

    Only rows this module suspended are touched: the ``billing:`` prefix is the
    marker, so an admin suspension for another reason stays in place and the key
    stays refused — resuming it would undo a decision billing has no standing to
    reverse.
    """
    keys = [
        k
        for k in await _affected_keys(session, principal_id)
        if k.suspended and (k.suspension_reason or "").startswith(BILLING_REASON_PREFIX)
    ]
    if not keys:
        return 0

    await session.exec(
        update(ApiKey)
        .where(ApiKey.id.in_([k.id for k in keys]))
        .values(suspended=False, suspension_reason=None)
    )
    await session.flush()
    if commit:
        await session.commit()
    await _drop_auth_cache(keys)
    logger.info(f"billing: resumed {len(keys)} api key(s) for principal {principal_id}")
    return len(keys)


async def wallet_suspension_reason(
    session: AsyncSession, principal_id: Optional[int]
) -> Optional[str]:
    """Why this principal's wallet refuses service, or None if it does not.

    Checked on the request path as well as trusting the key flag: a key created
    after a suspension, or one whose flag has not propagated yet, must not slip
    through in the window between the wallet running dry and the settler's next
    pass.
    """
    if principal_id is None:
        return None
    wallet = (
        await session.exec(
            select(Wallet).where(
                Wallet.principal_id == principal_id, Wallet.deleted_at.is_(None)
            )
        )
    ).first()
    if wallet is None or not wallet.suspended:
        return None
    return REASON_ARREARS


def _principal_for(api_key: Optional[ApiKey], user) -> Optional[int]:
    """Whose wallet pays for this request.

    The key's owning Org when it has one — that is the settlement subject the
    wallet was opened for — otherwise the caller's own principal.
    """
    if api_key is not None and getattr(api_key, "owner_principal_id", None):
        return api_key.owner_principal_id
    if api_key is not None and getattr(api_key, "user_id", None):
        return api_key.user_id
    return getattr(user, "id", None)


async def inherited_suspension(
    session: AsyncSession,
    *,
    owner_principal_id: Optional[int],
    user_id: Optional[int],
) -> Optional[str]:
    """The suspension a brand-new key is born under, or None.

    Called at key creation. Without it a key minted while its org is in arrears
    would be unflagged — the suspension pass already ran, and nothing would flag
    this one until the wallet was suspended *again* — so the org that ran out of
    money could keep serving traffic through the new key. Checking the owning Org
    first and the user principal second mirrors :func:`_affected_keys`: either
    can be the settlement subject.

    A failure to consult the wallet returns None rather than raising — a
    deliberate fail-open. Creating a key is not a billing operation and must not
    500 because the billing tables are unreachable (an un-migrated database, a
    caller holding no real session), and the gap self-heals: while arrears remain
    the settler retries every sweep and re-propagates the suspension to *every*
    key of the principal, including ones created since. It is logged, because
    "billing could not be consulted" is not a fact to swallow quietly.
    """
    for principal_id in (owner_principal_id, user_id):
        try:
            reason = await wallet_suspension_reason(session, principal_id)
        except Exception as e:
            logger.warning(
                f"billing: could not check wallet suspension for principal "
                f"{principal_id} while creating an api key ({e}); creating it "
                "unsuspended — the next settlement sweep re-propagates any "
                "active suspension"
            )
            return None
        if reason is not None:
            return reason
    return None


def _is_suspended(api_key) -> bool:
    """Whether a credential carries a real suspension flag.

    Only an actual boolean counts. ``api_keys.suspended`` is NOT NULL, so an
    ORM-loaded key always carries True or False — but this check sits on the
    authorization path of every inference call, where a false positive is an
    outage rather than an inconvenience. Anything that is merely truthy (a test
    stub, a ``MagicMock`` attribute, a duck-typed credential object) is
    therefore read as "not suspended".
    """
    value = getattr(api_key, "suspended", False)
    return value is True


async def assert_billing_active(
    session: AsyncSession,
    *,
    api_key: Optional[ApiKey],
    user=None,
    model_name: Optional[str] = None,
    openai_shaped: bool = False,
) -> None:
    """Raise 402 when the caller's credential is suspended for arrears.

    Called from both request paths (``/token-auth`` for the gateway,
    ``proxy_request_by_model`` for the in-process proxy) so a suspended org is
    refused identically whichever way its traffic arrives. 402 rather than 403:
    the credential is valid and the caller is authorized, the account owes money
    — a tenant that sees 403 rotates keys, one that sees 402 tops up.

    ``session`` and ``user`` are accepted and unused so the two call sites read
    alike and so a future check that does need them (a quota gate, say) does not
    have to change either signature. The flag itself is authoritative — see the
    module docstring for why no wallet read is needed here.

    ``openai_shaped`` renders the error in the OpenAI envelope, which is what the
    proxy path returns for everything else it raises; the gateway path keeps the
    platform's own shape because the plugin translates it.
    """
    if api_key is not None and _is_suspended(api_key):
        raise PaymentRequiredException(
            message=(
                getattr(api_key, "suspension_reason", None)
                or "API key suspended for billing reasons"
            )
            + (f" (model {model_name})" if model_name else ""),
            is_openai_exception=openai_shaped,
        )
