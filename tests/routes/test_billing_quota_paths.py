"""Both request paths must ask the quota gate — and ask it in the right order.

The gate's arithmetic is tested in ``tests/server/test_billing_quota.py``; what
could silently break here is the wiring. A ceiling that is enforced on one path
and not the other is worse than no ceiling at all, because it looks configured:
traffic through the gateway would be limited while traffic through the
in-process proxy is not, or vice versa, and the difference would only show up as
one tenant's usage inexplicably exceeding its allowance.

So these assert, for each path, that the gate is called with the subject ids the
caller's credential actually implies, that it is called *before* any routing or
upstream work, and that a refusal propagates out as 429 rather than being
swallowed. Suspension is asserted to run first, because 402 and 429 are
different remediations and an empty wallet is the more fundamental answer.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.datastructures import Headers

from gpustack.api.exceptions import (
    PaymentRequiredException,
    TooManyRequestsException,
)
from gpustack.api.auth import (
    GATEWAY_ASSERTED_ACCESS_KEY_HEADER,
    GATEWAY_AUTH_TOKEN_HEADER,
)
from gpustack.routes import openai as openai_route
from gpustack.routes import token as token_route
from gpustack.routes.token import server_auth
from gpustack.schemas.api_keys import PermissionScope
from gpustack.schemas.model_routes import AccessPolicyEnum
from gpustack.schemas.principals import PrincipalType
from gpustack.security import JWTManager
from tests.utils.mock import mock_async_session

GATEWAY_TOKEN = "derived-gateway-token"
ACCESS_KEY = "3192253c1f4a9b7e"
MODEL = "my-org/qwen3-8b"
KEY_ID = 58
USER_ID = 7
ORG_ID = 990101


# ---------------------------------------------------------------------------
# Gateway path — /token-auth
# ---------------------------------------------------------------------------


def _gateway_request():
    request = SimpleNamespace()
    request.state = SimpleNamespace()
    # Case-insensitive, as the real thing is: the security dependencies read
    # "Authorization" while the gateway sends lower-cased headers.
    request.headers = Headers(
        {
            GATEWAY_AUTH_TOKEN_HEADER: GATEWAY_TOKEN,
            GATEWAY_ASSERTED_ACCESS_KEY_HEADER: ACCESS_KEY,
            "x-higress-llm-model": MODEL,
        }
    )
    request.cookies = {}
    request.app = SimpleNamespace(
        state=SimpleNamespace(
            jwt_manager=JWTManager(secret_key="jwt-secret"),
            server_config=SimpleNamespace(
                gateway_mode=None,
                get_derived_gateway_token=lambda: GATEWAY_TOKEN,
            ),
        )
    )
    return request


def _asserted_key():
    return SimpleNamespace(
        id=KEY_ID,
        access_key=ACCESS_KEY,
        user_id=USER_ID,
        owner_principal_id=ORG_ID,
        expires_at=None,
        deleted_at=None,
        suspended=False,
        suspension_reason=None,
        scope=[PermissionScope.ALL],
        is_custom=False,
        secret_key_digest=None,
    )


def _stub_gateway_path(monkeypatch, api_key, user):
    """Form B: the gateway verified the credential and asserts the identity."""

    class _ModelRouteService:
        def __init__(self, session):
            pass

        async def get_model_auth_info_by_name(self, name):
            return AccessPolicyEnum.AUTHED, "registration-token"

    class _UserService:
        def __init__(self, session):
            pass

        async def model_allowed_for_user(self, model_name, user_id, api_key):
            return True

    class _APIKeyService:
        def __init__(self, session):
            pass

        async def get_by_access_key(self, access_key):
            return api_key

    class _AuthUserService:
        def __init__(self, session):
            pass

        async def get_by_id(self, user_id):
            return user

    async def _one_by_id(session, key_id):
        return api_key

    async def _authenticate_request(*args, **kwargs):
        raise AssertionError("form B must not re-authenticate the credential")

    monkeypatch.setattr(token_route, "ModelRouteService", _ModelRouteService)
    monkeypatch.setattr(token_route, "UserService", _UserService)
    monkeypatch.setattr(token_route, "authenticate_request", _authenticate_request)
    monkeypatch.setattr("gpustack.api.auth.APIKeyService", _APIKeyService)
    monkeypatch.setattr("gpustack.api.auth.UserService", _AuthUserService)
    monkeypatch.setattr("gpustack.api.auth.ApiKey.one_by_id", _one_by_id)


def _record_gate(monkeypatch, module):
    """Replace the module's gate with a recorder; returns the call list."""
    calls = []

    async def _gate(session, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(module, "check_quota", _gate)
    return calls


@pytest.mark.asyncio
async def test_the_gateway_path_asks_the_gate_about_the_credential(monkeypatch):
    api_key = _asserted_key()
    _stub_gateway_path(monkeypatch, api_key, SimpleNamespace(
        id=USER_ID, is_active=True, kind=PrincipalType.USER))
    calls = _record_gate(monkeypatch, token_route)

    response = await server_auth(_gateway_request(), session=object())

    assert response.status_code == 200
    assert calls == [
        {
            "api_key_id": KEY_ID,
            "user_id": USER_ID,
            # The owning Org pays, so that is the subject an ORG-scoped ceiling
            # has to be matched against.
            "principal_id": ORG_ID,
            "model_name": MODEL,
        }
    ]


@pytest.mark.asyncio
async def test_a_refusal_on_the_gateway_path_reaches_the_client(monkeypatch):
    api_key = _asserted_key()
    _stub_gateway_path(monkeypatch, api_key, SimpleNamespace(
        id=USER_ID, is_active=True, kind=PrincipalType.USER))

    async def _refuse(session, **kwargs):
        raise TooManyRequestsException(message="daily_tokens quota exhausted")

    monkeypatch.setattr(token_route, "check_quota", _refuse)

    with pytest.raises(TooManyRequestsException) as exc_info:
        await server_auth(_gateway_request(), session=object())
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_suspension_is_answered_before_the_quota_is(monkeypatch):
    """402 outranks 429: an empty wallet is the more fundamental refusal.

    Telling a tenant whose money ran out that they hit a daily cap sends them to
    wait for midnight instead of to top up.
    """
    api_key = _asserted_key()
    api_key.suspended = True
    api_key.suspension_reason = "billing:wallet balance exhausted"
    _stub_gateway_path(monkeypatch, api_key, SimpleNamespace(
        id=USER_ID, is_active=True, kind=PrincipalType.USER))
    calls = _record_gate(monkeypatch, token_route)

    with pytest.raises(PaymentRequiredException) as exc_info:
        await server_auth(_gateway_request(), session=object())

    assert exc_info.value.status_code == 402
    assert calls == []


# ---------------------------------------------------------------------------
# In-process path — proxy_request_by_model
# ---------------------------------------------------------------------------


def _proxy_request(api_key):
    request = MagicMock()
    request.url.path = "/v1-openai/chat/completions"
    request.state.api_key = api_key
    return request


def _stub_proxy_path(monkeypatch):
    """Wire the proxy up to the point of route resolution, past the gate."""
    monkeypatch.setattr(openai_route, "async_session", lambda: mock_async_session())
    monkeypatch.setattr(
        openai_route,
        "parse_request_body",
        AsyncMock(return_value=(MODEL, False, {"model": MODEL}, None)),
    )
    monkeypatch.setattr(
        openai_route.UserService, "model_allowed_for_user", AsyncMock(return_value=True)
    )
    resolve = AsyncMock(return_value=[])
    monkeypatch.setattr(
        openai_route.ModelRouteService, "resolve_route_targets", resolve
    )
    monkeypatch.setattr(
        openai_route.ModelRouteService, "get_by_name", AsyncMock(return_value=None)
    )
    return resolve


@pytest.mark.asyncio
async def test_the_in_process_path_asks_the_gate_before_routing(monkeypatch):
    _stub_proxy_path(monkeypatch)
    calls = _record_gate(monkeypatch, openai_route)

    from gpustack.api.exceptions import NotFoundException

    with pytest.raises(NotFoundException):
        # Resolution is stubbed empty, so the call ends at the 404 that follows
        # the gate — which is the point: the gate ran before any routing.
        await openai_route.proxy_request_by_model(
            request=_proxy_request(_asserted_key()),
            user=SimpleNamespace(id=USER_ID),
        )

    assert calls == [
        {
            "api_key_id": KEY_ID,
            "user_id": USER_ID,
            "principal_id": ORG_ID,
            "model_name": MODEL,
            "openai_shaped": True,
        }
    ]


@pytest.mark.asyncio
async def test_a_refusal_on_the_in_process_path_never_reaches_upstream(monkeypatch):
    resolve = _stub_proxy_path(monkeypatch)

    async def _refuse(session, **kwargs):
        raise TooManyRequestsException(
            message="daily_tokens quota exhausted", is_openai_exception=True
        )

    monkeypatch.setattr(openai_route, "check_quota", _refuse)

    with pytest.raises(TooManyRequestsException) as exc_info:
        await openai_route.proxy_request_by_model(
            request=_proxy_request(_asserted_key()),
            user=SimpleNamespace(id=USER_ID),
        )

    assert exc_info.value.status_code == 429
    resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_both_paths_refuse_with_the_same_status(monkeypatch):
    """The two paths must not drift apart on the one thing a client can see."""
    _stub_gateway_path(
        monkeypatch,
        _asserted_key(),
        SimpleNamespace(id=USER_ID, is_active=True, kind=PrincipalType.USER),
    )
    _stub_proxy_path(monkeypatch)

    async def _refuse(session, **kwargs):
        raise TooManyRequestsException(message="daily_tokens quota exhausted")

    monkeypatch.setattr(token_route, "check_quota", _refuse)
    monkeypatch.setattr(openai_route, "check_quota", _refuse)

    with pytest.raises(TooManyRequestsException) as gateway:
        await server_auth(_gateway_request(), session=object())
    with pytest.raises(TooManyRequestsException) as in_process:
        await openai_route.proxy_request_by_model(
            request=_proxy_request(_asserted_key()),
            user=SimpleNamespace(id=USER_ID),
        )

    assert gateway.value.status_code == in_process.value.status_code == 429
