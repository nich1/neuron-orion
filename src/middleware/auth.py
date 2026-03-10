"""Auth dependency — validates JWTs and API keys against neuron-cerberus."""
from __future__ import annotations

import hmac
from dataclasses import dataclass

import httpx
import structlog
from fastapi import HTTPException, Request

from ..settings import settings

log = structlog.get_logger(__name__)

_API_KEY_PREFIX = "nk_"


@dataclass
class AuthenticatedUser:
    user_id: int
    username: str = ""
    auth_method: str = "jwt"


async def _get_http_client(request: Request) -> httpx.AsyncClient:
    """Pull the shared httpx client from app state, falling back to a short-lived one."""
    client: httpx.AsyncClient | None = getattr(request.app.state, "http_client", None)
    if client is not None:
        return client
    return httpx.AsyncClient(timeout=10.0)


async def require_auth(request: Request) -> AuthenticatedUser:
    """FastAPI dependency that enforces authentication via neuron-cerberus.

    Supports two schemes:
      - Bearer JWT  → POST {AUTH_URL}/auth/validate
      - API key     → POST {AUTH_URL}/keys/validate  (key starts with 'nk_')
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        await log.awarning("auth.missing_header", path=request.url.path)
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        await log.awarning("auth.empty_token", path=request.url.path)
        raise HTTPException(status_code=401, detail="Empty Authorization token")

    if settings.INTERNAL_API_KEY and hmac.compare_digest(token, settings.INTERNAL_API_KEY):
        structlog.contextvars.bind_contextvars(user_id=0, username="internal-service")
        await log.adebug("auth.internal_service", path=request.url.path)
        return AuthenticatedUser(user_id=0, username="internal-service", auth_method="internal")

    client = await _get_http_client(request)

    try:
        if token.startswith(_API_KEY_PREFIX):
            user = await _validate_api_key(client, token)
        else:
            user = await _validate_jwt(client, token)
    except HTTPException:
        raise
    except httpx.ConnectError:
        await log.aerror("auth.server_unreachable", auth_url=settings.AUTH_URL)
        raise HTTPException(status_code=503, detail="Auth service unavailable")
    except Exception:
        await log.aerror("auth.unexpected_error", exc_info=True)
        raise HTTPException(status_code=503, detail="Auth service error")

    structlog.contextvars.bind_contextvars(
        user_id=user.user_id, username=user.username,
    )
    await log.adebug(
        "auth.success",
        user_id=user.user_id,
        username=user.username,
        method=user.auth_method,
        path=request.url.path,
    )
    return user


async def _validate_jwt(client: httpx.AsyncClient, token: str) -> AuthenticatedUser:
    resp = await client.post(
        f"{settings.AUTH_URL}/auth/validate",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
    )
    if resp.status_code != 200:
        await log.awarning("auth.jwt_rejected", status=resp.status_code)
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    body = resp.json()
    if not body.get("valid"):
        await log.awarning("auth.jwt_invalid", body=body)
        raise HTTPException(status_code=401, detail="Token validation failed")

    return AuthenticatedUser(
        user_id=body.get("user_id", 0),
        username=body.get("username", ""),
        auth_method="jwt",
    )


async def _validate_api_key(client: httpx.AsyncClient, key: str) -> AuthenticatedUser:
    resp = await client.post(
        f"{settings.AUTH_URL}/keys/validate",
        json={"key": key},
        timeout=10.0,
    )
    if resp.status_code != 200:
        await log.awarning("auth.api_key_rejected", status=resp.status_code)
        raise HTTPException(status_code=401, detail="Invalid API key")

    body = resp.json()
    if not body.get("valid"):
        await log.awarning("auth.api_key_invalid", body=body)
        raise HTTPException(status_code=401, detail="API key validation failed")

    return AuthenticatedUser(
        user_id=body.get("user_id", 0),
        username="",
        auth_method="api_key",
    )
