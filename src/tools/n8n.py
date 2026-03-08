"""n8n integration tools: trigger pipelines and post callbacks."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..settings import settings

if TYPE_CHECKING:
    from pydantic_ai import RunContext

    from ..deps import AgentDeps

log = logging.getLogger(__name__)


async def trigger_n8n_impl(
    ctx: RunContext[AgentDeps],
    webhook_path: str,
    payload: str = "{}",
) -> str:
    """Trigger an n8n webhook pipeline with the given payload (JSON string)."""
    import json

    url = f"{settings.N8N_URL}/webhook/{webhook_path}"
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        data = {"message": payload}

    try:
        resp = await ctx.deps.http_client.post(url, json=data, timeout=30.0)
        return f"n8n responded {resp.status_code}"
    except Exception:
        log.exception("n8n trigger failed: %s", url)
        return "n8n trigger failed"


async def post_callback(
    ctx: RunContext[AgentDeps],
    result: str,
) -> str:
    """Post results to the callback URL if one was provided for this run."""
    if not ctx.deps.callback_url:
        return "No callback URL configured"
    try:
        resp = await ctx.deps.http_client.post(
            ctx.deps.callback_url,
            json={"result": result},
            timeout=30.0,
        )
        return f"Callback responded {resp.status_code}"
    except Exception:
        log.exception("Callback to %s failed", ctx.deps.callback_url)
        return "Callback failed"
