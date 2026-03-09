"""Ntfy push-notification tools exposed to agents via RunContext."""
from __future__ import annotations

import logging
from pydantic_ai import RunContext

from ..deps import AgentDeps
from ..settings import settings

log = logging.getLogger(__name__)

PRIORITY_LEVELS = ("min", "low", "default", "high", "urgent")


async def notify_impl(
    ctx: RunContext[AgentDeps],
    title: str,
    message: str,
    topic: str = "neuron",
    priority: str = "default",
    tags: str = "robot",
) -> str:
    """Send a push notification via the self-hosted Ntfy server.

    Args:
        title: Notification title shown on the device.
        message: Notification body text.
        topic: Ntfy topic channel (default "neuron").
        priority: One of min, low, default, high, urgent.
        tags: Comma-separated emoji shortcodes (e.g. "robot" → 🤖).
    """
    if priority not in PRIORITY_LEVELS:
        priority = "default"

    url = f"{settings.NTFY_URL}/{topic}"
    try:
        resp = await ctx.deps.http_client.post(
            url,
            content=message,
            headers={
                "Title": title,
                "Priority": priority,
                "Tags": tags,
            },
            timeout=10.0,
        )
        log.info("Ntfy publish → %s [%s]", url, resp.status_code)
        return f"Notification sent ({resp.status_code})"
    except Exception:
        log.exception("Ntfy publish failed: %s", url)
        return "Notification delivery failed"
