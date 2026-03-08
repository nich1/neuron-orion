"""Cross-agent delegation tools."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..settings import settings

if TYPE_CHECKING:
    from pydantic_ai import RunContext

    from ..deps import AgentDeps

log = logging.getLogger(__name__)


async def call_agent_impl(
    ctx: RunContext[AgentDeps],
    agent_name: str,
    prompt: str,
) -> str:
    """Delegate a task to another agent via HTTP and return the run_id. Results arrive via callback."""
    url = f"http://localhost:{settings.PORT}/agents/{agent_name}/run"
    try:
        resp = await ctx.deps.http_client.post(
            url,
            json={"prompt": prompt, "callback_url": ctx.deps.callback_url},
            timeout=30.0,
        )
        resp.raise_for_status()
        return f"Delegated to {agent_name}: {resp.json()}"
    except Exception:
        log.exception("Agent delegation to '%s' failed", agent_name)
        return f"Failed to call agent '{agent_name}'"
