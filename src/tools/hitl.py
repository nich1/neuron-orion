"""Human-in-the-Loop flagging tools exposed to agents via RunContext."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ..memory.persistent import create_flag, get_flag
from ..settings import settings

if TYPE_CHECKING:
    from pydantic_ai import RunContext

    from ..deps import AgentDeps

log = logging.getLogger(__name__)


async def raise_flag_impl(
    ctx: RunContext[AgentDeps],
    category: str,
    summary: str,
    context: str,
    proposed_action: str,
    severity: str = "review",
) -> str:
    """Flag a decision for human review. Non-blocking: the agent should proceed with
    its best judgment after raising the flag.

    Args:
        category: Type of flag (e.g. "prune_conflict", "low_confidence", "policy").
        summary: One-line explanation of what needs review.
        context: JSON string of relevant data the human will need to decide.
        proposed_action: What the agent intends to do (or already did).
        severity: "info" (FYI), "review" (needs decision), or "blocking" (do NOT proceed).
    """
    try:
        ctx_dict = json.loads(context)
    except json.JSONDecodeError:
        ctx_dict = {"raw": context}

    agent_name = ctx.deps.memory_namespace or "unknown"
    flag_id = await create_flag(
        settings.DB_PATH,
        agent_name=agent_name,
        category=category,
        summary=summary,
        context=ctx_dict,
        proposed_action=proposed_action,
        severity=severity,
    )
    log.info("HITL flag raised: %s by %s — %s", flag_id, agent_name, summary)
    return f"Flag {flag_id} raised (severity={severity}). Continue with best judgment unless severity is 'blocking'."


async def check_flag_resolution_impl(
    ctx: RunContext[AgentDeps],
    flag_id: str,
) -> str:
    """Check if a previously raised HITL flag has been resolved by a human."""
    row = await get_flag(settings.DB_PATH, flag_id)
    if row is None:
        return f"No flag found with id '{flag_id}'"
    status = row["status"]
    if status == "pending":
        return f"Flag {flag_id} is still pending human review."
    return json.dumps({
        "flag_id": flag_id,
        "status": status,
        "user_response": row.get("user_response"),
        "resolved_at": row.get("resolved_at"),
    })
