"""Jarvis — natural language interface that orchestrates all other agents."""
from __future__ import annotations

from ..schemas import AgentConfig
from ..tools.agents import call_agent_impl
from ..tools.hitl import check_flag_resolution_impl, raise_flag_impl
from ..tools.memory import read_memory_impl, write_memory_impl
from ..tools.n8n import trigger_n8n_impl
from ..tools.rag import query_qdrant_impl
from ..tools.scraper import web_search_impl

SYSTEM_PROMPT = """\
You are Jarvis, the user's personal AI assistant and the orchestrator of the agent platform.

You can delegate tasks to specialist agents:
- call_research: validate claims, find sources
- call_morning_news: generate a daily briefing
- call_market: check market conditions

For simple lookups, use web_search directly instead of delegating.
For questions about stored knowledge, use query_qdrant.
For personal context, read_memory contains known facts about the user.

Always:
1. Determine which agents or tools are needed to answer the question
2. Delegate to specialists when their expertise is required
3. Synthesize results from multiple sources into a clear answer
4. Store important new facts about the user via write_memory
5. Check for pending HITL flags and surface them to the user proactively
6. Be conversational and direct

Persistent memory keys you manage:
- jarvis.user_profile — known facts, preferences, working style
- jarvis.interaction_history — summarized past interactions
- jarvis.inferred_trends — patterns from Memory Consolidation
- jarvis.proactive_queue — insights to surface at next interaction
"""

config = AgentConfig(
    name="jarvis",
    description="Natural language life assistant and agent orchestrator",
    system_prompt=SYSTEM_PROMPT,
    is_active=True,
    memory_namespace="jarvis",
    output_mode="text",
    max_concurrency=4,
)


def make_delegation_tools(registry_ref):
    """Build inline delegation tools for each specialist agent."""

    async def call_research(ctx, topic: str) -> str:
        """Delegate a research or fact-checking task to the Research Agent. Returns findings."""
        entry = registry_ref().get("research")
        if entry is None or not entry.config.is_active:
            return "Research agent not available"
        result = await entry.agent.run(topic, deps=ctx.deps, usage=ctx.usage)
        return result.output

    async def call_morning_news(ctx, request: str) -> str:
        """Ask the Morning News Agent to generate a briefing. Returns the briefing."""
        entry = registry_ref().get("morning_news")
        if entry is None or not entry.config.is_active:
            return "Morning News agent not available"
        result = await entry.agent.run(request, deps=ctx.deps, usage=ctx.usage)
        return result.output

    async def call_market(ctx, query: str) -> str:
        """Ask the Market Analytics Agent for a market check. Returns the analysis."""
        entry = registry_ref().get("market")
        if entry is None or not entry.config.is_active:
            return "Market agent not available"
        result = await entry.agent.run(query, deps=ctx.deps, usage=ctx.usage)
        return result.output

    return [call_research, call_morning_news, call_market]


tools = [
    web_search_impl,
    query_qdrant_impl,
    read_memory_impl,
    write_memory_impl,
    trigger_n8n_impl,
    call_agent_impl,
    raise_flag_impl,
    check_flag_resolution_impl,
]
