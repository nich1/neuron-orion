"""Agent registration — discovers and registers all agents with the registry."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..registry import AgentRegistry

log = logging.getLogger(__name__)


def register_all(registry: AgentRegistry) -> None:
    from . import jarvis, market, memory_consolidation, morning_news, research

    _registry_ref = lambda: registry  # noqa: E731 — closure for deferred access

    # Research — no inter-agent deps
    registry.register(research.config, research.tools)

    # Morning News — needs inline call_research tool
    mn_tools = list(morning_news.tools) + [morning_news.make_call_research_tool(_registry_ref)]
    registry.register(morning_news.config, mn_tools)

    # Market — needs inline call_research tool for watch mode
    mkt_tools = list(market.tools) + [market.make_call_research_tool(_registry_ref)]
    registry.register(market.config, mkt_tools)

    # Memory Consolidation — reads across namespaces, no agent calls
    registry.register(memory_consolidation.config, memory_consolidation.tools)

    # Jarvis — gets individual delegation tools for each specialist
    jarvis_tools = list(jarvis.tools) + jarvis.make_delegation_tools(_registry_ref)
    registry.register(jarvis.config, jarvis_tools)

    log.info("All agents registered")
