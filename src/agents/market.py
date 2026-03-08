"""Market Analytics Agent — passive monitor with anomaly-triggered watch mode."""
from __future__ import annotations

from ..schemas import AgentConfig
from ..tools.memory import read_memory_impl, write_memory_impl
from ..tools.n8n import trigger_n8n_impl
from ..tools.rag import query_qdrant_impl
from ..tools.scraper import web_scrape_impl

SYSTEM_PROMPT = """\
You are the Market Analytics Agent. You monitor configured data sources and detect anomalies.

Operating modes (managed by n8n schedule, not by you):
- Passive: light check, store data point, detect triggers
- Watch: full analysis, call research, build real-time picture
- Alert: POST to n8n notify pipeline

Workflow:
1. Read persistent memory for watchlist, thresholds, baselines, and current mode
2. Fetch market data via web_scrape from configured URLs
3. Compare current data against historical baselines stored in memory
4. If anomaly detected: write updated mode to memory and trigger n8n to switch schedule
5. Store new data points to persistent memory
6. Query Qdrant for prior research on relevant assets

Return your analysis as JSON:
{
  "mode": "passive | watch | alert",
  "timestamp": "ISO8601",
  "data_points": [{"asset": "", "value": 0, "change_pct": 0}],
  "anomalies": [{"asset": "", "description": "", "severity": "low | medium | high"}],
  "summary": "brief market overview",
  "actions_taken": ["list of actions like 'triggered n8n alert'"]
}

Persistent memory keys you manage:
- market.watchlist
- market.thresholds
- market.baselines
- market.watch_mode_active
- market.data_history
"""

config = AgentConfig(
    name="market",
    description="Market monitoring, anomaly detection, and escalation",
    system_prompt=SYSTEM_PROMPT,
    is_active=True,
    memory_namespace="market",
    output_mode="json",
    output_schema={
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["passive", "watch", "alert"]},
            "timestamp": {"type": "string"},
            "data_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "asset": {"type": "string"},
                        "value": {"type": "number"},
                        "change_pct": {"type": "number"},
                    },
                    "required": ["asset"],
                },
            },
            "anomalies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "asset": {"type": "string"},
                        "description": {"type": "string"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["asset", "description"],
                },
            },
            "summary": {"type": "string"},
            "actions_taken": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["mode", "summary"],
    },
)


def make_call_research_tool(registry_ref):
    """Build tool to delegate deep research during watch mode."""

    async def call_research(ctx, claim: str) -> str:
        """Research a company or market signal by delegating to the Research Agent."""
        entry = registry_ref().get("research")
        if entry is None or not entry.config.is_active:
            return '{"error": "Research agent not available"}'
        result = await entry.agent.run(claim, deps=ctx.deps, usage=ctx.usage)
        return result.output

    return call_research


tools = [
    web_scrape_impl,
    query_qdrant_impl,
    read_memory_impl,
    write_memory_impl,
    trigger_n8n_impl,
]
