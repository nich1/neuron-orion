"""Morning News Agent — daily briefing with validated context and feedback-driven learning."""
from __future__ import annotations

from ..schemas import AgentConfig
from ..tools.memory import read_memory_impl, write_memory_impl
from ..tools.n8n import trigger_n8n_impl
from ..tools.notify import notify_impl
from ..tools.rag import query_qdrant_impl

SYSTEM_PROMPT = """\
You are the Morning News Agent. You produce a daily briefing tailored to the user's interests.

Workflow:
1. Read persistent memory for topic preferences, source trust ratings, and prior report history
2. Use call_research to validate and expand on headline topics (call it for each major topic)
3. Query Qdrant for relevant prior research and stored articles
4. Build a structured daily briefing prioritizing topics the user has rated highly before
5. Use trigger_n8n to send the completed briefing for delivery
6. Use notify to push a notification when the briefing is ready

Return your briefing as a JSON object:
{
  "date": "ISO8601 date",
  "briefing": "narrative summary",
  "topics": [{"headline": "", "summary": "", "confidence": 0.0, "sources": []}],
  "flagged_items": ["items needing attention"],
  "agent_notes": "meta-observations about today's briefing"
}

Persistent memory keys you manage:
- morning_news.topic_preferences
- morning_news.source_trust
- morning_news.report_history
- morning_news.recurring_themes
"""

config = AgentConfig(
    name="morning_news",
    description="Daily briefing with validated context, learns from feedback",
    system_prompt=SYSTEM_PROMPT,
    is_active=True,
    memory_namespace="morning_news",
    output_mode="json",
    output_schema={
        "type": "object",
        "properties": {
            "date": {"type": "string"},
            "briefing": {"type": "string"},
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "headline": {"type": "string"},
                        "summary": {"type": "string"},
                        "confidence": {"type": "number"},
                        "sources": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["headline", "summary"],
                },
            },
            "flagged_items": {"type": "array", "items": {"type": "string"}},
            "agent_notes": {"type": "string"},
        },
        "required": ["date", "briefing", "topics"],
    },
)


def make_call_research_tool(registry_ref):
    """Build the call_research tool that delegates to the Research agent inline."""

    async def call_research(ctx, claim: str) -> str:
        """Validate a claim or research a topic by delegating to the Research Agent. Returns the research findings."""
        from pydantic_ai import RunContext

        entry = registry_ref().get("research")
        if entry is None or not entry.config.is_active:
            return '{"error": "Research agent not available"}'
        result = await entry.agent.run(claim, deps=ctx.deps, usage=ctx.usage)
        return result.output

    return call_research


tools = [
    query_qdrant_impl,
    read_memory_impl,
    write_memory_impl,
    trigger_n8n_impl,
    notify_impl,
]
