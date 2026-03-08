"""Research Agent — the platform's source of truth for claim validation and source compilation."""
from __future__ import annotations

from ..schemas import AgentConfig
from ..tools.memory import read_memory_impl, write_memory_impl
from ..tools.rag import ingest_to_qdrant_impl, query_qdrant_impl
from ..tools.scraper import web_scrape_impl, web_search_impl

SYSTEM_PROMPT = """\
You are a Research Agent. Your job is to validate claims, find credible sources, and produce structured verdicts.

When given a topic or claim:
1. Use web_search to find relevant sources
2. Use web_scrape to extract detailed content from promising URLs
3. Use query_qdrant to check if you have prior research on this topic
4. Cross-reference sources to identify agreement and contradiction
5. Store valuable findings with ingest_to_qdrant for future retrieval

Return your findings as a JSON object with this structure:
{
  "claim": "the original claim or topic",
  "verdict": "confirmed | disputed | unverified",
  "confidence": 0.0 to 1.0,
  "sources": [{"url": "", "summary": "", "relevance": 0.0}],
  "contradictions": ["any contradicting claims found"],
  "summary": "brief synthesis of findings"
}
"""

config = AgentConfig(
    name="research",
    description="Fact-check, source compilation, and claim validation",
    system_prompt=SYSTEM_PROMPT,
    is_active=True,
    memory_namespace="research",
    output_mode="json",
    output_schema={
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "verdict": {"type": "string", "enum": ["confirmed", "disputed", "unverified"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "summary": {"type": "string"},
                        "relevance": {"type": "number"},
                    },
                    "required": ["url", "summary"],
                },
            },
            "contradictions": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["claim", "verdict", "confidence", "summary"],
    },
)

tools = [
    web_search_impl,
    web_scrape_impl,
    query_qdrant_impl,
    ingest_to_qdrant_impl,
    read_memory_impl,
    write_memory_impl,
]
