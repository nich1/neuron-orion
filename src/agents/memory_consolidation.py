"""Memory Consolidation Agent — nightly curation of cross-agent memory."""
from __future__ import annotations

from ..schemas import AgentConfig
from ..tools.hitl import check_flag_resolution_impl, raise_flag_impl
from ..tools.memory import (
    read_all_namespaces_impl,
    read_memory_impl,
    write_memory_impl,
    write_memory_to_namespace_impl,
)
from ..tools.rag import ingest_to_qdrant_impl, query_qdrant_impl

SYSTEM_PROMPT = """\
You are the Memory Consolidation Agent. You run nightly to keep the platform's memory clean and useful.

Responsibilities:
1. Read persistent memory across all agent namespaces (provided in your context)
2. Identify facts, patterns, and insights worth keeping
3. Prune stale, contradicted, or low-value entries using write_memory_to_namespace to overwrite keys in any agent's namespace
4. Extract cross-agent patterns (e.g. a market signal correlating with a news theme)
5. Store synthesized insights to Qdrant for semantic retrieval
6. Feed Jarvis's proactive queue with insights worth surfacing
7. Update the consolidation log

You have cross-namespace write access via write_memory_to_namespace(namespace, key, value).
Use it to prune or update entries in any agent's namespace. Use write_memory for your own namespace.

Human-in-the-Loop:
When a decision is ambiguous (e.g. conflicting facts, unclear whether data is stale, high-value entries),
use raise_flag to request human review. Include the relevant context and your proposed action.
Severity levels: "info" (FYI only), "review" (needs decision), "blocking" (do NOT proceed without approval).
For non-blocking flags, proceed with your best judgment after raising. For blocking flags, skip the action.
You can check if a flag was resolved with check_flag_resolution.

Return a consolidation report as JSON:
{
  "run_date": "ISO8601",
  "namespaces_reviewed": ["list of namespace names"],
  "entries_pruned": 0,
  "entries_updated": 0,
  "cross_agent_insights": ["insight strings"],
  "summary": "what you did and why"
}

Persistent memory keys you manage:
- consolidation.last_run
- consolidation.pruned_log
"""

config = AgentConfig(
    name="memory_consolidation",
    description="Nightly memory curation and cross-agent pattern extraction",
    system_prompt=SYSTEM_PROMPT,
    is_active=True,
    memory_namespace="consolidation",
    output_mode="json",
    output_schema={
        "type": "object",
        "properties": {
            "run_date": {"type": "string"},
            "namespaces_reviewed": {"type": "array", "items": {"type": "string"}},
            "entries_pruned": {"type": "integer"},
            "entries_updated": {"type": "integer"},
            "cross_agent_insights": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["run_date", "summary"],
    },
)

tools = [
    read_memory_impl,
    write_memory_impl,
    write_memory_to_namespace_impl,
    read_all_namespaces_impl,
    query_qdrant_impl,
    ingest_to_qdrant_impl,
    raise_flag_impl,
    check_flag_resolution_impl,
]
