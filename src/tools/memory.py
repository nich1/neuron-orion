"""Persistent memory tools exposed to agents via RunContext."""
from __future__ import annotations

import json
import logging
from pydantic_ai import RunContext

from ..deps import AgentDeps
from ..memory.persistent import list_namespaces, load_namespace, read_memory, write_memory
from ..settings import settings

log = logging.getLogger(__name__)


async def read_memory_impl(
    ctx: RunContext[AgentDeps],
    key: str,
) -> str:
    """Read a value from persistent memory for this agent's namespace."""
    namespace = ctx.deps.memory_namespace
    value = await read_memory(settings.DB_PATH, namespace, key)
    if value is None:
        return f"No memory found for key '{key}'"
    return json.dumps(value)


async def write_memory_impl(
    ctx: RunContext[AgentDeps],
    key: str,
    value: str,
) -> str:
    """Write a value to persistent memory for this agent's namespace. Value should be a JSON string."""
    namespace = ctx.deps.memory_namespace
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    await write_memory(settings.DB_PATH, namespace, key, parsed)
    return f"Saved '{key}' to memory"


async def read_all_namespaces_impl(
    ctx: RunContext[AgentDeps],
) -> str:
    """Read all memory namespaces and their contents. Used by the Memory Consolidation agent."""
    namespaces = await list_namespaces(settings.DB_PATH)
    result: dict[str, dict] = {}
    for ns in namespaces:
        result[ns] = await load_namespace(settings.DB_PATH, ns)
    return json.dumps(result, indent=2)


async def write_memory_to_namespace_impl(
    ctx: RunContext[AgentDeps],
    namespace: str,
    key: str,
    value: str,
) -> str:
    """Write a value to any agent's persistent memory namespace. Only for Memory Consolidation use."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    await write_memory(settings.DB_PATH, namespace, key, parsed)
    return f"Saved '{key}' to namespace '{namespace}'"
