"""Nich Neuron — Agent Platform"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from .log import setup_logging
from .memory.persistent import init_db, list_flags, get_flag, resolve_flag
from .registry import AgentRegistry
from .schemas import HITLResolveRequest
from .settings import settings

setup_logging(seq_url=settings.SEQ_URL, seq_api_key=settings.SEQ_API_KEY)
log = logging.getLogger(__name__)

registry: AgentRegistry | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry

    await init_db(settings.DB_PATH)

    http_client = httpx.AsyncClient(timeout=120.0)
    qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL)

    registry = AgentRegistry(app, http_client, qdrant_client)

    from .agents import register_all
    register_all(registry)

    await _ensure_qdrant_collections(qdrant_client, registry)

    log.info("Agent platform ready — %d agent(s) registered", len(registry.list_agents()))
    yield

    await http_client.aclose()
    await qdrant_client.close()


async def _ensure_qdrant_collections(
    qdrant_client: AsyncQdrantClient, registry: AgentRegistry
) -> None:
    existing = {c.name for c in (await qdrant_client.get_collections()).collections}
    namespaces = {
        info.name for info in registry.list_agents() if info.name
    }
    namespaces.add("knowledge")
    for ns in sorted(namespaces):
        if ns not in existing:
            await qdrant_client.create_collection(
                collection_name=ns,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            log.info("Created Qdrant collection '%s'", ns)


app = FastAPI(title="Agent Platform", version="0.1.0", lifespan=lifespan)


@app.get("/")
async def root():
    return {"service": "nich-neuron", "status": "ok", "docs": "/docs"}


@app.get("/health")
async def health():
    agent_count = len(registry.list_agents()) if registry else 0
    pending = await list_flags(settings.DB_PATH, status="pending")
    return {
        "status": "ok",
        "ollama": settings.OLLAMA_URL,
        "qdrant": settings.QDRANT_URL,
        "n8n": settings.N8N_URL,
        "agents_registered": agent_count,
        "hitl_pending": len(pending),
    }


# ---------------------------------------------------------------------------
# HITL endpoints
# ---------------------------------------------------------------------------

@app.get("/hitl/flags")
async def hitl_list_flags(status: str | None = None):
    """List HITL flags, optionally filtered by status (pending, approved, rejected, modified)."""
    return await list_flags(settings.DB_PATH, status=status)


@app.get("/hitl/flags/{flag_id}")
async def hitl_get_flag(flag_id: str):
    """Get a single HITL flag by ID."""
    row = await get_flag(settings.DB_PATH, flag_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Flag not found")
    return row


@app.post("/hitl/flags/{flag_id}/resolve")
async def hitl_resolve_flag(flag_id: str, body: HITLResolveRequest):
    """Resolve a pending HITL flag (approve, reject, or modify)."""
    updated = await resolve_flag(
        settings.DB_PATH, flag_id, status=body.status, response=body.response
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Flag not found or already resolved")
    return {"flag_id": flag_id, "status": body.status}
