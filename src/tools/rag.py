"""RAG tools: embedding, Qdrant search, and ingestion."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from ..settings import settings

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from qdrant_client import AsyncQdrantClient

    from ..deps import AgentDeps

log = logging.getLogger(__name__)


async def embed_text(http_client: httpx.AsyncClient, text: str) -> list[float]:
    resp = await http_client.post(
        f"{settings.OLLAMA_URL}/api/embed",
        json={"model": settings.EMBEDDING_MODEL, "input": text},
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


async def query_qdrant_impl(
    ctx: RunContext[AgentDeps],
    query: str,
    collection: str = "",
) -> list[str]:
    """Search the vector database for relevant context chunks."""
    collection = collection or ctx.deps.memory_namespace
    if not collection:
        return []
    try:
        embedding = await embed_text(ctx.deps.http_client, query)
        results = await ctx.deps.qdrant_client.search(
            collection_name=collection,
            query_vector=embedding,
            limit=5,
        )
        return [r.payload["text"] for r in results if r.payload and "text" in r.payload]
    except Exception:
        log.exception("Qdrant query failed (collection=%s)", collection)
        return []


async def ingest_to_qdrant_impl(
    ctx: RunContext[AgentDeps],
    text: str,
    collection: str = "",
) -> str:
    """Embed text and store it in the vector database for future retrieval."""
    from qdrant_client.models import Distance, PointStruct, VectorParams

    collection = collection or ctx.deps.memory_namespace
    if not collection:
        return "No collection specified"

    try:
        collections = await ctx.deps.qdrant_client.get_collections()
        existing = [c.name for c in collections.collections]
        if collection not in existing:
            await ctx.deps.qdrant_client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

        embedding = await embed_text(ctx.deps.http_client, text)
        import uuid

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": text},
        )
        await ctx.deps.qdrant_client.upsert(collection_name=collection, points=[point])
        return f"Stored in '{collection}'"
    except Exception:
        log.exception("Qdrant ingest failed (collection=%s)", collection)
        return "Ingest failed"
