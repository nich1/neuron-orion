from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
from qdrant_client import AsyncQdrantClient


@dataclass
class AgentDeps:
    http_client: httpx.AsyncClient
    qdrant_client: AsyncQdrantClient
    callback_url: str | None = None
    memory_namespace: str = ""
    persistent_context: dict[str, Any] = field(default_factory=dict)
