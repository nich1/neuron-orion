from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    name: str
    description: str
    system_prompt: str
    model: str = "qwen2.5:7b"
    is_active: bool = False
    max_concurrency: int = 2
    memory_namespace: str = ""
    output_mode: Literal["text", "json"] = "text"
    output_schema: dict[str, Any] | None = None
    max_retries: int = 3


class AgentRunRequest(BaseModel):
    prompt: str
    callback_url: str | None = None


class AgentRunResponse(BaseModel):
    run_id: str
    agent_name: str
    status: str


class RunStatus(BaseModel):
    run_id: str
    agent_name: str
    status: Literal["running", "completed", "failed"] = "running"
    result: str | None = None
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class AgentInfo(BaseModel):
    name: str
    description: str
    model: str
    is_active: bool
    output_mode: str
    endpoint: str


class HITLFlag(BaseModel):
    id: str
    agent_name: str
    category: str
    severity: Literal["info", "review", "blocking"] = "review"
    summary: str
    context: dict[str, Any] = {}
    proposed_action: str
    status: Literal["pending", "approved", "rejected", "modified"] = "pending"
    user_response: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None


class HITLResolveRequest(BaseModel):
    status: Literal["approved", "rejected", "modified"]
    response: str | None = None
