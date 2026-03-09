from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from jsonschema import ValidationError, validate
from pydantic_ai import Agent

from .deps import AgentDeps
from .memory.persistent import load_namespace
from .middleware.auth import AuthenticatedUser, require_auth
from .schemas import AgentConfig, AgentInfo, AgentRunRequest, AgentRunResponse, RunStatus
from .settings import settings

log = logging.getLogger(__name__)


@dataclass
class AgentEntry:
    config: AgentConfig
    agent: Agent[AgentDeps, str]
    semaphore: asyncio.Semaphore
    router: APIRouter


class AgentRegistry:
    def __init__(self, app: FastAPI, http_client: httpx.AsyncClient, qdrant_client: Any) -> None:
        self._agents: dict[str, AgentEntry] = {}
        self._runs: dict[str, RunStatus] = {}
        self._app = app
        self._http_client = http_client
        self._qdrant_client = qdrant_client
        self._mount_management_routes()

    def register(self, config: AgentConfig, tools: list[Callable] | None = None) -> None:
        if config.name in self._agents:
            raise ValueError(f"Agent '{config.name}' is already registered")

        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        model = OpenAIChatModel(
            model_name=config.model,
            provider=OpenAIProvider(base_url=f"{settings.OLLAMA_URL}/v1"),
        )

        agent = Agent(
            model,
            deps_type=AgentDeps,
            output_type=str,
            instructions=config.system_prompt,
            retries=config.max_retries,
        )

        if tools:
            for tool_fn in tools:
                agent.tool(tool_fn)

        semaphore = asyncio.Semaphore(config.max_concurrency)
        router = self._build_router(config.name)

        entry = AgentEntry(config=config, agent=agent, semaphore=semaphore, router=router)
        self._agents[config.name] = entry
        self._app.include_router(router)

        log.info("Registered agent '%s' (model=%s, active=%s)", config.name, config.model, config.is_active)

    def activate(self, name: str) -> None:
        self._get_entry(name).config.is_active = True
        log.info("Activated agent '%s'", name)

    def deactivate(self, name: str) -> None:
        self._get_entry(name).config.is_active = False
        log.info("Deactivated agent '%s'", name)

    def get(self, name: str) -> AgentEntry | None:
        return self._agents.get(name)

    def list_agents(self) -> list[AgentInfo]:
        return [
            AgentInfo(
                name=e.config.name,
                description=e.config.description,
                model=e.config.model,
                is_active=e.config.is_active,
                output_mode=e.config.output_mode,
                endpoint=f"/agents/{e.config.name}/run",
            )
            for e in self._agents.values()
        ]

    def get_run(self, run_id: str) -> RunStatus | None:
        return self._runs.get(run_id)

    async def run_agent(self, name: str, prompt: str, callback_url: str | None = None) -> str:
        entry = self._get_entry(name)
        config = entry.config

        run_id = str(uuid.uuid4())
        status = RunStatus(run_id=run_id, agent_name=name)
        self._runs[run_id] = status

        asyncio.create_task(self._execute_run(entry, run_id, prompt, callback_url))
        return run_id

    async def _execute_run(
        self,
        entry: AgentEntry,
        run_id: str,
        prompt: str,
        callback_url: str | None,
    ) -> None:
        config = entry.config
        status = self._runs[run_id]
        namespace = config.memory_namespace or config.name

        try:
            async with entry.semaphore:
                persistent_context = await load_namespace(settings.DB_PATH, namespace)

                deps = AgentDeps(
                    http_client=self._http_client,
                    qdrant_client=self._qdrant_client,
                    callback_url=callback_url,
                    memory_namespace=namespace,
                    persistent_context=persistent_context,
                )

                result_text = await self._run_with_output_mode(entry, prompt, deps, config)

                status.status = "completed"
                status.result = result_text
                status.completed_at = datetime.now(timezone.utc)

        except Exception:
            log.exception("Agent '%s' run %s failed", config.name, run_id)
            status.status = "failed"
            status.error = "Internal agent error"
            status.completed_at = datetime.now(timezone.utc)
            result_text = None

        if callback_url and result_text is not None:
            try:
                await self._http_client.post(
                    callback_url,
                    json={"run_id": run_id, "agent": config.name, "result": result_text},
                    timeout=30.0,
                )
            except Exception:
                log.exception("Callback to %s failed for run %s", callback_url, run_id)

    async def _run_with_output_mode(
        self,
        entry: AgentEntry,
        prompt: str,
        deps: AgentDeps,
        config: AgentConfig,
    ) -> str:
        model_settings: dict[str, Any] = {}
        if config.output_mode == "json" and config.output_schema:
            model_settings["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": config.name, "schema": config.output_schema},
            }

        current_prompt = prompt
        for attempt in range(config.max_retries + 1):
            result = await entry.agent.run(current_prompt, deps=deps, model_settings=model_settings)

            if config.output_mode == "text":
                return result.output

            try:
                parsed = json.loads(result.output)
                if config.output_schema:
                    validate(instance=parsed, schema=config.output_schema)
                return result.output
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == config.max_retries:
                    return json.dumps({"raw": result.output, "error": str(e), "valid": False})
                current_prompt = f"Your JSON was invalid: {e}. Original request: {prompt}. Try again."

        return result.output  # type: ignore[possibly-undefined]

    def _get_entry(self, name: str) -> AgentEntry:
        entry = self._agents.get(name)
        if entry is None:
            raise ValueError(f"Agent '{name}' is not registered")
        return entry

    def _build_router(self, agent_name: str) -> APIRouter:
        router = APIRouter(prefix=f"/agents/{agent_name}", tags=[agent_name])
        registry = self

        @router.post("/run", response_model=AgentRunResponse)
        async def run_endpoint(
            request: AgentRunRequest,
            _user: AuthenticatedUser = Depends(require_auth),
        ) -> AgentRunResponse:
            entry = registry._get_entry(agent_name)
            if not entry.config.is_active:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' is not active")
            run_id = await registry.run_agent(agent_name, request.prompt, request.callback_url)
            return AgentRunResponse(run_id=run_id, agent_name=agent_name, status="accepted")

        return router

    def _mount_management_routes(self) -> None:
        router = APIRouter(prefix="/agents", tags=["management"])
        registry = self

        @router.get("")
        async def list_agents(
            _user: AuthenticatedUser = Depends(require_auth),
        ) -> list[AgentInfo]:
            return registry.list_agents()

        @router.post("/{name}/activate")
        async def activate_agent(
            name: str,
            _user: AuthenticatedUser = Depends(require_auth),
        ) -> dict[str, str]:
            try:
                registry.activate(name)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            return {"status": "activated", "agent": name}

        @router.post("/{name}/deactivate")
        async def deactivate_agent(
            name: str,
            _user: AuthenticatedUser = Depends(require_auth),
        ) -> dict[str, str]:
            try:
                registry.deactivate(name)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            return {"status": "deactivated", "agent": name}

        @router.get("/{name}/runs/{run_id}")
        async def get_run_status(
            name: str,
            run_id: str,
            _user: AuthenticatedUser = Depends(require_auth),
        ) -> RunStatus:
            status = registry.get_run(run_id)
            if status is None or status.agent_name != name:
                raise HTTPException(status_code=404, detail="Run not found")
            return status

        self._app.include_router(router)
