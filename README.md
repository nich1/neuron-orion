# Nich Agent

Pydantic AI agent platform for the Neuron home server. Orchestrates specialist agents (Research, Morning News, Market Analytics, Memory Consolidation) through a central Jarvis interface.

## Stack

- FastAPI + Uvicorn
- Pydantic AI with Ollama (OpenAI-compatible provider)
- Qdrant for RAG vector storage
- SQLite (aiosqlite) for persistent memory and HITL flags
- structlog for structured logging to Seq

## Agents

| Agent | Description | Output |
|-------|-------------|--------|
| Research | Fact-checking and source validation | JSON (claim, verdict, sources) |
| Morning News | Daily briefing generation | JSON (sections, headlines) |
| Market Analytics | Market monitoring and alerts | JSON (overview, watchlist, alerts) |
| Memory Consolidation | Nightly cross-agent memory curation | JSON (consolidation report) |
| Jarvis | Orchestrator and user interface | Text |

## Endpoints

- `POST /agents/{name}/run` -- trigger an agent run
- `GET /agents` -- list registered agents
- `POST /agents/{name}/activate` / `deactivate` -- toggle agents
- `GET /agents/{name}/runs/{run_id}` -- check run status
- `GET /hitl/flags` -- list HITL flags
- `POST /hitl/flags/{id}/resolve` -- resolve a flag
- `GET /health` -- platform health check

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `AUTH_URL` | `http://localhost:8081` | Auth server endpoint |
| `N8N_URL` | `http://localhost:5678` | n8n endpoint |
| `SEQ_URL` | `http://localhost:5341` | Seq logging endpoint |
| `SEQ_API_KEY` | (empty) | Seq API key |
| `DB_PATH` | `data/memory.db` | SQLite database path |
| `DEFAULT_MODEL` | `qwen2.5:7b` | Default LLM model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |

## Docker

```bash
docker build -t niches1/neuron-orion:latest .
docker push niches1/neuron-orion:latest
```

Pull the image:

```bash
docker pull niches1/neuron-orion:latest
```

## Development

```bash
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
