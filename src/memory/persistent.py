from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

_SCHEMA = (Path(__file__).parent / "schema.sql").read_text()


async def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_SCHEMA)
        await db.commit()


async def load_namespace(db_path: str, namespace: str) -> dict[str, Any]:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT key, value FROM memory WHERE namespace = ?",
            (namespace,),
        )
        rows = await cursor.fetchall()
    return {k: json.loads(v) for k, v in rows}


async def read_memory(db_path: str, namespace: str, key: str) -> Any | None:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT value FROM memory WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        row = await cursor.fetchone()
    return json.loads(row[0]) if row else None


async def write_memory(db_path: str, namespace: str, key: str, value: Any) -> None:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO memory (namespace, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (namespace, key, json.dumps(value), now),
        )
        await db.commit()


async def delete_memory(db_path: str, namespace: str, key: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "DELETE FROM memory WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        await db.commit()


async def list_namespaces(db_path: str) -> list[str]:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT DISTINCT namespace FROM memory")
        rows = await cursor.fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# HITL flag operations
# ---------------------------------------------------------------------------

async def create_flag(
    db_path: str,
    agent_name: str,
    category: str,
    summary: str,
    context: dict[str, Any],
    proposed_action: str,
    severity: str = "review",
) -> str:
    flag_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO hitl_flags
               (id, agent_name, category, severity, summary, context, proposed_action, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (flag_id, agent_name, category, severity, summary, json.dumps(context), proposed_action, now),
        )
        await db.commit()
    return flag_id


async def list_flags(db_path: str, status: str | None = None) -> list[dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        if status:
            cursor = await db.execute(
                "SELECT * FROM hitl_flags WHERE status = ? ORDER BY created_at DESC", (status,)
            )
        else:
            cursor = await db.execute("SELECT * FROM hitl_flags ORDER BY created_at DESC")
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_flag(db_path: str, flag_id: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM hitl_flags WHERE id = ?", (flag_id,))
        row = await cursor.fetchone()
    return dict(row) if row else None


async def resolve_flag(
    db_path: str, flag_id: str, status: str, response: str | None = None
) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        result = await db.execute(
            """UPDATE hitl_flags SET status = ?, user_response = ?, resolved_at = ?
               WHERE id = ? AND status = 'pending'""",
            (status, response, now, flag_id),
        )
        await db.commit()
    return result.rowcount > 0
