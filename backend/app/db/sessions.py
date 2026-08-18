import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from app.config import SESSIONS_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verifications (
    thread_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (thread_id, message_index)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def _connect() -> aiosqlite.Connection:
    Path(SESSIONS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(SESSIONS_DB_PATH)
    connection.row_factory = aiosqlite.Row
    await connection.executescript(SCHEMA)
    return connection


async def create_thread() -> dict:
    thread = {"thread_id": str(uuid.uuid4()), "title": "New chat",
              "created_at": _now(), "updated_at": _now()}
    connection = await _connect()
    try:
        await connection.execute(
            "INSERT INTO threads VALUES (:thread_id, :title, :created_at, :updated_at)", thread)
        await connection.commit()
    finally:
        await connection.close()
    return thread


async def list_threads() -> list[dict]:
    connection = await _connect()
    try:
        cursor = await connection.execute("SELECT * FROM threads ORDER BY updated_at DESC")
        return [dict(row) for row in await cursor.fetchall()]
    finally:
        await connection.close()


async def record_message(thread_id: str, first_user_message: str) -> None:
    title = first_user_message[:60] + ("..." if len(first_user_message) > 60 else "")
    connection = await _connect()
    try:
        await connection.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?) ON CONFLICT(thread_id) DO UPDATE SET "
            "updated_at = excluded.updated_at, "
            "title = CASE WHEN threads.title = 'New chat' THEN excluded.title ELSE threads.title END",
            (thread_id, title, _now(), _now()))
        await connection.commit()
    finally:
        await connection.close()


async def record_verification(thread_id: str, message_index: int, payload: dict) -> None:
    connection = await _connect()
    try:
        await connection.execute(
            "INSERT OR REPLACE INTO verifications VALUES (?, ?, ?)",
            (thread_id, message_index, json.dumps(payload)))
        await connection.commit()
    finally:
        await connection.close()


async def get_verifications(thread_id: str) -> dict[int, dict]:
    connection = await _connect()
    try:
        cursor = await connection.execute(
            "SELECT message_index, payload FROM verifications WHERE thread_id = ?", (thread_id,))
        return {row["message_index"]: json.loads(row["payload"]) for row in await cursor.fetchall()}
    finally:
        await connection.close()
