from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import CHECKPOINT_DB_PATH


async def create_checkpointer() -> AsyncSqliteSaver:
    Path(CHECKPOINT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(CHECKPOINT_DB_PATH)
    saver = AsyncSqliteSaver(connection)
    await saver.setup()
    return saver
