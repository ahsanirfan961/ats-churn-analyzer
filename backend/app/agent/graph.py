from langgraph.prebuilt import create_react_agent

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.summarization import keep_recent_turns
from app.db.checkpointer import create_checkpointer
from app.llm import chat_model
from app.tools.registry import get_tools

_graph = None


async def get_graph():
    global _graph
    if _graph is None:
        _graph = create_react_agent(
            chat_model(),
            get_tools(),
            prompt=SYSTEM_PROMPT,
            pre_model_hook=keep_recent_turns,
            checkpointer=await create_checkpointer(),
        )
    return _graph


async def close_graph():
    global _graph
    if _graph is not None:
        await _graph.checkpointer.conn.close()
        _graph = None
