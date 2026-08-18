from fastapi import APIRouter
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import get_graph
from app.db import sessions

router = APIRouter()


def _serialize(message, index: int, verifications: dict) -> dict:
    if isinstance(message, HumanMessage):
        role = "correction" if message.additional_kwargs.get("correction") else "user"
        return {"role": role, "content": message.content}
    if isinstance(message, ToolMessage):
        return {"role": "tool", "name": message.name, "content": message.content}
    if isinstance(message, AIMessage):
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [{"name": c["name"], "args": c["args"]} for c in message.tool_calls],
            "verification": verifications.get(index),
        }
    return {"role": message.type, "content": str(message.content)}


@router.get("/threads")
async def list_threads() -> list[dict]:
    return await sessions.list_threads()


@router.get("/threads/{thread_id}/messages")
async def get_messages(thread_id: str) -> dict:
    graph = await get_graph()
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", []) if state.values else []
    verifications = await sessions.get_verifications(thread_id)
    return {
        "thread_id": thread_id,
        "messages": [_serialize(m, i, verifications) for i, m in enumerate(messages)],
    }
