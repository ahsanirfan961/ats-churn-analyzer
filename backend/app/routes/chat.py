import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.graph import get_graph
from app.agent.verified_turn import run_verified_turn
from app.db import sessions

router = APIRouter()


class MessageIn(BaseModel):
    message: str


def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"


async def _turn_events(thread_id: str, message: str):
    await sessions.record_message(thread_id, message)
    try:
        graph = await get_graph()
        verification = None
        async for event in run_verified_turn(graph, thread_id, message):
            if event["type"] == "verification":
                verification = event
            yield _sse(event)

        if verification is not None:
            state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
            await sessions.record_verification(
                thread_id, len(state.values["messages"]) - 1, verification)
    except Exception as exc:
        yield _sse({"type": "error",
                    "message": f"The agent could not complete this turn ({type(exc).__name__})."})


@router.post("/threads")
async def create_thread() -> dict:
    return await sessions.create_thread()


@router.post("/threads/{thread_id}/messages")
async def post_message(thread_id: str, body: MessageIn) -> StreamingResponse:
    return StreamingResponse(_turn_events(thread_id, body.message),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
