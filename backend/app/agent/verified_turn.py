from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.prompts import CORRECTION_TEMPLATE
from app.agent.summarization import split_at_window
from app.llm import chat_model
from app.verification.judge import stream_faithfulness
from app.verification.models import FaithfulnessResult

CAVEAT = ("Heads up: my own faithfulness check could not tie every figure below back to a tool "
          "result, and I ran out of retries. Treat the flagged numbers with caution.\n\n")


def tool_calls_in_window(messages: list) -> list[dict]:
    _, recent = split_at_window(messages)
    calls = {}
    for message in recent:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call["id"]] = {"tool_name": call["name"], "args": call["args"], "output": None}
    for message in recent:
        if isinstance(message, ToolMessage) and message.tool_call_id in calls:
            calls[message.tool_call_id]["output"] = message.content
    return list(calls.values())


def _problem_lines(problems: list) -> str:
    return "\n".join(f"- \"{p.text}\" ({p.verdict}): {p.reason or 'not supported by any tool result'}"
                     for p in problems)


async def _stream_turn(graph, config, message) -> AsyncIterator[dict]:
    async for mode, payload in graph.astream({"messages": [message]}, config,
                                             stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk = payload[0]
            if isinstance(chunk, AIMessage) and isinstance(chunk.content, str) and chunk.content:
                yield {"type": "token", "text": chunk.content}
        elif mode == "updates":
            for node_state in payload.values():
                for updated in (node_state or {}).get("messages", []):
                    for call in getattr(updated, "tool_calls", None) or []:
                        yield {"type": "tool_call", "name": call["name"], "args": call["args"]}
                    if isinstance(updated, ToolMessage):
                        yield {"type": "tool_result", "name": updated.name,
                               "output": updated.content}


async def run_verified_turn(graph, thread_id: str, user_message: str, max_retries: int = 1,
                            judge_model=None) -> AsyncIterator[dict]:
    config = {"configurable": {"thread_id": thread_id}}
    judge_model = judge_model or chat_model()
    message = HumanMessage(user_message)
    retries_left = max_retries

    while True:
        async for event in _stream_turn(graph, config, message):
            yield event

        state = await graph.aget_state(config)
        messages = state.values["messages"]
        draft = messages[-1].content
        yield {"type": "status", "phase": "verifying"}
        claims = []
        try:
            async for claim in stream_faithfulness(draft, tool_calls_in_window(messages), judge_model):
                claims.append(claim)
                yield {
                    "type": "claim_check",
                    "claim": {
                        "text": claim.text,
                        "verdict": claim.verdict,
                        "reason": claim.reason,
                    },
                }
            result = FaithfulnessResult(claims=claims)
        except (RuntimeError, ValueError) as exc:
            result = FaithfulnessResult(parse_error=str(exc))

        if result.is_faithful:
            yield {"type": "verification", "passed": True, "problems": [],
                   "parse_error": result.parse_error}
            yield {"type": "done", "answer": draft}
            return

        problems = [p.model_dump() for p in result.problems]
        if retries_left <= 0:
            yield {"type": "verification", "passed": False, "problems": problems,
                   "parse_error": result.parse_error}
            yield {"type": "done", "answer": CAVEAT + draft}
            return

        yield {"type": "verification", "passed": False, "problems": problems,
               "parse_error": result.parse_error, "retrying": True}
        yield {"type": "retry", "reason": "faithfulness check failed, rewriting the answer"}
        message = HumanMessage(
            CORRECTION_TEMPLATE.format(problems=_problem_lines(result.problems)),
            additional_kwargs={"correction": True},
        )
        retries_left -= 1
