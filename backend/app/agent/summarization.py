from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.config import RECENT_TURNS_WINDOW
from app.llm import chat_model

SUMMARY_INSTRUCTION = """Summarise this earlier part of a churn-analysis conversation in a short
prose paragraph. Record what the user asked about and what was concluded. Do not restate specific
figures as facts - say which quantities were looked up, not what they were, because these numbers
will no longer be verifiable."""

_summary_cache: dict[str, str] = {}


def _is_turn_start(message: BaseMessage) -> bool:
    return isinstance(message, HumanMessage) and not message.additional_kwargs.get("correction")


def split_at_window(messages: list[BaseMessage],
                    window: int = RECENT_TURNS_WINDOW) -> tuple[list, list]:
    turn_starts = [i for i, m in enumerate(messages) if _is_turn_start(m)]
    if len(turn_starts) <= window:
        return [], list(messages)
    cutoff = turn_starts[-window]
    return list(messages[:cutoff]), list(messages[cutoff:])


async def summarize(older: list[BaseMessage]) -> str:
    key = str(older[-1].id)
    if key not in _summary_cache:
        transcript = "\n".join(f"{m.type}: {str(m.content)[:1500]}" for m in older)
        response = await chat_model().ainvoke(f"{SUMMARY_INSTRUCTION}\n\n{transcript}")
        _summary_cache[key] = str(response.content)
    return _summary_cache[key]


async def keep_recent_turns(state: dict) -> dict:
    older, recent = split_at_window(state["messages"])
    if not older:
        return {"llm_input_messages": recent}

    try:
        summary = await summarize(older)
    except Exception:
        summary = "(earlier turns of this conversation could not be summarised)"
    preamble = SystemMessage(
        f"Summary of earlier turns, outside the current verification window. Treat any figures "
        f"mentioned here as unverified and recompute them if needed:\n{summary}")
    return {"llm_input_messages": [preamble, *recent]}
