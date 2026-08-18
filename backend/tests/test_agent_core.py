import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.summarization import keep_recent_turns, split_at_window


def turn(question: str, with_tool: bool = False):
    messages = [HumanMessage(question)]
    if with_tool:
        messages.append(AIMessage("", tool_calls=[{"name": "segment_stats", "args": {}, "id": "t1"}]))
        messages.append(ToolMessage("{}", tool_call_id="t1"))
    messages.append(AIMessage(f"answer to {question}"))
    return messages


def conversation(n: int):
    return [m for i in range(n) for m in turn(f"q{i}", with_tool=True)]


def test_short_conversation_is_kept_whole():
    messages = conversation(2)
    older, recent = split_at_window(messages, window=3)
    assert older == []
    assert recent == messages


def test_window_keeps_exactly_the_last_n_turns():
    messages = conversation(5)
    older, recent = split_at_window(messages, window=3)
    assert [m.content for m in recent if isinstance(m, HumanMessage)] == ["q2", "q3", "q4"]
    assert [m.content for m in older if isinstance(m, HumanMessage)] == ["q0", "q1"]


def test_window_keeps_tool_messages_of_kept_turns():
    older, recent = split_at_window(conversation(5), window=3)
    assert sum(isinstance(m, ToolMessage) for m in recent) == 3


def test_correction_message_does_not_open_a_new_turn():
    messages = conversation(3)
    messages.append(HumanMessage("fix your claim", additional_kwargs={"correction": True}))
    messages.append(AIMessage("corrected"))
    older, recent = split_at_window(messages, window=3)
    assert older == []
    assert recent[-1].content == "corrected"


@pytest.mark.asyncio
async def test_hook_passes_short_history_through_untouched():
    messages = conversation(2)
    result = await keep_recent_turns({"messages": messages})
    assert result["llm_input_messages"] == messages
