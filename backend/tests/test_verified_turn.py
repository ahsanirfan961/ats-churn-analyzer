import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.verified_turn import CAVEAT, run_verified_turn, tool_calls_in_window

TOOL_CALL = {"name": "segment_stats", "args": {"filters": {"InternetService": "DSL"}}, "id": "t1"}


def turn_messages(question: str, answer: str):
    return [
        HumanMessage(question),
        AIMessage("", tool_calls=[TOOL_CALL]),
        ToolMessage('{"churn_rate": 0.1896}', tool_call_id="t1", name="segment_stats"),
        AIMessage(answer),
    ]


class ClosableConnection:
    async def close(self):
        pass


class FakeGraph:
    checkpointer = type("Saver", (), {"conn": ClosableConnection()})()

    def __init__(self, answers):
        self.answers = list(answers)
        self.messages = []
        self.sent = []

    async def astream(self, payload, config, stream_mode=None):
        self.sent.append(payload["messages"][0])
        answer = self.answers.pop(0)
        self.messages += turn_messages(payload["messages"][0].content, answer)
        yield "updates", {"agent": {"messages": [AIMessage("", tool_calls=[TOOL_CALL])]}}
        yield "updates", {"tools": {"messages": [self.messages[-2]]}}
        yield "messages", (AIMessage(answer), {})

    async def aget_state(self, config):
        return type("State", (), {"values": {"messages": self.messages}})()


class ScriptedJudge:
    def __init__(self, *payloads):
        self.payloads = list(payloads)

    async def ainvoke(self, prompt):
        return type("Response", (), {"content": json.dumps(self.payloads.pop(0))})()


GROUNDED = {"claims": [{"text": "DSL churns at 19%", "verdict": "grounded",
                        "source_tool_call_index": 0}]}
MISLABELED = {"claims": [{"text": "risk score of 19%", "verdict": "mislabeled",
                          "source_tool_call_index": 0,
                          "reason": "19% is the DSL churn rate, not a risk score"}]}


async def collect(graph, judge, max_retries=1):
    return [e async for e in run_verified_turn(graph, "thread-1", "How does DSL churn?",
                                               max_retries=max_retries, judge_model=judge)]


def test_tool_calls_in_window_pairs_calls_with_their_outputs():
    calls = tool_calls_in_window(turn_messages("q", "a"))
    assert len(calls) == 1
    assert calls[0]["tool_name"] == "segment_stats"
    assert "0.1896" in calls[0]["output"]


@pytest.mark.asyncio
async def test_faithful_draft_passes_through_untouched():
    graph = FakeGraph(["DSL churns at 19%."])
    events = await collect(graph, ScriptedJudge(GROUNDED))

    assert [e["type"] for e in events if e["type"] in {"tool_call", "tool_result"}] == [
        "tool_call", "tool_result"]
    verification = next(e for e in events if e["type"] == "verification")
    assert verification["passed"] is True
    assert events[-1] == {"type": "done", "answer": "DSL churns at 19%."}
    assert len(graph.sent) == 1


@pytest.mark.asyncio
async def test_unfaithful_draft_triggers_one_retry_with_the_flagged_claim():
    graph = FakeGraph(["Their risk score is 19%.", "DSL churns at 19%."])
    events = await collect(graph, ScriptedJudge(MISLABELED, GROUNDED))

    assert any(e["type"] == "retry" for e in events)
    correction = graph.sent[1]
    assert correction.additional_kwargs["correction"] is True
    assert "risk score of 19%" in correction.content
    assert "not a risk score" in correction.content
    assert events[-1]["answer"] == "DSL churns at 19%."


@pytest.mark.asyncio
async def test_exhausted_retries_ship_the_answer_with_a_caveat():
    graph = FakeGraph(["Their risk score is 19%.", "Still their risk score is 19%."])
    events = await collect(graph, ScriptedJudge(MISLABELED, MISLABELED))

    assert sum(e["type"] == "retry" for e in events) == 1
    final_verification = [e for e in events if e["type"] == "verification"][-1]
    assert final_verification["passed"] is False
    assert final_verification["problems"][0]["verdict"] == "mislabeled"
    assert events[-1]["answer"].startswith(CAVEAT)


@pytest.mark.asyncio
async def test_zero_retries_never_reinvokes_the_graph():
    graph = FakeGraph(["Their risk score is 19%."])
    events = await collect(graph, ScriptedJudge(MISLABELED), max_retries=0)

    assert len(graph.sent) == 1
    assert events[-1]["answer"].startswith(CAVEAT)
