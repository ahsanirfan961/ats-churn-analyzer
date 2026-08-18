import json

import pytest
from fastapi.testclient import TestClient

from tests.test_verified_turn import GROUNDED, FakeGraph, ScriptedJudge


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import main
    from app.agent import graph as graph_module
    from app.agent import verified_turn
    from app.db import sessions

    monkeypatch.setattr(sessions, "SESSIONS_DB_PATH", str(tmp_path / "sessions.sqlite"))
    monkeypatch.setattr(graph_module, "_graph", FakeGraph(["DSL customers churn at 19%."]))
    monkeypatch.setattr(verified_turn, "chat_model", lambda *a, **k: ScriptedJudge(GROUNDED))
    with TestClient(main.app) as test_client:
        yield test_client


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        events.append((lines[0].removeprefix("event: "),
                       json.loads(lines[1].removeprefix("data: "))))
    return events


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_thread_lifecycle_and_sse_stream(client):
    thread_id = client.post("/threads").json()["thread_id"]

    listed = client.get("/threads").json()
    assert [t["thread_id"] for t in listed] == [thread_id]

    response = client.post(f"/threads/{thread_id}/messages",
                           json={"message": "How does DSL churn?"})
    assert response.status_code == 200
    events = parse_sse(response.text)
    types = [name for name, _ in events]
    assert types[0] == "tool_call"
    assert "tool_result" in types and "verification" in types
    assert types[-1] == "done"
    assert dict(events)["done"]["answer"] == "DSL customers churn at 19%."


def test_thread_title_is_taken_from_the_first_user_message(client):
    thread_id = client.post("/threads").json()["thread_id"]
    client.post(f"/threads/{thread_id}/messages", json={"message": "How does DSL churn?"})
    assert client.get("/threads").json()[0]["title"] == "How does DSL churn?"


def test_history_reload_includes_tools_and_verification(client):
    thread_id = client.post("/threads").json()["thread_id"]
    client.post(f"/threads/{thread_id}/messages", json={"message": "How does DSL churn?"})

    history = client.get(f"/threads/{thread_id}/messages").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"]
    assert history[1]["tool_calls"][0]["name"] == "segment_stats"
    assert history[-1]["verification"]["passed"] is True


def test_agent_failure_becomes_an_error_event_not_a_traceback(client, monkeypatch):
    from app.routes import chat

    async def explode(*args, **kwargs):
        raise RuntimeError("upstream 500")
        yield

    monkeypatch.setattr(chat, "run_verified_turn", explode)
    thread_id = client.post("/threads").json()["thread_id"]
    response = client.post(f"/threads/{thread_id}/messages", json={"message": "hi"})

    assert response.status_code == 200
    name, payload = parse_sse(response.text)[-1]
    assert name == "error"
    assert "upstream 500" not in payload["message"]
