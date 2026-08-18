import asyncio
import sys
import uuid

from langchain_core.messages import HumanMessage

from app.agent.graph import close_graph, get_graph

QUESTIONS = [
    "What is the churn rate for fiber optic customers?",
    "What about DSL instead?",
    "Which contract type has the highest average monthly charges?",
    "How many customers are on paperless billing?",
    "Remind me what the fiber optic churn rate was.",
]


async def ask(graph, thread_id: str, question: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.ainvoke({"messages": [HumanMessage(question)]}, config)
    used = [c["name"] for m in state["messages"] for c in getattr(m, "tool_calls", []) or []]
    print(f"\n> {question}")
    print(f"  tools so far: {used}")
    print(f"  {state['messages'][-1].content}")


async def main():
    thread_id = sys.argv[1] if len(sys.argv) > 1 else str(uuid.uuid4())
    print(f"thread_id: {thread_id}")
    graph = await get_graph()
    try:
        for question in QUESTIONS:
            await ask(graph, thread_id, question)
    finally:
        await close_graph()


if __name__ == "__main__":
    asyncio.run(main())
