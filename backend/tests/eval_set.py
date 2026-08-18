import asyncio
import json
import re
import sys
import uuid
from pathlib import Path

from app.agent.graph import close_graph, get_graph
from app.agent.verified_turn import run_verified_turn

REPORT_PATH = Path(__file__).resolve().parents[2] / "docs" / "eval_report.md"


def rate_forms(rate: float) -> list[str]:
    percent = rate * 100
    return [f"{rate:.4f}", f"{rate:.3f}", f"{percent:.2f}", f"{percent:.1f}", f"{percent:.0f}"]


def number_forms(value: float) -> list[str]:
    return [f"{value:.4f}", f"{value:.2f}", f"{value:.1f}", f"{value:.0f}", f"{int(value):,}"]


QUESTIONS = [
    {
        "id": "eda-numeric",
        "question": "What is the average monthly charge across the customer base?",
        "expected": "mean MonthlyCharges = 64.7617",
        "accept": number_forms(64.7617),
        "expected_tools": ["describe_column"],
    },
    {
        "id": "eda-categorical",
        "question": "How many customers are on each contract type?",
        "expected": "Month-to-month 3875, One year 1473, Two year 1695",
        "accept_all": ["3875", "1473", "1695"],
        "expected_tools": ["describe_column", "segment_stats"],
    },
    {
        "id": "segment-churn",
        "question": "Which contract type has the highest churn rate, and what is it?",
        "expected": "Month-to-month at 0.4271",
        "accept": rate_forms(0.4271),
        "accept_all": ["Month-to-month"],
        "expected_tools": ["segment_stats"],
    },
    {
        "id": "segment-two-filters",
        "question": "What is the churn rate for senior citizens on month-to-month contracts?",
        "expected": "0.5465 over 807 customers",
        "accept": rate_forms(0.5465),
        "expected_tools": ["segment_stats"],
    },
    {
        "id": "correlation-numeric",
        "question": "How strongly are tenure and total charges correlated?",
        "expected": "Pearson r = 0.8262",
        "accept": number_forms(0.8262),
        "expected_tools": ["correlation"],
    },
    {
        "id": "correlation-categorical",
        "question": "Is there a relationship between tenure and whether a customer churns?",
        "expected": "point-biserial r = -0.3522",
        "accept": number_forms(-0.3522) + ["0.35", "0.3522"],
        "expected_tools": ["correlation"],
    },
    {
        "id": "rank-raw-column",
        "question": "Show me the top 3 highest-paying customers who have no tech support.",
        "expected": "7279-BUYWN, 5236-PERKL, 4829-ZLJTK",
        "accept_all": ["7279-BUYWN", "5236-PERKL", "4829-ZLJTK"],
        "expected_tools": ["rank_customers"],
    },
    {
        "id": "single-customer-risk",
        "question": "Why is customer 4424-TKOPW at risk of churning?",
        "expected": "risk 0.8502, driven by tenure and fiber optic internet",
        "accept": rate_forms(0.8502),
        "accept_all": ["tenure"],
        "expected_tools": ["predict_churn_risk"],
    },
    {
        "id": "enrichment-context",
        "question": "For customer 4424-TKOPW, how does their tenure compare to the rest of the base?",
        "expected": "bottom decile, 0-2 months, that decile churns at 0.5835",
        "accept": rate_forms(0.5835) + ["8.9", "decile"],
        "expected_tools": ["predict_churn_risk"],
    },
    {
        "id": "what-if",
        "question": "What would happen to customer 4424-TKOPW's risk on a two-year contract?",
        "expected": "0.8502 -> 0.5938, a drop of 0.2564",
        "accept": rate_forms(0.5938) + rate_forms(0.2564),
        "expected_tools": ["compare_scenarios"],
    },
    {
        "id": "hypothetical",
        "question": "How risky is a brand new month-to-month fiber optic customer with 1 month "
                    "of tenure?",
        "expected": "a high risk score, with the defaulted fields stated",
        "accept_all": ["default"],
        "expected_tools": ["predict_hypothetical"],
    },
    {
        "id": "multi-step",
        "question": "Who are the model's 3 highest-risk customers, and do they skew toward any "
                    "particular payment method?",
        "expected": "5178-LMXOP, 9497-QCMMS, 9300-AGZNL and a payment method breakdown",
        "accept_all": ["5178-LMXOP"],
        "expected_tools": ["rank_customers"],
        "min_tool_calls": 2,
    },
    {
        "id": "followup-in-window",
        "question": "What is the churn rate for fiber optic customers?",
        "expected": "0.4189",
        "accept": rate_forms(0.4189),
        "expected_tools": ["segment_stats"],
    },
    {
        "id": "followup-in-window-2",
        "question": "What about DSL instead?",
        "expected": "0.1896",
        "accept": rate_forms(0.1896),
        "follow_up": True,
        "expected_tools": ["segment_stats"],
    },
    {
        "id": "trap-missing-column",
        "question": "Which region has the highest churn rate?",
        "expected": "an explicit statement that the dataset has no region column",
        "accept_any_phrase": ["does not", "doesn't", "no region", "not available", "not in the"],
        "must_not_contain": ["north", "south", "east", "west"],
    },
    {
        "id": "invalid-input",
        "question": "What is the churn risk for customer 9999-NOPE?",
        "expected": "a clean not-found message, no invented score",
        "accept_any_phrase": ["not found", "no customer", "does not exist", "doesn't exist",
                              "not in the dataset"],
    },
]


def grade(item: dict, answer: str) -> tuple[bool, str]:
    lowered = answer.lower()

    for forbidden in item.get("must_not_contain", []):
        if forbidden in lowered:
            return False, f"answer mentions {forbidden!r}, which does not exist in this dataset"

    for required in item.get("accept_all", []):
        if required.lower() not in lowered:
            return False, f"answer is missing {required!r}"

    if "accept_any_phrase" in item:
        if not any(phrase in lowered for phrase in item["accept_any_phrase"]):
            return False, "answer does not acknowledge the missing field or bad input"

    if "accept" in item:
        normalized = re.sub(r"[,\s]", "", lowered)
        if not any(form.lstrip("-") in normalized for form in item["accept"]):
            return False, f"answer does not contain the expected value ({item['expected']})"

    return True, "ok"


async def run_question(graph, thread_id: str, item: dict) -> dict:
    answer, tools, verification = "", [], None
    async for event in run_verified_turn(graph, thread_id, item["question"]):
        if event["type"] == "tool_call":
            tools.append(event["name"])
        elif event["type"] == "verification":
            verification = event
        elif event["type"] == "done":
            answer = event["answer"]
        elif event["type"] == "error":
            answer = f"[stream error] {event['message']}"

    passed, note = grade(item, answer)
    if passed and item.get("min_tool_calls") and len(tools) < item["min_tool_calls"]:
        passed, note = False, f"expected at least {item['min_tool_calls']} tool calls, got {len(tools)}"

    return {
        "id": item["id"],
        "question": item["question"],
        "expected": item["expected"],
        "answer": answer,
        "tools": tools,
        "passed": passed,
        "note": note,
        "flagged": bool(verification and not verification["passed"]),
        "judge_problems": (verification or {}).get("problems", []),
        "judge_unreadable": bool(verification and verification.get("parse_error")),
    }


def write_report(results: list[dict]) -> None:
    total = len(results)
    passed = sum(r["passed"] for r in results)
    flagged = sum(r["flagged"] for r in results)
    unreadable = sum(r["judge_unreadable"] for r in results)
    calls = sum(len(r["tools"]) for r in results)

    lines = [
        "# Eval report",
        "",
        f"- Questions: {total}",
        f"- Correct answers: {passed}/{total} ({passed / total:.0%})",
        f"- Turns the faithfulness judge flagged: {flagged}/{total} ({flagged / total:.0%})",
        f"- Turns where the judge output could not be parsed: {unreadable}/{total}",
        f"- Tool calls per question: {calls / total:.1f} average",
        "",
        "| id | pass | tools called | judge | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        judge = "flagged" if r["flagged"] else ("unreadable" if r["judge_unreadable"] else "clean")
        lines.append(f"| {r['id']} | {'yes' if r['passed'] else 'NO'} | "
                     f"{', '.join(r['tools']) or '-'} | {judge} | {r['note']} |")

    lines += ["", "## Answers", ""]
    for r in results:
        lines += [
            f"### {r['id']}",
            "",
            f"**Q:** {r['question']}",
            "",
            f"**Expected:** {r['expected']}",
            "",
            f"**Answer:** {r['answer']}",
            "",
        ]
        if r["judge_problems"]:
            lines += ["**Judge flagged:**", "",
                      "```json", json.dumps(r["judge_problems"], indent=2), "```", ""]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    graph = await get_graph()
    results = []
    thread_id = str(uuid.uuid4())
    try:
        for item in QUESTIONS:
            if not item.get("follow_up"):
                thread_id = str(uuid.uuid4())
            result = await run_question(graph, thread_id, item)
            results.append(result)
            print(f"{'PASS' if result['passed'] else 'FAIL'}  {result['id']}: {result['note']}")
    finally:
        await close_graph()

    write_report(results)
    print(f"\nwrote {REPORT_PATH}")
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
