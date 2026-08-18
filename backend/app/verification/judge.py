import json

from app.verification.models import FaithfulnessResult

JUDGE_INSTRUCTIONS = """You are auditing a data analyst's draft answer against the tool results it was written from.

Work claim by claim:
1. List every specific numeric or statistical claim in the draft answer. A claim is specific if it
   states a number, a rate, a count, a ranking, a statistic, or a named customer or segment value.
   Ignore purely qualitative wording that states no figure.
2. For each claim, find the tool result that supports it. A claim is only supported when the tool
   result both contains that value AND measures the same thing the claim says it measures. A value
   that appears somewhere in the tool results but describes a different metric, a different segment,
   or a different customer does NOT support the claim.
3. Give each claim a verdict:
   - "grounded": a tool result contains the value and measures exactly what the claim says.
   - "mislabeled": the value exists in a tool result but is attached to the wrong metric, segment,
     customer or unit in the draft.
   - "fabricated": no tool result contains the value at all.
   Give a one-line reason for anything that is not grounded, and set source_tool_call_index to the
   index of the tool result you compared against (null when nothing matched).

Reply with strict JSON and nothing else, in this shape:
{"claims": [{"text": "...", "verdict": "grounded", "source_tool_call_index": 0, "reason": null}]}
If the draft makes no specific numeric claims, reply {"claims": []}."""


def build_judge_prompt(draft_answer: str, tool_calls: list[dict]) -> str:
    if tool_calls:
        rendered = "\n\n".join(
            f"[{i}] tool: {call.get('tool_name')}\n"
            f"    args: {json.dumps(call.get('args'), default=str)}\n"
            f"    output: {json.dumps(call.get('output'), default=str)}"
            for i, call in enumerate(tool_calls)
        )
    else:
        rendered = "(no tool results are available for this answer)"

    return (
        f"{JUDGE_INSTRUCTIONS}\n\n"
        f"TOOL RESULTS\n{rendered}\n\n"
        f"DRAFT ANSWER\n{draft_answer}"
    )


def parse_judge_response(content: str) -> FaithfulnessResult:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return FaithfulnessResult(parse_error="judge did not return JSON")
    try:
        return FaithfulnessResult.model_validate(json.loads(text[start:end + 1]))
    except Exception as exc:
        return FaithfulnessResult(parse_error=f"could not parse judge output: {exc}")


async def check_faithfulness(draft_answer: str, tool_calls: list[dict],
                             judge_model) -> FaithfulnessResult:
    try:
        response = await judge_model.ainvoke(build_judge_prompt(draft_answer, tool_calls))
    except Exception as exc:
        return FaithfulnessResult(parse_error=f"judge call failed: {type(exc).__name__}: {exc}")

    content = response.content if isinstance(response.content, str) else str(response.content)
    return parse_judge_response(content)
