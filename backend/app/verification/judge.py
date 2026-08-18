import json
import re
from collections.abc import AsyncIterator

from app.verification.models import ClaimVerdict, FaithfulnessResult

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

Rounding: treat display rounding as grounded. If a tool returns 0.5835 (58.35%) and the draft says
58.4%, that is grounded — normal one-decimal rounding. The same applies to counts and other figures
rounded for readability. Only flag a number when it is materially wrong, not when it is a rounded
presentation of a tool value.

Reply with one JSON object per line (NDJSON). Output each claim verdict on its own line as you
complete it, with no wrapping array or prose. Example:
{"text": "41.9% churn rate", "verdict": "grounded", "source_tool_call_index": 0, "reason": null}
{"text": "30 months tenure", "verdict": "grounded", "source_tool_call_index": 0, "reason": null}

If the draft makes no specific numeric claims, output nothing."""

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


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


def _extract_numbers_from_text(text: str) -> list[float]:
    values = [float(match.group(1)) for match in _PERCENT_RE.finditer(text)]
    if values:
        return values
    return [float(match) for match in _NUMBER_RE.findall(text)]


def _all_numeric_values(value) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return _extract_numbers_from_text(value)
    if isinstance(value, dict):
        numbers: list[float] = []
        for nested in value.values():
            numbers.extend(_all_numeric_values(nested))
        return numbers
    if isinstance(value, list):
        numbers: list[float] = []
        for nested in value:
            numbers.extend(_all_numeric_values(nested))
        return numbers
    return []


def _tool_numeric_values(tool_calls: list[dict]) -> list[float]:
    numbers: list[float] = []
    for call in tool_calls:
        numbers.extend(_all_numeric_values(call.get("args")))
        numbers.extend(_all_numeric_values(call.get("output")))
    return numbers


def _display_scales(value: float) -> list[float]:
    scales = [value]
    if 0 <= value <= 1:
        scales.append(value * 100)
    elif 1 < value <= 100:
        scales.append(value / 100)
    return scales


def _matches_within_rounding(claimed: float, tool_values: list[float], tolerance: float = 0.15) -> bool:
    for raw in tool_values:
        for candidate in _display_scales(raw):
            for decimals in range(4):
                if abs(round(candidate, decimals) - claimed) < 1e-9:
                    return True
            if abs(candidate - claimed) <= tolerance:
                return True
    return False


def apply_rounding_tolerance(claim: ClaimVerdict, tool_calls: list[dict]) -> ClaimVerdict:
    if claim.verdict != "fabricated":
        return claim

    claimed_numbers = _extract_numbers_from_text(claim.text)
    if not claimed_numbers:
        return claim

    tool_values = _tool_numeric_values(tool_calls)
    if all(_matches_within_rounding(number, tool_values) for number in claimed_numbers):
        return claim.model_copy(update={"verdict": "grounded", "reason": None})
    return claim


class _NdjsonClaimParser:
    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[ClaimVerdict]:
        self._buffer += text
        claims: list[ClaimVerdict] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            claim = _parse_claim_line(line)
            if claim is not None:
                claims.append(claim)
        return claims

    def finish(self) -> tuple[list[ClaimVerdict], str | None]:
        claims = []
        leftover = self._buffer.strip()
        if leftover:
            if leftover.startswith("```"):
                leftover = _strip_fences(leftover)
            claim = _parse_claim_line(leftover)
            if claim is not None:
                claims.append(claim)
            elif leftover:
                legacy = _parse_legacy_json(leftover)
                if legacy is not None:
                    return legacy.claims, legacy.parse_error
                return [], "judge did not return valid NDJSON"
        self._buffer = ""
        return claims, None


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _parse_claim_line(line: str) -> ClaimVerdict | None:
    line = line.strip()
    if not line or line.startswith("```"):
        return None
    try:
        return ClaimVerdict.model_validate(json.loads(line))
    except Exception:
        return None


def _parse_legacy_json(text: str) -> FaithfulnessResult | None:
    text = _strip_fences(text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return FaithfulnessResult.model_validate(json.loads(text[start:end + 1]))
    except Exception as exc:
        return FaithfulnessResult(parse_error=f"could not parse judge output: {exc}")


def _chunk_text(chunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


async def stream_faithfulness(draft_answer: str, tool_calls: list[dict],
                              judge_model) -> AsyncIterator[ClaimVerdict]:
    parser = _NdjsonClaimParser()
    try:
        async for chunk in judge_model.astream(build_judge_prompt(draft_answer, tool_calls)):
            for claim in parser.feed(_chunk_text(chunk)):
                yield apply_rounding_tolerance(claim, tool_calls)
    except Exception as exc:
        raise RuntimeError(f"judge call failed: {type(exc).__name__}: {exc}") from exc

    trailing, error = parser.finish()
    for claim in trailing:
        yield apply_rounding_tolerance(claim, tool_calls)
    if error:
        raise ValueError(error)


async def check_faithfulness(draft_answer: str, tool_calls: list[dict],
                             judge_model) -> FaithfulnessResult:
    claims: list[ClaimVerdict] = []
    try:
        async for claim in stream_faithfulness(draft_answer, tool_calls, judge_model):
            claims.append(claim)
    except RuntimeError as exc:
        return FaithfulnessResult(parse_error=str(exc))
    except ValueError as exc:
        return FaithfulnessResult(parse_error=str(exc))
    return FaithfulnessResult(claims=claims)
