import json

import pytest

from app.verification.judge import apply_rounding_tolerance, build_judge_prompt, check_faithfulness
from app.verification.models import FaithfulnessResult

TOOL_CALLS = [
    {
        "tool_name": "segment_stats",
        "args": {"filters": {"InternetService": "DSL"}},
        "output": {"n_matched": 2421, "groups": [{"key": {}, "n": 2421, "churn_rate": 0.42}]},
    },
    {
        "tool_name": "predict_churn_risk",
        "args": {"customer_id": "4424-TKOPW"},
        "output": {"customer_id": "4424-TKOPW", "risk_score": 0.8502},
    },
]


def judge_ndjson(*claims):
    return "\n".join(json.dumps(c) for c in claims)


def judge_payload(*claims):
    return judge_ndjson(*claims)


class FakeJudge:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    async def astream(self, prompt):
        self.prompts.append(prompt)
        yield type("Chunk", (), {"content": self.content})()

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return type("Response", (), {"content": self.content})()


class FailingJudge:
    async def astream(self, prompt):
        raise RuntimeError("upstream 429")
        yield

    async def ainvoke(self, prompt):
        raise RuntimeError("upstream 429")


def test_prompt_contains_draft_and_every_tool_result():
    prompt = build_judge_prompt("DSL churns at 42%.", TOOL_CALLS)
    assert "DSL churns at 42%." in prompt
    assert "segment_stats" in prompt and "predict_churn_risk" in prompt
    assert "[0]" in prompt and "[1]" in prompt


@pytest.mark.asyncio
async def test_grounded_draft_passes():
    judge = FakeJudge(judge_payload(
        {"text": "DSL customers churn at 42%", "verdict": "grounded", "source_tool_call_index": 0}))
    result = await check_faithfulness("DSL customers churn at 42%.", TOOL_CALLS, judge)
    assert result.is_faithful
    assert result.problems == []


@pytest.mark.asyncio
async def test_fabricated_number_is_flagged():
    judge = FakeJudge(judge_payload(
        {"text": "DSL customers churn at 61%", "verdict": "fabricated",
         "source_tool_call_index": None, "reason": "no tool result contains 61%"}))
    result = await check_faithfulness("DSL customers churn at 61%.", TOOL_CALLS, judge)
    assert not result.is_faithful
    assert result.problems[0].verdict == "fabricated"


@pytest.mark.asyncio
async def test_mislabeled_number_is_flagged():
    draft = "Customer 4424-TKOPW has a risk score of 42%."
    judge = FakeJudge(judge_payload(
        {"text": "risk score of 42%", "verdict": "mislabeled", "source_tool_call_index": 0,
         "reason": "42% is the DSL segment churn rate, not this customer's risk score"}))
    result = await check_faithfulness(draft, TOOL_CALLS, judge)
    assert not result.is_faithful
    assert result.problems[0].verdict == "mislabeled"
    assert "churn rate" in result.problems[0].reason


@pytest.mark.asyncio
async def test_fenced_json_is_still_parsed():
    judge = FakeJudge("```json\n" + judge_ndjson(
        {"text": "risk score 0.8502", "verdict": "grounded", "source_tool_call_index": 1}) + "\n```")
    result = await check_faithfulness("Risk score 0.8502.", TOOL_CALLS, judge)
    assert result.is_faithful and len(result.claims) == 1


@pytest.mark.asyncio
async def test_legacy_wrapped_json_is_still_parsed():
    judge = FakeJudge(json.dumps({"claims": [
        {"text": "DSL customers churn at 42%", "verdict": "grounded", "source_tool_call_index": 0},
    ]}))
    result = await check_faithfulness("DSL customers churn at 42%.", TOOL_CALLS, judge)
    assert result.is_faithful and len(result.claims) == 1


@pytest.mark.asyncio
async def test_malformed_judge_output_does_not_crash():
    result = await check_faithfulness("anything", TOOL_CALLS, FakeJudge("I think it looks fine!"))
    assert isinstance(result, FaithfulnessResult)
    assert result.parse_error is not None
    assert result.claims == []


@pytest.mark.asyncio
async def test_judge_api_failure_does_not_crash():
    result = await check_faithfulness("anything", TOOL_CALLS, FailingJudge())
    assert result.parse_error is not None and "judge call failed" in result.parse_error


def test_rounding_difference_is_treated_as_grounded():
    from app.verification.models import ClaimVerdict

    claim = apply_rounding_tolerance(
        ClaimVerdict(
            text="58.4%",
            verdict="fabricated",
            source_tool_call_index=0,
            reason="Tool has 0.5835 (58.35%), not 58.4%",
        ),
        [{"tool_name": "segment_stats", "args": {}, "output": {"churn_rate": 0.5835}}],
    )
    assert claim.verdict == "grounded"


def test_materially_wrong_number_stays_fabricated():
    from app.verification.models import ClaimVerdict

    claim = apply_rounding_tolerance(
        ClaimVerdict(
            text="DSL customers churn at 61%",
            verdict="fabricated",
            source_tool_call_index=None,
            reason="no tool result contains 61%",
        ),
        TOOL_CALLS,
    )
    assert claim.verdict == "fabricated"


def test_mislabeled_claim_is_not_relaxed_by_rounding():
    from app.verification.models import ClaimVerdict

    claim = apply_rounding_tolerance(
        ClaimVerdict(
            text="risk score of 42%",
            verdict="mislabeled",
            source_tool_call_index=0,
            reason="42% is the DSL churn rate, not this customer's risk score",
        ),
        TOOL_CALLS,
    )
    assert claim.verdict == "mislabeled"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_live_judge_returns_parsable_verdicts():
    from app.llm import chat_model

    draft = "DSL customers churn at 42%, and customer 4424-TKOPW has a risk score of 0.8502."
    result = await check_faithfulness(draft, TOOL_CALLS, chat_model())
    assert result.parse_error is None
    assert len(result.claims) >= 1
