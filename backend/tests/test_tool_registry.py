import pytest

from app.tools.registry import get_tools

SAMPLE_INPUTS = {
    "predict_churn_risk": {"customer_id": "4424-TKOPW"},
    "predict_hypothetical": {"features": {"Contract": "Month-to-month", "tenure": 3}},
    "compare_scenarios": {"changes": {"Contract": "Two year"}, "customer_id": "4424-TKOPW"},
    "describe_column": {"column": "tenure"},
    "segment_stats": {"filters": {"Contract": "Month-to-month"}, "group_by": ["InternetService"],
                      "metrics": [{"column": "MonthlyCharges", "aggfunc": "mean"}]},
    "rank_customers": {"sort_by": "risk_score", "n": 5},
    "correlation": {"col_a": "tenure", "col_b": "Churn"},
    "run_pandas": {"code": "len(df)"},
}


def test_registry_exposes_exactly_the_eight_planned_tools():
    names = [t.name for t in get_tools()]
    assert names == list(SAMPLE_INPUTS)


@pytest.mark.parametrize("tool", get_tools(), ids=lambda t: t.name)
def test_each_tool_invokes_through_the_langchain_wrapper(tool):
    result = tool.invoke(SAMPLE_INPUTS[tool.name])
    assert isinstance(result, dict)
    assert "error" not in result


@pytest.mark.parametrize("tool", get_tools(), ids=lambda t: t.name)
def test_each_tool_has_a_description_and_schema(tool):
    assert len(tool.description) > 40
    assert tool.args_schema is not None


def test_schema_validation_rejects_missing_required_field():
    describe = next(t for t in get_tools() if t.name == "describe_column")
    with pytest.raises(Exception):
        describe.invoke({})
