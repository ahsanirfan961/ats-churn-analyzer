from langchain_core.tools import StructuredTool

from app.data.correlate import correlation as _correlation
from app.data.describe import describe_column as _describe_column
from app.data.rank import rank_customers as _rank_customers
from app.data.segment import segment_stats as _segment_stats
from app.model.predict import compare_scenarios as _compare_scenarios
from app.model.predict import predict_churn_risk as _predict_churn_risk
from app.model.predict import predict_hypothetical as _predict_hypothetical
from app.sandbox.run_pandas import run_pandas as _run_pandas
from app.tools.schemas import (
    CompareScenariosInput,
    CorrelationInput,
    DescribeColumnInput,
    MetricSpec,
    PredictChurnRiskInput,
    PredictHypotheticalInput,
    RankCustomersInput,
    RunPandasInput,
    SegmentStatsInput,
)


def _segment_stats_tool(filters=None, group_by=None, metrics=None) -> dict:
    pairs = [(m.column, m.aggfunc) if isinstance(m, MetricSpec) else tuple(m)
             for m in (metrics or [])]
    return _segment_stats(filters or {}, group_by, pairs)


def get_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_predict_churn_risk,
            name="predict_churn_risk",
            description="Score one existing customer with the trained churn model. Returns a "
                        "continuous risk_score plus the top contributing factors, each with the "
                        "customer's own value and how that value's cohort churns against the "
                        "overall baseline.",
            args_schema=PredictChurnRiskInput,
        ),
        StructuredTool.from_function(
            func=_predict_hypothetical,
            name="predict_hypothetical",
            description="Score a hypothetical customer described by a partial feature dict. "
                        "Unspecified features are filled from the dataset and listed in "
                        "defaulted_fields, which you must mention when summarising the answer.",
            args_schema=PredictHypotheticalInput,
        ),
        StructuredTool.from_function(
            func=_compare_scenarios,
            name="compare_scenarios",
            description="What-if comparison. Takes a baseline (an existing customer_id or a "
                        "base_features dict) plus a dict of changes, and returns both risk "
                        "scores and the delta. Use this for what-if questions instead of "
                        "subtracting two separate predictions yourself.",
            args_schema=CompareScenariosInput,
        ),
        StructuredTool.from_function(
            func=_describe_column,
            name="describe_column",
            description="Univariate summary of a single column: distribution statistics and IQR "
                        "outliers for numeric columns, value counts for categorical ones.",
            args_schema=DescribeColumnInput,
        ),
        StructuredTool.from_function(
            func=_segment_stats_tool,
            name="segment_stats",
            description="Cohort aggregation. Filters the customer base, optionally groups by up "
                        "to two columns, and reports n and churn rate per group plus any extra "
                        "metrics requested. Main tool for which-segment-churns-most questions.",
            args_schema=SegmentStatsInput,
        ),
        StructuredTool.from_function(
            func=_rank_customers,
            name="rank_customers",
            description="Top-N listing. Sorts the customer base by any column, including the "
                        "model's risk_score, after applying optional filters. A plain sort with "
                        "no risk threshold applied.",
            args_schema=RankCustomersInput,
        ),
        StructuredTool.from_function(
            func=_correlation,
            name="correlation",
            description="Association between two columns, with the statistical test chosen from "
                        "their dtypes. Returns the statistic, p-value, sample size and a "
                        "plain-English reading of direction and strength.",
            args_schema=CorrelationInput,
        ),
        StructuredTool.from_function(
            func=_run_pandas,
            name="run_pandas",
            description="Escape hatch that runs restricted pandas code against the customer "
                        "table. Only reach for this when none of the other tools can express "
                        "the question.",
            args_schema=RunPandasInput,
        ),
    ]
