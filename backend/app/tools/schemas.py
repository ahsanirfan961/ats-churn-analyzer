from typing import Any, Literal

from pydantic import BaseModel, Field

FILTERS_DESCRIPTION = (
    "Column=value equality filters, ANDed together, e.g. {'Contract': 'Month-to-month', "
    "'InternetService': 'Fiber optic'}. Use exact category values as they appear in the "
    "dataset. Omit or pass {} for the whole customer base."
)

FEATURES_DESCRIPTION = (
    "Partial customer description, e.g. {'Contract': 'Month-to-month', 'tenure': 3, "
    "'InternetService': 'Fiber optic'}. Any feature you leave out is filled with the "
    "dataset median (numeric) or mode (categorical) and reported back in defaulted_fields."
)


class PredictChurnRiskInput(BaseModel):
    customer_id: str = Field(description="Exact customerID from the dataset, e.g. '4424-TKOPW'.")
    top_k: int = Field(default=3, description="How many contributing factors to explain.")


class PredictHypotheticalInput(BaseModel):
    features: dict[str, Any] = Field(description=FEATURES_DESCRIPTION)
    top_k: int = Field(default=3, description="How many contributing factors to explain.")


class CompareScenariosInput(BaseModel):
    changes: dict[str, Any] = Field(
        description="Fields to override on the baseline, e.g. {'Contract': 'Two year'}.")
    customer_id: str | None = Field(
        default=None, description="Use this existing customer as the baseline.")
    base_features: dict[str, Any] | None = Field(
        default=None,
        description="Use this hypothetical customer as the baseline instead. Give exactly one "
                    "of customer_id or base_features.")
    top_k: int = Field(default=3, description="How many contributing factors to explain.")


class DescribeColumnInput(BaseModel):
    column: str = Field(
        description="Column to summarise. Numeric columns return spread and IQR outliers, "
                    "categorical columns return value counts.")


class MetricSpec(BaseModel):
    column: str = Field(description="Numeric column to aggregate, e.g. 'MonthlyCharges'.")
    aggfunc: Literal["mean", "sum", "count", "median"] = Field(description="Aggregation to apply.")


class SegmentStatsInput(BaseModel):
    filters: dict[str, Any] = Field(default_factory=dict, description=FILTERS_DESCRIPTION)
    group_by: list[str] | None = Field(
        default=None,
        description="Up to 2 columns to break the result down by, e.g. ['Contract']. "
                    "Every group always reports n and churn_rate.")
    metrics: list[MetricSpec] | None = Field(
        default=None, description="Extra aggregations to compute per group, on top of churn rate.")


class RankCustomersInput(BaseModel):
    sort_by: str = Field(
        description="Column to sort on. Use 'risk_score' for model-predicted churn risk, or any "
                    "raw column such as 'MonthlyCharges' or 'tenure'.")
    filters: dict[str, Any] = Field(default_factory=dict, description=FILTERS_DESCRIPTION)
    ascending: bool = Field(default=False, description="False returns the highest values first.")
    n: int = Field(default=10, description="How many customers to return.")
    columns: list[str] | None = Field(
        default=None, description="Columns to include per customer. customerID is always included.")


class CorrelationInput(BaseModel):
    col_a: str = Field(description="First column.")
    col_b: str = Field(
        description="Second column. The statistical test is chosen from the two dtypes: Pearson "
                    "for numeric pairs, point-biserial for numeric vs two-level categorical, "
                    "correlation ratio for numeric vs multi-level categorical, Cramer's V for "
                    "categorical pairs.")


class RunPandasInput(BaseModel):
    code: str = Field(
        description="Python expression or short script evaluated against 'df' (the full customer "
                    "table plus a risk_score column), with 'pd' and 'np' available. The value of "
                    "the last expression is returned. Imports, file access and attribute tricks "
                    "are rejected. Use this only when no other tool can answer the question.")
