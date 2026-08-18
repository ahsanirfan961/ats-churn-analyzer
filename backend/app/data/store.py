import pandas as pd

from app.config import DATASET_PATH
from app.model.loader import feature_cols, numeric_cols, pipeline

df = pd.read_csv(DATASET_PATH)
churned = (df["Churn"] == "Yes").astype(int)

scored_df = df.copy()
scored_df["risk_score"] = pipeline.predict_proba(df[feature_cols])[:, 1].round(4)

row_count = len(df)
baseline_churn_rate = round(float(churned.mean()), 4)

NUMERIC_COLUMNS = set(numeric_cols) | {"risk_score"}
CATEGORICAL_COLUMNS = [c for c in df.columns if c not in NUMERIC_COLUMNS and c != "customerID"]
COLUMN_UNITS = {"tenure": "months", "MonthlyCharges": "$", "TotalCharges": "$"}

categorical_levels = {c: sorted(str(v) for v in df[c].unique()) for c in CATEGORICAL_COLUMNS}


def _decile_table(column: str) -> pd.DataFrame:
    deciles = pd.qcut(scored_df[column], 10, duplicates="drop")
    grouped = churned.groupby(deciles, observed=True)
    return pd.DataFrame({"n": grouped.size(), "churn_rate": grouped.mean().round(4)})


def _segment_table(column: str) -> pd.DataFrame:
    grouped = churned.groupby(df[column], observed=True)
    counts = grouped.size()
    return pd.DataFrame({
        "n": counts,
        "share": (counts / row_count).round(4),
        "churn_rate": grouped.mean().round(4),
    })


decile_tables = {c: _decile_table(c) for c in NUMERIC_COLUMNS}
segment_tables = {c: _segment_table(c) for c in CATEGORICAL_COLUMNS}


def column_names() -> list[str]:
    return list(scored_df.columns)


def unknown_column_error(column: str) -> dict:
    return {"error": f"unknown column {column!r}; valid columns are {column_names()}"}
