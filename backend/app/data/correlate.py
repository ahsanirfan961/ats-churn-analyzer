import numpy as np
import pandas as pd
from scipy import stats

from app.data.store import NUMERIC_COLUMNS, scored_df, unknown_column_error


def _strength(magnitude: float) -> str:
    if magnitude < 0.1:
        return "negligible"
    if magnitude < 0.3:
        return "weak"
    if magnitude < 0.5:
        return "moderate"
    return "strong"


def _pearson(a, b, col_a, col_b) -> dict:
    r, p = stats.pearsonr(a, b)
    direction = "higher" if r > 0 else "lower"
    return {
        "method": "pearson",
        "statistic": round(float(r), 4),
        "p_value": round(float(p), 6),
        "n": int(len(a)),
        "interpretation": f"{_strength(abs(r))} relationship: higher {col_a} goes with {direction} {col_b}",
    }


def _point_biserial(numeric, categorical, num_col, cat_col) -> dict:
    levels = sorted(str(v) for v in categorical.unique())
    binary = (categorical.astype(str) == levels[1]).astype(int)
    r, p = stats.pointbiserialr(binary, numeric)
    direction = "higher" if r > 0 else "lower"
    return {
        "method": "point-biserial",
        "statistic": round(float(r), 4),
        "p_value": round(float(p), 6),
        "n": int(len(numeric)),
        "interpretation": (f"{_strength(abs(r))} relationship: {cat_col}={levels[1]} customers have "
                           f"{direction} {num_col} than {cat_col}={levels[0]} customers"),
    }


def _correlation_ratio(numeric, categorical, num_col, cat_col) -> dict:
    groups = [numeric[categorical == level] for level in categorical.unique()]
    grand_mean = numeric.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = float(((numeric - grand_mean) ** 2).sum())
    eta = float(np.sqrt(ss_between / ss_total)) if ss_total else 0.0
    _, p = stats.f_oneway(*groups)
    return {
        "method": "correlation-ratio (eta)",
        "statistic": round(eta, 4),
        "p_value": round(float(p), 6),
        "n": int(len(numeric)),
        "interpretation": f"{_strength(eta)} association: {num_col} varies by {cat_col} group",
    }


def _cramers_v(a, b, col_a, col_b) -> dict:
    table = pd.crosstab(a, b).values
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = int(table.sum())
    v = float(np.sqrt((chi2 / n) / (min(table.shape) - 1)))
    return {
        "method": "cramers-v",
        "statistic": round(v, 4),
        "p_value": round(float(p), 6),
        "n": n,
        "interpretation": f"{_strength(v)} association between {col_a} and {col_b}",
    }


def correlation(col_a: str, col_b: str) -> dict:
    for column in (col_a, col_b):
        if column not in scored_df.columns:
            return unknown_column_error(column)
    if col_a == col_b:
        return {"error": "col_a and col_b must be different columns"}

    a, b = scored_df[col_a], scored_df[col_b]
    a_numeric, b_numeric = col_a in NUMERIC_COLUMNS, col_b in NUMERIC_COLUMNS

    if a_numeric and b_numeric:
        return _pearson(a, b, col_a, col_b)
    if a_numeric or b_numeric:
        numeric, num_col = (a, col_a) if a_numeric else (b, col_b)
        categorical, cat_col = (b, col_b) if a_numeric else (a, col_a)
        if categorical.nunique() == 2:
            return _point_biserial(numeric, categorical, num_col, cat_col)
        return _correlation_ratio(numeric, categorical, num_col, cat_col)
    return _cramers_v(a, b, col_a, col_b)
