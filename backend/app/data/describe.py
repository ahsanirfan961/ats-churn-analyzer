from app.data.store import NUMERIC_COLUMNS, scored_df, unknown_column_error


def describe_column(column: str) -> dict:
    if column not in scored_df.columns:
        return unknown_column_error(column)

    series = scored_df[column]
    if column not in NUMERIC_COLUMNS:
        return {
            "column": column,
            "type": "categorical",
            "count": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "value_counts": {str(k): int(v) for k, v in series.value_counts().items()},
        }

    q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
    iqr = q3 - q1
    outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
    return {
        "column": column,
        "type": "numeric",
        "count": int(series.notna().sum()),
        "missing": int(series.isna().sum()),
        "mean": round(float(series.mean()), 4),
        "median": round(float(series.median()), 4),
        "std": round(float(series.std()), 4),
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "outlier_count": int(len(outliers)),
        "example_outliers": [round(float(v), 4) for v in outliers.head(5)],
    }
