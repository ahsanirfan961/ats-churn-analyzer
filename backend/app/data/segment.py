from app.data.store import NUMERIC_COLUMNS, scored_df, unknown_column_error

AGGFUNCS = {"mean", "sum", "count", "median"}


def validate_filters(filters: dict) -> dict | None:
    for column, value in filters.items():
        if column not in scored_df.columns:
            return unknown_column_error(column)
        if column not in NUMERIC_COLUMNS and value not in set(scored_df[column]):
            valid = sorted(str(v) for v in scored_df[column].unique())
            return {"error": f"{value!r} is not a value of {column!r}; valid values are {valid}"}
    return None


def _summarize(frame, metrics: list[tuple[str, str]]) -> dict:
    summary = {
        "n": int(len(frame)),
        "churn_rate": round(float((frame["Churn"] == "Yes").mean()), 4),
    }
    for column, aggfunc in metrics:
        summary[f"{aggfunc}_{column}"] = round(float(frame[column].agg(aggfunc)), 4)
    return summary


def segment_stats(filters: dict, group_by: list[str] | None = None,
                  metrics: list[tuple[str, str]] | None = None) -> dict:
    filters = filters or {}
    group_by = group_by or []
    metrics = [tuple(m) for m in (metrics or [])]

    invalid = validate_filters(filters)
    if invalid:
        return invalid
    for column in group_by:
        if column not in scored_df.columns:
            return unknown_column_error(column)
    if len(group_by) > 2:
        return {"error": "group_by supports at most 2 columns"}
    for column, aggfunc in metrics:
        if column not in scored_df.columns:
            return unknown_column_error(column)
        if aggfunc not in AGGFUNCS:
            return {"error": f"unknown aggfunc {aggfunc!r}; valid aggfuncs are {sorted(AGGFUNCS)}"}
        if aggfunc != "count" and column not in NUMERIC_COLUMNS:
            return {"error": f"{aggfunc!r} needs a numeric column, but {column!r} is categorical"}

    subset = scored_df
    for column, value in filters.items():
        subset = subset[subset[column] == value]
    if subset.empty:
        return {"error": f"no rows matched the filters {filters}"}

    if not group_by:
        return {"n_matched": int(len(subset)), "groups": [{"key": {}, **_summarize(subset, metrics)}]}

    groups = []
    for key, frame in subset.groupby(group_by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        groups.append({
            "key": {c: (v.item() if hasattr(v, "item") else v) for c, v in zip(group_by, key)},
            **_summarize(frame, metrics),
        })
    return {"n_matched": int(len(subset)), "groups": groups}
