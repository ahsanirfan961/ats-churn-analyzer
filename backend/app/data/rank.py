import json

from app.data.segment import normalize_filters
from app.data.store import scored_df, unknown_column_error


def rank_customers(sort_by: str, filters: dict | None = None, ascending: bool = False,
                   n: int = 10, columns: list[str] | None = None) -> dict:
    filters = filters or {}
    if sort_by not in scored_df.columns:
        return unknown_column_error(sort_by)

    filters, invalid = normalize_filters(filters)
    if invalid:
        return invalid

    columns = columns or ["customerID", sort_by, "risk_score"]
    for column in columns:
        if column not in scored_df.columns:
            return unknown_column_error(column)
    columns = list(dict.fromkeys(["customerID", sort_by, *columns]))

    subset = scored_df
    for column, value in filters.items():
        subset = subset[subset[column] == value]
    if subset.empty:
        return {"error": f"no rows matched the filters {filters}"}

    ranked = subset.sort_values(sort_by, ascending=ascending).head(n)
    return {
        "n_matched": int(len(subset)),
        "sorted_by": sort_by,
        "ascending": ascending,
        "customers": json.loads(ranked[columns].to_json(orient="records")),
    }
