from scipy import stats

from app.data.store import (
    COLUMN_UNITS,
    NUMERIC_COLUMNS,
    baseline_churn_rate,
    decile_tables,
    scored_df,
    segment_tables,
)


def _decile_row(column: str, value: float):
    table = decile_tables[column]
    try:
        return table.iloc[table.index.get_loc(value)]
    except KeyError:
        return table.iloc[0] if value < table["low"].iloc[0] else table.iloc[-1]


def enrich_factor(feature_name: str, customer_value) -> dict:
    if feature_name in NUMERIC_COLUMNS:
        row = _decile_row(feature_name, customer_value)
        unit = COLUMN_UNITS.get(feature_name, "")
        return {
            "percentile": round(float(stats.percentileofscore(scored_df[feature_name], customer_value, kind="strict")), 1),
            "decile_range": f"{row['low']:g}-{row['high']:g} {unit}".strip(),
            "decile_n": int(row["n"]),
            "decile_churn_rate": float(row["churn_rate"]),
            "baseline_churn_rate": baseline_churn_rate,
        }

    table = segment_tables.get(feature_name)
    if table is None or customer_value not in table.index:
        return {"baseline_churn_rate": baseline_churn_rate}

    row = table.loc[customer_value]
    return {
        "segment_n": int(row["n"]),
        "share_of_base": float(row["share"]),
        "segment_churn_rate": float(row["churn_rate"]),
        "baseline_churn_rate": baseline_churn_rate,
    }
