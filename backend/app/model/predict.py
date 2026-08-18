import pandas as pd

from app.data.store import df
from app.model.loader import (
    all_feature_names,
    feature_cols,
    numeric_cols,
    pipeline,
)


def _score_row(x: pd.DataFrame, top_k: int) -> tuple[float, list[dict]]:
    risk_score = float(pipeline.predict_proba(x)[0, 1])

    transformed = pipeline.named_steps["pre"].transform(x)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    contributions = transformed[0] * pipeline.named_steps["clf"].coef_[0]

    ranked = sorted(range(len(contributions)), key=lambda i: abs(contributions[i]), reverse=True)
    top_factors = [
        {
            "feature": all_feature_names[i],
            "direction": "increases_risk" if contributions[i] > 0 else "decreases_risk",
            "contribution": round(float(contributions[i]), 4),
        }
        for i in ranked[:top_k]
    ]
    return round(risk_score, 4), top_factors


def _dataset_default(column: str):
    if column in numeric_cols:
        return float(df[column].median())
    value = df[column].mode().iloc[0]
    return value.item() if hasattr(value, "item") else value


def predict_churn_risk(customer_id: str, top_k: int = 3) -> dict:
    row = df.loc[df["customerID"] == customer_id]
    if row.empty:
        return {"error": f"customer_id {customer_id!r} not found in the dataset"}

    risk_score, top_factors = _score_row(row[feature_cols], top_k)
    return {"customer_id": customer_id, "risk_score": risk_score, "top_factors": top_factors}


def predict_hypothetical(features: dict, top_k: int = 3) -> dict:
    unknown = [c for c in features if c not in feature_cols]
    if unknown:
        return {"error": f"unknown feature fields {unknown}; valid fields are {feature_cols}"}

    filled = {}
    defaulted_fields = []
    for column in feature_cols:
        if features.get(column) is None:
            filled[column] = _dataset_default(column)
            defaulted_fields.append(column)
        else:
            filled[column] = features[column]

    risk_score, top_factors = _score_row(pd.DataFrame([filled])[feature_cols], top_k)
    return {"risk_score": risk_score, "top_factors": top_factors, "defaulted_fields": defaulted_fields}


def compare_scenarios(changes: dict, customer_id: str | None = None,
                      base_features: dict | None = None, top_k: int = 3) -> dict:
    if (customer_id is None) == (base_features is None):
        return {"error": "provide exactly one baseline: either customer_id or base_features"}
    if not changes:
        return {"error": "changes must contain at least one field to override"}

    if customer_id is not None:
        row = df.loc[df["customerID"] == customer_id]
        if row.empty:
            return {"error": f"customer_id {customer_id!r} not found in the dataset"}
        baseline_features = row[feature_cols].iloc[0].to_dict()
    else:
        baseline_features = dict(base_features)

    baseline = predict_hypothetical(baseline_features, top_k)
    if "error" in baseline:
        return baseline
    modified = predict_hypothetical({**baseline_features, **changes}, top_k)
    if "error" in modified:
        return modified

    return {
        "baseline": baseline,
        "modified": modified,
        "delta": round(modified["risk_score"] - baseline["risk_score"], 4),
        "changed_fields": list(changes),
    }
