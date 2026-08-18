import pytest

from app.data.store import df
from app.model.loader import feature_cols
from app.model.predict import compare_scenarios, predict_churn_risk, predict_hypothetical

KNOWN_ID = "4424-TKOPW"


@pytest.fixture
def known_features():
    return df.loc[df["customerID"] == KNOWN_ID, feature_cols].iloc[0].to_dict()


def test_predict_churn_risk_known_customer():
    result = predict_churn_risk(KNOWN_ID)
    assert result["customer_id"] == KNOWN_ID
    assert 0.0 <= result["risk_score"] <= 1.0
    assert len(result["top_factors"]) == 3
    assert result["top_factors"][0]["direction"] in {"increases_risk", "decreases_risk"}


def test_predict_churn_risk_unknown_customer():
    result = predict_churn_risk("NOT-A-REAL-ID")
    assert "error" in result
    assert "not found" in result["error"]


def test_predict_hypothetical_fully_specified(known_features):
    result = predict_hypothetical(known_features)
    assert result["defaulted_fields"] == []
    assert result["risk_score"] == predict_churn_risk(KNOWN_ID)["risk_score"]


def test_predict_hypothetical_partial_input():
    result = predict_hypothetical({"Contract": "Two year", "tenure": 60})
    assert set(result["defaulted_fields"]) == set(feature_cols) - {"Contract", "tenure"}
    assert 0.0 <= result["risk_score"] <= 1.0


def test_predict_hypothetical_rejects_unknown_field():
    result = predict_hypothetical({"region": "north"})
    assert "error" in result


def test_compare_scenarios_delta_matches_manual_subtraction():
    result = compare_scenarios({"Contract": "Two year"}, customer_id=KNOWN_ID)
    manual = result["modified"]["risk_score"] - result["baseline"]["risk_score"]
    assert result["delta"] == pytest.approx(manual, abs=1e-9)
    assert result["changed_fields"] == ["Contract"]


def test_compare_scenarios_accepts_base_features(known_features):
    result = compare_scenarios({"tenure": 72}, base_features=known_features)
    assert result["baseline"]["risk_score"] == predict_churn_risk(KNOWN_ID)["risk_score"]
    assert result["modified"]["risk_score"] != result["baseline"]["risk_score"]


def test_compare_scenarios_requires_exactly_one_baseline(known_features):
    assert "error" in compare_scenarios({"tenure": 1})
    assert "error" in compare_scenarios({"tenure": 1}, customer_id=KNOWN_ID, base_features=known_features)


def test_top_factors_carry_numeric_distribution_context():
    factors = predict_churn_risk(KNOWN_ID)["top_factors"]
    tenure = next(f for f in factors if f["feature"] == "tenure")
    assert tenure["customer_value"] == 2
    assert set(tenure["context"]) == {"percentile", "decile_range", "decile_n",
                                      "decile_churn_rate", "baseline_churn_rate"}
    assert tenure["context"]["decile_churn_rate"] > tenure["context"]["baseline_churn_rate"]


def test_top_factors_carry_categorical_segment_context():
    factors = predict_churn_risk(KNOWN_ID)["top_factors"]
    internet = next(f for f in factors if f["feature"] == "InternetService")
    assert internet["customer_value"] == "Fiber optic"
    assert set(internet["context"]) == {"segment_n", "share_of_base",
                                        "segment_churn_rate", "baseline_churn_rate"}
    assert internet["context"]["segment_n"] == 3096


def test_hypothetical_factors_are_enriched_too():
    factors = predict_hypothetical({"Contract": "Month-to-month", "tenure": 1})["top_factors"]
    assert all("context" in f and "baseline_churn_rate" in f["context"] for f in factors)
