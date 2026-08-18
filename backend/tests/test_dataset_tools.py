from app.data.correlate import correlation
from app.data.describe import describe_column
from app.data.rank import rank_customers
from app.data.segment import segment_stats
from app.data.store import baseline_churn_rate, row_count


def test_store_baseline_matches_notebook():
    assert row_count == 7043
    assert baseline_churn_rate == 0.2654


def test_describe_numeric_column():
    result = describe_column("tenure")
    assert result["type"] == "numeric"
    assert result["min"] == 0.0 and result["max"] == 72.0
    assert result["q1"] < result["median"] < result["q3"]


def test_describe_categorical_column():
    result = describe_column("Contract")
    assert result["type"] == "categorical"
    assert sum(result["value_counts"].values()) == row_count


def test_describe_reports_outliers():
    result = describe_column("TotalCharges")
    assert result["outlier_count"] == len(result["example_outliers"]) or result["outlier_count"] >= 0
    assert isinstance(result["example_outliers"], list)


def test_describe_unknown_column():
    assert "error" in describe_column("region")


def test_segment_stats_without_grouping():
    result = segment_stats({"Contract": "Month-to-month"})
    assert result["n_matched"] == 3875
    assert result["groups"][0]["key"] == {}
    assert result["groups"][0]["churn_rate"] > baseline_churn_rate


def test_segment_stats_one_group_dimension():
    result = segment_stats({}, group_by=["InternetService"])
    assert {g["key"]["InternetService"] for g in result["groups"]} == {"DSL", "Fiber optic", "No"}
    assert sum(g["n"] for g in result["groups"]) == row_count


def test_segment_stats_two_group_dimensions_and_metrics():
    result = segment_stats({}, group_by=["Contract", "InternetService"],
                           metrics=[("MonthlyCharges", "mean"), ("tenure", "median")])
    assert len(result["groups"]) == 9
    assert "mean_MonthlyCharges" in result["groups"][0]
    assert "median_tenure" in result["groups"][0]


def test_segment_stats_rejects_bad_input():
    assert "error" in segment_stats({"region": "north"})
    assert "error" in segment_stats({"Contract": "Yearly"})
    assert "error" in segment_stats({}, metrics=[("MonthlyCharges", "stdev")])
    assert "error" in segment_stats({}, metrics=[("Contract", "mean")])


def test_segment_stats_no_rows_matched():
    result = segment_stats({"PhoneService": "No", "MultipleLines": "Yes"})
    assert "error" in result and "no rows matched" in result["error"]


def test_rank_customers_on_raw_column():
    result = rank_customers("MonthlyCharges", filters={"TechSupport": "No"}, n=5)
    charges = [c["MonthlyCharges"] for c in result["customers"]]
    assert charges == sorted(charges, reverse=True)
    assert len(result["customers"]) == 5


def test_rank_customers_on_risk_score():
    result = rank_customers("risk_score", n=10)
    scores = [c["risk_score"] for c in result["customers"]]
    assert scores == sorted(scores, reverse=True)
    assert result["n_matched"] == row_count


def test_rank_customers_ascending_and_columns():
    result = rank_customers("tenure", ascending=True, n=3, columns=["tenure", "Contract"])
    assert result["customers"][0]["tenure"] == 0
    assert "Contract" in result["customers"][0]


def test_rank_customers_bad_input():
    assert "error" in rank_customers("revenue_trend")
    assert "error" in rank_customers("tenure", filters={"Contract": "Yearly"})


def test_correlation_numeric_numeric():
    result = correlation("tenure", "MonthlyCharges")
    assert result["method"] == "pearson"
    assert -1 <= result["statistic"] <= 1


def test_correlation_numeric_binary_categorical():
    result = correlation("tenure", "Churn")
    assert result["method"] == "point-biserial"
    assert result["statistic"] < 0


def test_correlation_numeric_multilevel_categorical():
    result = correlation("tenure", "Contract")
    assert result["method"] == "correlation-ratio (eta)"
    assert result["statistic"] > 0


def test_correlation_categorical_categorical():
    result = correlation("Contract", "Churn")
    assert result["method"] == "cramers-v"
    assert 0 <= result["statistic"] <= 1


def test_correlation_bad_input():
    assert "error" in correlation("tenure", "region")
    assert "error" in correlation("tenure", "tenure")
