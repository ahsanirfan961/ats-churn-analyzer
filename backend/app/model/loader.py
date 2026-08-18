import joblib

from app.config import MODEL_PATH

_artifact = joblib.load(MODEL_PATH)

pipeline = _artifact["pipeline"]
feature_cols = list(_artifact["feature_cols"])
all_feature_names = list(_artifact["all_feature_names"])
default_threshold = _artifact["default_threshold"]

_column_groups = {name: cols for name, _, cols in pipeline.named_steps["pre"].transformers_}
categorical_cols = list(_column_groups["cat"])
numeric_cols = list(_column_groups["num"])
