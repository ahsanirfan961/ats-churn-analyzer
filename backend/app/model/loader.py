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


def _build_feature_map() -> dict[str, tuple[str, object]]:
    encoder = pipeline.named_steps["pre"].named_transformers_["cat"]
    source_pairs = []
    for index, column in enumerate(categorical_cols):
        dropped = None if encoder.drop_idx_ is None else encoder.drop_idx_[index]
        for position, value in enumerate(encoder.categories_[index]):
            if dropped is None or position != dropped:
                source_pairs.append((column, value))
    encoded = dict(zip(encoder.get_feature_names_out(categorical_cols), source_pairs))
    return {**encoded, **{column: (column, None) for column in numeric_cols}}


feature_map = _build_feature_map()
