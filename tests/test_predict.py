"""Tahmin modülü smoke testleri."""

from pathlib import Path

import dill
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.config import FEATURE_COLUMNS, MODEL_PATH, RANDOM_STATE
from src.data.loader import get_feature_target, load_data
from src.models.predict import predict, predict_top3
from src.preprocessing import build_pipeline, encode_labels, fit_label_encoder


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model henüz eğitilmemiş")
def test_predict_rice_scenario():
    features = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 23.6,
        "humidity": 82.2,
        "ph": 6.5,
        "rainfall": 236.1,
    }
    result = predict(features)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model henüz eğitilmemiş")
def test_predict_top3_returns_three():
    features = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 23.6,
        "humidity": 82.2,
        "ph": 6.5,
        "rainfall": 236.1,
    }
    top3 = predict_top3(features)
    assert len(top3) == 3
    assert all(isinstance(item[1], float) for item in top3)


def test_dill_artifact_roundtrip(tmp_path):
    """Dill ile kaydedilen pipeline+encoder artifact'i yüklendikten sonra tahmin üretmeli."""
    df = load_data()
    X, y = get_feature_target(df)
    X_small = X.iloc[:60].reset_index(drop=True)
    y_small = y.iloc[:60].reset_index(drop=True)

    label_encoder = fit_label_encoder(y_small)
    y_enc = np.array(encode_labels(label_encoder, y_small))

    pipeline = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE))
    pipeline.fit(X_small, y_enc)

    artifact = {"pipeline": pipeline, "label_encoder": label_encoder}
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        dill.dump(artifact, f)

    with open(model_path, "rb") as f:
        loaded = dill.load(f)

    features = X_small.iloc[0][FEATURE_COLUMNS].to_dict()
    result = predict(features, artifact=loaded)

    assert isinstance(result, str)
    assert result in label_encoder.classes_
    assert loaded["pipeline"].predict(X_small.iloc[[0]])[0] == pipeline.predict(X_small.iloc[[0]])[0]
