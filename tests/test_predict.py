"""Tahmin modülü smoke testleri."""

from pathlib import Path

import pytest

from src.config import MODEL_PATH
from src.models.predict import predict, predict_top3


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
