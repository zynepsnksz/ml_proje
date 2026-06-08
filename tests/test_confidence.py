"""Prediction Confidence Analysis testleri."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import (
    CONFIDENCE_LEVEL_HIGH,
    CONFIDENCE_LEVEL_LOW,
    CONFIDENCE_LEVEL_MEDIUM,
)
from src.models.confidence import (
    analyze_confidence_from_proba,
    build_prediction_result,
    classify_confidence_level,
    confidence_score_from_proba,
    get_confidence_level_label,
    probability_margin_from_proba,
    summarize_test_confidence,
)
from src.models.predict import predict_with_confidence


def test_confidence_score_from_proba():
    """En yüksek olasılık 0-100 yüzde skoruna doğru çevrilmeli."""
    proba = np.array([0.942, 0.031, 0.018, 0.009])
    assert confidence_score_from_proba(proba) == 94.2


def test_probability_margin_from_proba():
    """Top-1 ve top-2 farkı yüzde puanı olarak hesaplanmalı."""
    proba = np.array([0.55, 0.52, 0.01, 0.02])
    assert probability_margin_from_proba(proba) == pytest.approx(3.0, abs=0.1)

    proba_clear = np.array([0.942, 0.031, 0.018, 0.009])
    assert probability_margin_from_proba(proba_clear) == pytest.approx(91.1, abs=0.1)


@pytest.mark.parametrize(
    "score,expected_level",
    [
        (90.0, CONFIDENCE_LEVEL_HIGH),
        (94.2, CONFIDENCE_LEVEL_HIGH),
        (70.0, CONFIDENCE_LEVEL_MEDIUM),
        (85.0, CONFIDENCE_LEVEL_MEDIUM),
        (69.9, CONFIDENCE_LEVEL_LOW),
        (45.0, CONFIDENCE_LEVEL_LOW),
    ],
)
def test_classify_confidence_level_thresholds(score: float, expected_level: str):
    """Güven seviyesi eşikleri config ile uyumlu olmalı."""
    assert classify_confidence_level(score) == expected_level


def test_analyze_confidence_from_proba_integration():
    """Tek örnek analizi tüm güven alanlarını üretmeli."""
    proba = np.array([0.942, 0.031, 0.018, 0.009])
    result = analyze_confidence_from_proba(proba)
    assert result["confidence_score"] == 94.2
    assert result["confidence_level"] == CONFIDENCE_LEVEL_HIGH
    assert result["probability_margin"] == pytest.approx(91.1, abs=0.1)


def test_build_prediction_result_schema():
    """Tahmin sonucu beklenen şema alanlarını içermeli."""
    proba = np.array([0.55, 0.52, 0.02, 0.01])
    ranked = [("rice", 0.55), ("jute", 0.52), ("coffee", 0.02)]
    result = build_prediction_result("rice", ranked, proba)

    assert set(result.keys()) == {
        "prediction",
        "confidence_score",
        "confidence_level",
        "probability_margin",
        "top_predictions",
    }
    assert result["prediction"] == "rice"
    assert result["confidence_score"] == 55.0
    assert result["confidence_level"] == CONFIDENCE_LEVEL_LOW
    assert result["probability_margin"] == pytest.approx(3.0, abs=0.1)
    assert len(result["top_predictions"]) == 3
    assert result["top_predictions"][0] == {"crop": "rice", "probability": 0.55}


def test_summarize_test_confidence_batch_stats():
    """Test seti özet istatistikleri doğru hesaplanmalı."""
    y_proba = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.80, 0.15, 0.05],
            [0.60, 0.35, 0.05],
        ]
    )
    summary = summarize_test_confidence(y_proba)

    assert "mean_confidence" in summary
    assert "median_confidence" in summary
    assert "min_confidence" in summary
    assert "max_confidence" in summary
    assert "high_confidence_rate" in summary
    assert "medium_confidence_rate" in summary
    assert "low_confidence_rate" in summary
    rate_sum = (
        summary["high_confidence_rate"]
        + summary["medium_confidence_rate"]
        + summary["low_confidence_rate"]
    )
    assert rate_sum == pytest.approx(1.0, abs=0.01)
    assert summary["min_confidence"] == 60.0
    assert summary["max_confidence"] == 95.0


def test_get_confidence_level_label():
    """Seviye kodları kullanıcı dostu etiketlere çevrilmeli."""
    assert get_confidence_level_label(CONFIDENCE_LEVEL_HIGH) == "High Confidence"
    assert get_confidence_level_label(CONFIDENCE_LEVEL_MEDIUM) == "Medium Confidence"
    assert get_confidence_level_label(CONFIDENCE_LEVEL_LOW) == "Low Confidence"


def test_streamlit_confidence_ui_fields_present():
    """Streamlit uygulaması confidence alanlarını göstermeli."""
    streamlit_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    source = streamlit_path.read_text(encoding="utf-8")

    assert "Prediction Confidence" in source
    assert "Confidence Score" in source
    assert "Confidence Level" in source
    assert "Probability Margin" in source
    assert "predict_with_confidence" in source
    assert "confidence_score" in source
    assert "confidence_level" in source
    assert "probability_margin" in source


@pytest.mark.skipif(
    not Path(__file__).resolve().parent.parent.joinpath("models", "best_model.pkl").exists(),
    reason="Model henüz eğitilmemiş",
)
def test_predict_with_confidence_end_to_end():
    """Eğitilmiş model ile tam tahmin sonucu üretilmeli."""
    features = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 23.6,
        "humidity": 82.2,
        "ph": 6.5,
        "rainfall": 236.1,
    }
    result = predict_with_confidence(features)

    assert isinstance(result["prediction"], str)
    assert 0.0 <= result["confidence_score"] <= 100.0
    assert result["confidence_level"] in {
        CONFIDENCE_LEVEL_HIGH,
        CONFIDENCE_LEVEL_MEDIUM,
        CONFIDENCE_LEVEL_LOW,
    }
    assert 0.0 <= result["probability_margin"] <= 100.0
    assert len(result["top_predictions"]) == 3
