"""Prediction Confidence Analysis — güven skoru, seviye ve margin hesaplamaları."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from src.config import (
    CALIBRATION_CURVE_PATH,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_LEVEL_HIGH,
    CONFIDENCE_LEVEL_LABELS,
    CONFIDENCE_LEVEL_LOW,
    CONFIDENCE_LEVEL_MEDIUM,
    CONFIDENCE_MEDIUM_THRESHOLD,
)


class TopPrediction(TypedDict):
    """Tek bir top-N tahmin girdisi."""

    crop: str
    probability: float


class PredictionResult(TypedDict):
    """Güven analizi ile zenginleştirilmiş tahmin sonucu."""

    prediction: str
    confidence_score: float
    confidence_level: str
    probability_margin: float
    top_predictions: list[TopPrediction]


def confidence_score_from_proba(proba: np.ndarray) -> float:
    """En yüksek sınıf olasılığını 0-100 yüzde skoruna çevirir."""
    return round(float(np.max(proba)) * 100.0, 1)


def probability_margin_from_proba(proba: np.ndarray) -> float:
    """Top-1 ve top-2 olasılık farkını yüzde puanı olarak hesaplar."""
    sorted_proba = np.sort(proba)[::-1]
    if len(sorted_proba) < 2:
        return round(float(sorted_proba[0]) * 100.0, 1)
    margin = (sorted_proba[0] - sorted_proba[1]) * 100.0
    return round(float(margin), 1)


def classify_confidence_level(confidence_score: float) -> str:
    """Güven skorunu High / Medium / Low seviyesine sınıflandırır."""
    if confidence_score >= CONFIDENCE_HIGH_THRESHOLD:
        return CONFIDENCE_LEVEL_HIGH
    if confidence_score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return CONFIDENCE_LEVEL_MEDIUM
    return CONFIDENCE_LEVEL_LOW


def analyze_confidence_from_proba(proba: np.ndarray) -> dict[str, float | str]:
    """Tek örnek olasılık vektöründen güven metriklerini üretir."""
    score = confidence_score_from_proba(proba)
    return {
        "confidence_score": score,
        "confidence_level": classify_confidence_level(score),
        "probability_margin": probability_margin_from_proba(proba),
    }


def build_prediction_result(
    prediction: str,
    ranked_predictions: list[tuple[str, float]],
    proba: np.ndarray,
) -> PredictionResult:
    """Tahmin, top-N listesi ve olasılık vektöründen tam sonuç yapısı oluşturur."""
    confidence = analyze_confidence_from_proba(proba)
    top_predictions: list[TopPrediction] = [
        {"crop": crop, "probability": float(prob)} for crop, prob in ranked_predictions
    ]
    return PredictionResult(
        prediction=prediction,
        confidence_score=float(confidence["confidence_score"]),
        confidence_level=str(confidence["confidence_level"]),
        probability_margin=float(confidence["probability_margin"]),
        top_predictions=top_predictions,
    )


def summarize_test_confidence(y_proba: np.ndarray) -> dict[str, float]:
    """Test seti üzerinde güven skoru özet istatistiklerini hesaplar."""
    scores = np.max(y_proba, axis=1) * 100.0
    sorted_proba = np.sort(y_proba, axis=1)
    margins = (sorted_proba[:, -1] - sorted_proba[:, -2]) * 100.0

    levels = [classify_confidence_level(float(s)) for s in scores]
    n = len(levels)

    return {
        "mean_confidence": round(float(np.mean(scores)), 2),
        "median_confidence": round(float(np.median(scores)), 2),
        "min_confidence": round(float(np.min(scores)), 2),
        "max_confidence": round(float(np.max(scores)), 2),
        "mean_probability_margin": round(float(np.mean(margins)), 2),
        "high_confidence_rate": round(sum(1 for level in levels if level == CONFIDENCE_LEVEL_HIGH) / n, 4),
        "medium_confidence_rate": round(
            sum(1 for level in levels if level == CONFIDENCE_LEVEL_MEDIUM) / n, 4
        ),
        "low_confidence_rate": round(sum(1 for level in levels if level == CONFIDENCE_LEVEL_LOW) / n, 4),
    }


def _interpret_calibration(mean_predicted: np.ndarray, fraction_positive: np.ndarray) -> dict[str, str]:
    """Kalibrasyon eğrisinden kısa akademik yorum üretir."""
    gap = float(np.mean(fraction_positive - mean_predicted))
    mae = float(np.mean(np.abs(fraction_positive - mean_predicted)))

    if mae < 0.05:
        reliability = "Model olasılıkları genel olarak iyi kalibre edilmiş görünüyor."
        bias = "neither"
    elif gap < -0.05:
        reliability = "Model olasılıkları gerçek doğruluk oranından yüksek; aşırı güvenli (overconfident) eğilim var."
        bias = "overconfident"
    elif gap > 0.05:
        reliability = "Model olasılıkları gerçek doğruluk oranından düşük; eksik güvenli (underconfident) eğilim var."
        bias = "underconfident"
    else:
        reliability = "Model olasılıkları makul düzeyde kalibre; hafif sapmalar gözlenebilir."
        bias = "slight_miscalibration"

    return {
        "reliability_summary": reliability,
        "calibration_bias": bias,
        "mean_calibration_gap": round(gap, 4),
        "mean_absolute_calibration_error": round(mae, 4),
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path | str = CALIBRATION_CURVE_PATH,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Top-1 güven skorları için reliability diagram üretir ve analiz döndürür."""
    y_pred = np.argmax(y_proba, axis=1)
    max_proba = np.max(y_proba, axis=1)
    correct = (y_pred == y_true).astype(int)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        correct,
        max_proba,
        n_bins=n_bins,
        strategy="uniform",
    )

    interpretation = _interpret_calibration(mean_predicted_value, fraction_of_positives)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        "o-",
        color="#2980b9",
        linewidth=2,
        markersize=8,
        label="Model",
    )
    ax.set_xlabel("Mean Predicted Probability (Confidence)")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title("Calibration Curve — Top-1 Prediction Confidence")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return {
        "n_bins": n_bins,
        "mean_predicted_value": mean_predicted_value.tolist(),
        "fraction_of_positives": fraction_of_positives.tolist(),
        **interpretation,
    }


def get_confidence_user_message(confidence_level: str) -> tuple[str, str]:
    """Streamlit için (mesaj_tipi, kullanıcı_metni) döndürür.

    mesaj_tipi: 'success' | 'warning' | 'error'
    """
    messages: dict[str, tuple[str, str]] = {
        CONFIDENCE_LEVEL_HIGH: (
            "success",
            "Model bu tahminden oldukça emin. Önerilen mahsul güçlü olasılık desteğine sahip; "
            "yine de saha koşullarınızı ve yerel uzman görüşünü dikkate almanız önerilir.",
        ),
        CONFIDENCE_LEVEL_MEDIUM: (
            "warning",
            "Tahmin makul düzeyde güvenilir ancak belirsizlik mevcut. Alternatif mahsulleri "
            "değerlendirmeniz ve ölçüm değerlerinizi doğrulamanız faydalı olabilir.",
        ),
        CONFIDENCE_LEVEL_LOW: (
            "error",
            "Tahmin belirsiz — model karar vermekte zorlanıyor. Ek toprak/iklim ölçümleri "
            "yapmanız ve bir tarım uzmanına danışmanız şiddetle önerilir.",
        ),
    }
    return messages.get(
        confidence_level,
        (
            "warning",
            "Güven seviyesi belirlenemedi. Lütfen girdi değerlerinizi kontrol edin.",
        ),
    )


def get_confidence_level_label(confidence_level: str) -> str:
    """Kısa seviye kodunu kullanıcı dostu etikete çevirir."""
    return CONFIDENCE_LEVEL_LABELS.get(confidence_level, confidence_level)
