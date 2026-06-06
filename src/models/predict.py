"""Eğitilmiş model ile tahmin."""

from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, MODEL_PATH
from src.preprocessing import decode_labels


def load_model(path=MODEL_PATH) -> dict[str, Any]:
    """Joblib ile kaydedilmiş model artifact'ini yükler."""
    return joblib.load(path)


def _to_dataframe(features: dict | pd.Series | pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Girdiyi DataFrame formatına çevirir."""
    if isinstance(features, pd.DataFrame):
        return features[FEATURE_COLUMNS]
    if isinstance(features, pd.Series):
        return pd.DataFrame([features[FEATURE_COLUMNS].values], columns=FEATURE_COLUMNS)
    if isinstance(features, dict):
        return pd.DataFrame([features], columns=FEATURE_COLUMNS)
    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


def predict(features: dict | pd.Series | pd.DataFrame | np.ndarray, artifact=None) -> str:
    """Tek örnek için crop sınıfı tahmin eder."""
    artifact = artifact or load_model()
    X = _to_dataframe(features)
    y_pred = artifact["pipeline"].predict(X)[0]
    return decode_labels(artifact["label_encoder"], [y_pred])[0]


def predict_proba(
    features: dict | pd.Series | pd.DataFrame | np.ndarray, artifact=None
) -> dict[str, float]:
    """Tüm sınıflar için olasılık sözlüğü döndürür."""
    artifact = artifact or load_model()
    X = _to_dataframe(features)
    pipeline = artifact["pipeline"]
    encoder = artifact["label_encoder"]

    if not hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        raise AttributeError("Seçilen model predict_proba desteklemiyor.")

    proba = pipeline.predict_proba(X)[0]
    classes = decode_labels(encoder, range(len(proba)))
    return dict(zip(classes, proba.tolist()))


def predict_top3(
    features: dict | pd.Series | pd.DataFrame | np.ndarray, artifact=None
) -> list[tuple[str, float]]:
    """En yüksek olasılıklı 3 crop önerisini döndürür."""
    proba_dict = predict_proba(features, artifact=artifact)
    ranked = sorted(proba_dict.items(), key=lambda item: item[1], reverse=True)
    return ranked[:3]
