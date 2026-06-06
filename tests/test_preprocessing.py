"""Preprocessing pipeline testleri."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import FEATURE_COLUMNS
from src.preprocessing import build_pipeline, encode_labels, fit_label_encoder


def test_pipeline_fit_predict():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((50, 7)), columns=FEATURE_COLUMNS)
    y = rng.integers(0, 3, size=50)

    pipeline = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipeline.fit(X, y)
    preds = pipeline.predict(X[:5])
    assert len(preds) == 5


def test_label_encoder_roundtrip():
    labels = ["rice", "apple", "cotton"]
    encoder = fit_label_encoder(labels)
    encoded = encode_labels(encoder, labels)
    # LabelEncoder alfabetik sıralar: apple=0, cotton=1, rice=2
    assert encoded == [2, 0, 1]
