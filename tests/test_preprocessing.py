"""Preprocessing pipeline testleri."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from src.config import FEATURE_COLUMNS
from src.preprocessing import OutlierClipper, build_pipeline, encode_labels, fit_label_encoder


def test_pipeline_fit_predict():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((50, 7)), columns=FEATURE_COLUMNS)
    y = rng.integers(0, 3, size=50)

    pipeline = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipeline.fit(X, y)
    preds = pipeline.predict(X[:5])
    assert len(preds) == 5


def test_outlier_clipper_iqr_bounds():
    """OutlierClipper, EDA ile aynı 1.5×IQR sınırlarına göre kırpar."""
    train = pd.DataFrame({"x": [0.0, 10.0, 20.0, 30.0, 100.0]})
    clipper = OutlierClipper(iqr_multiplier=1.5)
    clipper.fit(train)

    # Q1=10, Q3=30, IQR=20 → alt=10-30=-20, üst=30+30=60
    assert clipper.lower_bounds_["x"] == -20.0
    assert clipper.upper_bounds_["x"] == 60.0

    clipped = clipper.transform(pd.DataFrame({"x": [-50.0, 50.0, 100.0]}))
    assert clipped["x"].tolist() == [-20.0, 50.0, 60.0]


def test_pipeline_includes_clipper_step():
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    assert "clipper" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["clipper"], OutlierClipper)


def test_pipeline_includes_imputer_and_imputes():
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    assert "imputer" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)

    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.random((10, 7)), columns=FEATURE_COLUMNS)
    y_train = rng.integers(0, 2, size=10)
    pipeline.fit(X_train, y_train)

    X_test = pd.DataFrame([[np.nan, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], columns=FEATURE_COLUMNS)
    X_trans = pipeline[:-1].transform(X_test)
    assert not np.isnan(X_trans).any()


def test_label_encoder_roundtrip():
    labels = ["rice", "apple", "cotton"]
    encoder = fit_label_encoder(labels)
    encoded = encode_labels(encoder, labels)
    # LabelEncoder alfabetik sıralar: apple=0, cotton=1, rice=2
    assert encoded == [2, 0, 1]
