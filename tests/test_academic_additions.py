"""Akademik denetim maddeleri için genişletilmiş test senaryoları."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.config import ALL_FEATURE_COLUMNS, FEATURE_COLUMNS, MODEL_PATH, RANDOM_STATE
from src.data.loader import get_feature_target, load_data, split_data, validate_inputs
from src.explainability.lime_analysis import create_lime_explainer, explain_instance_lime
from src.explainability.shap_analysis import plot_summary
from src.models.predict import predict
from src.models.train import tune_hyperparameters
from src.preprocessing import FeatureEngineer, build_pipeline, encode_labels, fit_label_encoder

pytest.importorskip("xgboost")
from xgboost import XGBClassifier  # noqa: E402


def _small_train_set(n_per_class: int = 4) -> tuple[pd.DataFrame, np.ndarray, object]:
    """Hızlı testler için küçük stratified alt küme üretir."""
    df = load_data()
    X, y = get_feature_target(df)
    subset_idx = []
    for label in y.unique():
        subset_idx.extend(y[y == label].index[:n_per_class].tolist())
    X_small = X.loc[subset_idx].reset_index(drop=True)
    y_small = y.loc[subset_idx].reset_index(drop=True)
    label_encoder = fit_label_encoder(y_small)
    y_enc = np.array(encode_labels(label_encoder, y_small))
    return X_small, y_enc, label_encoder


class _CapturingRandomizedSearchCV:
    """RandomizedSearchCV yerine param_distributions yakalayan hafif mock."""

    last_param_dist: dict | None = None

    def __init__(self, estimator, param_distributions, **kwargs):
        self.estimator = estimator
        self.param_distributions = param_distributions
        type(self).last_param_dist = param_distributions
        self.best_params_ = {
            key: values[0] for key, values in param_distributions.items()
        }
        self.best_estimator_ = estimator

    def fit(self, X, y):
        return self


@pytest.mark.parametrize(
    "classifier,expected_keys",
    [
        (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators",
                "classifier__max_depth",
                "classifier__min_samples_split",
                "classifier__min_samples_leaf",
                "classifier__max_features",
            },
        ),
        (
            XGBClassifier(
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="mlogloss",
            ),
            {
                "classifier__n_estimators",
                "classifier__max_depth",
                "classifier__learning_rate",
                "classifier__subsample",
                "classifier__colsample_bytree",
                "classifier__min_child_weight",
            },
        ),
    ],
)
def test_tune_hyperparameters_param_space(classifier, expected_keys):
    """Her model tipi için doğru hiperparametre arama anahtarları kullanılmalı."""
    X_small, y_enc, _ = _small_train_set(n_per_class=3)
    pipeline = build_pipeline(classifier)

    with patch("src.models.train.RandomizedSearchCV", _CapturingRandomizedSearchCV):
        _, best_params = tune_hyperparameters(pipeline, X_small, y_enc)

    assert _CapturingRandomizedSearchCV.last_param_dist is not None
    assert set(_CapturingRandomizedSearchCV.last_param_dist.keys()) == expected_keys
    assert set(best_params.keys()).issubset(expected_keys)


def test_explainers_smoke(tmp_path):
    """SHAP summary ve LIME yerel açıklama hatasız grafik üretmeli."""
    X_small, y_enc, label_encoder = _small_train_set(n_per_class=5)
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=15, random_state=RANDOM_STATE))
    pipeline.fit(X_small, y_enc)

    shap_path = tmp_path / "shap_summary.png"
    plot_summary(
        pipeline,
        X_small,
        shap_path,
        class_names=label_encoder.classes_.tolist(),
    )
    assert shap_path.exists()

    instance_raw = X_small.iloc[[0]]
    instance_engineered = FeatureEngineer().transform(instance_raw)
    label_index = int(pipeline.predict(instance_raw)[0])

    lime_explainer = create_lime_explainer(
        FeatureEngineer().transform(X_small),
        ALL_FEATURE_COLUMNS,
        label_encoder.classes_.tolist(),
    )
    predict_fn = lambda x: pipeline[1:].predict_proba(
        pd.DataFrame(x, columns=ALL_FEATURE_COLUMNS)
    )
    lime_fig = explain_instance_lime(
        pipeline,
        lime_explainer,
        instance_engineered,
        label_index,
        predict_fn=predict_fn,
    )
    assert lime_fig is not None
    plt.close(lime_fig)
    plt.close("all")


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model henüz eğitilmemiş")
def test_regression_confusion_pairs():
    """Sınıf ortalama profilleri bilinen rice ve maize örnekleri için doğru tahmin edilmeli."""
    rice_profile = {
        "N": 79.89,
        "P": 47.58,
        "K": 39.87,
        "temperature": 23.69,
        "humidity": 82.27,
        "ph": 6.43,
        "rainfall": 236.18,
    }
    maize_profile = {
        "N": 77.76,
        "P": 48.44,
        "K": 19.79,
        "temperature": 22.39,
        "humidity": 65.09,
        "ph": 6.25,
        "rainfall": 84.77,
    }

    assert predict(rice_profile) == "rice"
    assert predict(maize_profile) == "maize"


def test_predict_validation_edge_cases():
    """Geçersiz domain girdileri validate_inputs üzerinden ValueError fırlatmalı."""
    invalid_ph = pd.DataFrame(
        [
            {
                "N": 50,
                "P": 40,
                "K": 30,
                "temperature": 25.0,
                "humidity": 80.0,
                "ph": -1.0,
                "rainfall": 200.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="pH"):
        validate_inputs(invalid_ph)

    invalid_humidity = pd.DataFrame(
        [
            {
                "N": 50,
                "P": 40,
                "K": 30,
                "temperature": 25.0,
                "humidity": 105.0,
                "ph": 6.5,
                "rainfall": 200.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="Nem"):
        validate_inputs(invalid_humidity)

    with pytest.raises(ValueError, match="pH"):
        predict(
            {
                "N": 50,
                "P": 40,
                "K": 30,
                "temperature": 25.0,
                "humidity": 80.0,
                "ph": -1.0,
                "rainfall": 200.0,
            }
        )

    with pytest.raises(ValueError, match="Nem"):
        predict(
            {
                "N": 50,
                "P": 40,
                "K": 30,
                "temperature": 25.0,
                "humidity": 105.0,
                "ph": 6.5,
                "rainfall": 200.0,
            }
        )
