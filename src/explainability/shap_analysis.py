"""SHAP tabanlı model açıklama fonksiyonları."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.config import FEATURE_COLUMNS

TREE_EXPLAINER_TYPES = (
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "DecisionTreeClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "LGBMClassifier",
)


def supports_tree_explainer(pipeline: Pipeline) -> bool:
    """Pipeline sınıflandırıcısı TreeExplainer ile uyumlu mu kontrol eder."""
    classifier = pipeline.named_steps["classifier"]
    return type(classifier).__name__ in TREE_EXPLAINER_TYPES


def get_shap_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    """Pipeline içindeki ağaç tabanlı sınıflandırıcı için TreeExplainer oluşturur."""
    if not supports_tree_explainer(pipeline):
        raise ValueError(
            "Seçilen model SHAP TreeExplainer ile uyumlu değil. "
            "KernelExplainer veya Permutation Importance kullanılmalıdır."
        )
    classifier = pipeline.named_steps["classifier"]
    return shap.TreeExplainer(classifier)


def plot_summary(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    output_path: Path,
    sample_size: int = 300,
) -> None:
    """Global SHAP summary plot (beeswarm) üretir ve kaydeder."""
    if not supports_tree_explainer(pipeline):
        return

    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    X_sample = X_train[FEATURE_COLUMNS]
    if len(X_sample) > sample_size:
        X_sample = X_sample.sample(sample_size, random_state=42)

    X_scaled = scaler.transform(X_sample)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_scaled)

    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(
            shap_values,
            X_scaled,
            feature_names=FEATURE_COLUMNS,
            show=False,
            max_display=len(FEATURE_COLUMNS),
        )
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Çok sınıflı RF: (n_samples, n_features, n_classes) → sınıflar üzerinden ortalama |SHAP|
        values_for_plot = np.abs(shap_values).mean(axis=2)
        shap.summary_plot(
            values_for_plot,
            X_scaled,
            feature_names=FEATURE_COLUMNS,
            show=False,
            max_display=len(FEATURE_COLUMNS),
        )
    else:
        shap.summary_plot(
            shap_values,
            X_scaled,
            feature_names=FEATURE_COLUMNS,
            show=False,
            max_display=len(FEATURE_COLUMNS),
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _build_waterfall_explanation(
    explainer: shap.TreeExplainer,
    scaled_instance: np.ndarray,
    label_index: int,
) -> shap.Explanation:
    """Tek örnek ve hedef sınıf için SHAP Explanation nesnesi üretir."""
    shap_values = explainer(scaled_instance)

    if isinstance(shap_values, list):
        values = shap_values[label_index][0]
        base_value = explainer.expected_value[label_index]
    elif len(shap_values.shape) == 3:
        values = shap_values.values[0, :, label_index]
        base_value = explainer.expected_value[label_index]
    else:
        values = shap_values.values[0]
        base_value = explainer.expected_value

    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=scaled_instance[0],
        feature_names=FEATURE_COLUMNS,
    )


def plot_local_waterfall(
    pipeline: Pipeline,
    instance_df: pd.DataFrame,
    label_index: int,
    label_encoder: LabelEncoder,
) -> Figure:
    """Tek bir örnek için SHAP waterfall grafiği oluşturur."""
    scaler = pipeline.named_steps["scaler"]
    scaled_instance = scaler.transform(instance_df[FEATURE_COLUMNS])
    explainer = get_shap_explainer(pipeline)
    explanation = _build_waterfall_explanation(explainer, scaled_instance, label_index)

    crop_name = label_encoder.inverse_transform([label_index])[0]

    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False, max_display=len(FEATURE_COLUMNS))
    fig = plt.gcf()
    fig.suptitle(f"SHAP Waterfall — {crop_name}", fontsize=14, fontweight="bold", y=1.02)

    return fig
