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

from src.config import ALL_FEATURE_COLUMNS, FEATURE_COLUMNS

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
    class_names: list[str] | None = None,
) -> None:
    """Global SHAP summary plot (stacked bar chart) üretir ve kaydeder."""
    if not supports_tree_explainer(pipeline):
        return

    engineer = pipeline.named_steps.get("engineer", None)
    clipper = pipeline.named_steps.get("clipper", None)
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]

    X_sample = X_train[FEATURE_COLUMNS]
    if len(X_sample) > sample_size:
        X_sample = X_sample.sample(sample_size, random_state=42)

    if engineer is not None:
        X_processed = engineer.transform(X_sample)
        feature_names = ALL_FEATURE_COLUMNS
    else:
        X_processed = X_sample
        feature_names = FEATURE_COLUMNS

    if clipper is not None:
        X_processed = clipper.transform(X_processed)

    X_scaled = scaler.transform(X_processed)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_scaled)

    # 3D array'i list of 2D arrays'e çevirerek stacked bar chart için uyumlu hale getirelim
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    else:
        shap_values_list = shap_values

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_list,
        X_scaled,
        feature_names=feature_names,
        class_names=class_names,
        plot_type="bar",
        show=False,
        max_display=len(feature_names),
    )
    plt.title("Global SHAP Feature Importance (Stacked Bar Chart per Crop)", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _build_waterfall_explanation(
    explainer: shap.TreeExplainer,
    scaled_instance: np.ndarray,
    label_index: int,
    feature_names: list[str],
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
        feature_names=feature_names,
    )


def plot_local_waterfall(
    pipeline: Pipeline,
    instance_df: pd.DataFrame,
    label_index: int,
    label_encoder: LabelEncoder,
) -> Figure:
    """Tek bir örnek için SHAP waterfall grafiği oluşturur."""
    engineer = pipeline.named_steps.get("engineer", None)
    clipper = pipeline.named_steps.get("clipper", None)
    scaler = pipeline.named_steps["scaler"]

    if engineer is not None:
        processed = engineer.transform(instance_df[FEATURE_COLUMNS])
        feature_names = ALL_FEATURE_COLUMNS
    else:
        processed = instance_df[FEATURE_COLUMNS]
        feature_names = FEATURE_COLUMNS

    if clipper is not None:
        processed = clipper.transform(processed)

    scaled_instance = scaler.transform(processed)
        
    explainer = get_shap_explainer(pipeline)
    explanation = _build_waterfall_explanation(explainer, scaled_instance, label_index, feature_names)

    crop_name = label_encoder.inverse_transform([label_index])[0]

    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False, max_display=len(feature_names))
    fig = plt.gcf()
    fig.suptitle(f"SHAP Waterfall — {crop_name}", fontsize=14, fontweight="bold", y=1.02)

    return fig
