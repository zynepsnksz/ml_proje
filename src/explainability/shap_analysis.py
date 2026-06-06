"""SHAP tabanlı model açıklama fonksiyonları."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.config import FEATURE_COLUMNS


def get_shap_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    """Pipeline içindeki Random Forest sınıflandırıcısı için TreeExplainer oluşturur.

    Args:
        pipeline: ``StandardScaler`` + ``classifier`` adımlarından oluşan eğitilmiş pipeline.

    Returns:
        SHAP TreeExplainer nesnesi.
    """
    classifier = pipeline.named_steps["classifier"]
    return shap.TreeExplainer(classifier)


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
    """Tek bir örnek için SHAP waterfall grafiği oluşturur.

    Args:
        pipeline: Eğitilmiş model pipeline'ı (scaler + classifier).
        instance_df: Tek satırlık özellik DataFrame'i.
        label_index: SHAP değerlerinin gösterileceği sınıf indeksi.
        label_encoder: Tahmin edilen sınıf adını çözmek için LabelEncoder.

    Returns:
        Streamlit'te ``st.pyplot(fig)`` ile gösterilebilecek matplotlib Figure.
    """
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
