"""LIME tabanlı local model açıklama fonksiyonları."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, RANDOM_STATE


def create_lime_explainer(
    X_train: pd.DataFrame,
    feature_names: list[str],
    class_names: list[str],
) -> LimeTabularExplainer:
    """Ham eğitim verisi üzerinde LIME Tabular explainer oluşturur.

    Args:
        X_train: Ölçeklenmemiş eğitim özellikleri.
        feature_names: Özellik sütun adları.
        class_names: Sınıf/mahsul isimleri.

    Returns:
        Yapılandırılmış ``LimeTabularExplainer`` nesnesi.
    """
    return lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        mode="classification",
        feature_names=feature_names,
        class_names=class_names,
        verbose=False,
        random_state=RANDOM_STATE,
    )


def explain_instance_lime(
    pipeline: Pipeline,
    explainer: LimeTabularExplainer,
    instance_df: pd.DataFrame,
    label_index: int,
    predict_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Figure:
    """Tek bir örnek için LIME açıklama grafiği oluşturur.

    Args:
        pipeline: Eğitilmiş model pipeline'ı (scaler + classifier).
        explainer: ``create_lime_explainer`` ile oluşturulmuş LIME explainer.
        instance_df: Tek satırlık özellik DataFrame'i.
        label_index: Açıklanacak tahmin edilen sınıf indeksi.
        predict_fn: Opsiyonel olasılık tahmin fonksiyonu. Verilmezse
            ``pipeline.predict_proba`` kullanılır.

    Returns:
        Streamlit'te ``st.pyplot(fig)`` ile gösterilebilecek matplotlib Figure.
    """
    instance = instance_df[FEATURE_COLUMNS].iloc[0].values
    predict_fn = predict_fn or pipeline.predict_proba

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        labels=[label_index],
        num_features=len(FEATURE_COLUMNS),
    )

    crop_name = explainer.class_names[label_index]

    fig = explanation.as_pyplot_figure(label=label_index)
    fig.axes[0].set_title(
        f"LIME Explanation — {crop_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig
