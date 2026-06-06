"""Crop Recommendation — Streamlit karar destek ve XAI arayüzü."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLUMNS, MODEL_PATH
from src.data.loader import get_feature_target, load_data, split_data
from src.explainability.lime_analysis import create_lime_explainer, explain_instance_lime
from src.explainability.shap_analysis import plot_local_waterfall
from src.models.predict import predict_top3

SCENARIOS = {
    "rice": {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 23.6,
        "humidity": 82.2,
        "ph": 6.5,
        "rainfall": 236.1,
    },
    "apple": {
        "N": 20,
        "P": 125,
        "K": 200,
        "temperature": 22.3,
        "humidity": 92.3,
        "ph": 5.9,
        "rainfall": 112.9,
    },
    "cotton": {
        "N": 120,
        "P": 40,
        "K": 20,
        "temperature": 23.9,
        "humidity": 79.8,
        "ph": 6.9,
        "rainfall": 80.7,
    },
}

SLIDER_DEFAULTS = {
    "N": 50,
    "P": 50,
    "K": 50,
    "temperature": 25.0,
    "humidity": 60.0,
    "ph": 6.5,
    "rainfall": 100.0,
}


def _init_session_state() -> None:
    """Slider ve senaryo durumunu başlatır."""
    for key, value in SLIDER_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _apply_scenario(scenario: dict[str, float]) -> None:
    """Örnek senaryo değerlerini session state'e yazar."""
    for key, value in scenario.items():
        st.session_state[key] = value


@st.cache_resource
def load_artifact() -> dict:
    """Model artifact'ini bir kez yükler."""
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_lime_explainer(_artifact: dict) -> object:
    """Eğitim seti üzerinde LIME explainer oluşturur."""
    df = load_data()
    X, y = get_feature_target(df)
    X_train, _, _, _ = split_data(X, y)
    label_encoder = _artifact["label_encoder"]
    return create_lime_explainer(
        X_train[FEATURE_COLUMNS],
        FEATURE_COLUMNS,
        label_encoder.classes_.tolist(),
    )


def _build_instance_df() -> pd.DataFrame:
    """Sidebar slider değerlerinden tek satırlık DataFrame oluşturur."""
    return pd.DataFrame(
        [
            {
                "N": st.session_state["N"],
                "P": st.session_state["P"],
                "K": st.session_state["K"],
                "temperature": st.session_state["temperature"],
                "humidity": st.session_state["humidity"],
                "ph": st.session_state["ph"],
                "rainfall": st.session_state["rainfall"],
            }
        ]
    )


def _plot_top3_bar(top3: list[tuple[str, float]]) -> plt.Figure:
    """Top-3 olasılıkları için yatay bar grafiği oluşturur."""
    crops = [item[0] for item in top3]
    probs = [item[1] * 100 for item in top3]
    colors = ["#2ecc71", "#3498db", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(crops[::-1], probs[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Güven Skoru (%)")
    ax.set_title("Top-3 Mahsul Olasılık Dağılımı", fontweight="bold")
    for bar, prob in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{prob:.1f}%", va="center")
    plt.tight_layout()
    return fig


def main() -> None:
    """Streamlit uygulamasını çalıştırır."""
    st.set_page_config(page_title="Akıllı Tarım & XAI Karar Destek", layout="wide")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #f0fff4 0%, #e8f8ef 100%);
            border: 1px solid #b7e4c7;
            padding: 1rem;
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌱 Akıllı Tarım Karar Destek ve Mahsul Öneri Sistemi")
    st.markdown(
        "Toprak bileşenleri ve iklim verilerine göre 22 farklı ürün arasından "
        "en uygun mahsulleri öneren ve yapay zeka kararlarını (SHAP/LIME) açıklayan kontrol paneli."
    )

    _init_session_state()

    with st.sidebar:
        st.header("Saha Ölçüm Değerleri")

        st.slider("Azot (N)", min_value=0, max_value=150, key="N")
        st.slider("Fosfor (P)", min_value=5, max_value=150, key="P")
        st.slider("Potasyum (K)", min_value=5, max_value=210, key="K")
        st.slider("Sıcaklık (Temp °C)", min_value=8.0, max_value=45.0, step=0.1, key="temperature")
        st.slider("Nem (%)", min_value=10.0, max_value=100.0, step=0.1, key="humidity")
        st.slider("pH (Toprak Asitliği)", min_value=3.5, max_value=9.9, step=0.1, key="ph")
        st.slider("Yağış (Rainfall mm)", min_value=20.0, max_value=300.0, step=0.1, key="rainfall")

        st.markdown("---")
        st.subheader("Hızlı Test Senaryoları")
        if st.button("🌾 Rice (Çeltik) Senaryosu", use_container_width=True):
            _apply_scenario(SCENARIOS["rice"])
            st.rerun()
        if st.button("🍎 Apple (Elma) Senaryosu", use_container_width=True):
            _apply_scenario(SCENARIOS["apple"])
            st.rerun()
        if st.button("🧵 Cotton (Pamuk) Senaryosu", use_container_width=True):
            _apply_scenario(SCENARIOS["cotton"])
            st.rerun()

    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    lime_explainer = load_lime_explainer(artifact)

    instance_df = _build_instance_df()
    top3 = predict_top3(instance_df, artifact=artifact)
    label_index = int(pipeline.predict(instance_df)[0])

    best_crop, best_score = top3[0]

    st.subheader("📊 Tahmin Sonuçları")

    # En uygun mahsulü ve alternatifleri aynı boyutta göstermek için üç metrik kullanıyoruz
    col_best, col2, col3 = st.columns(3)
    with col_best:
        st.metric("En Uygun Mahsul", best_crop.title(), f"{best_score * 100:.2f}%")
    with col2:
        st.metric("2. Alternatif", top3[1][0].title(), f"{top3[1][1] * 100:.2f}%")
    with col3:
        st.metric("3. Alternatif", top3[2][0].title(), f"{top3[2][1] * 100:.2f}%")

    bar_fig = _plot_top3_bar(top3)
    st.pyplot(bar_fig)
    plt.close(bar_fig)

    st.markdown("---")
    st.subheader("🔍 Yapay Zeka Karar Açıklamaları (XAI)")

    tab_shap, tab_lime = st.tabs(["SHAP Waterfall Açıklaması", "LIME Yerel Katkı Açıklaması"])

    with tab_shap:
        st.markdown(
            "SHAP, modelin tahminine her özelliğin ne kadar katkı sağladığını gösterir."
        )
        shap_fig = plot_local_waterfall(pipeline, instance_df, label_index, label_encoder)
        st.pyplot(shap_fig)
        plt.close(shap_fig)

    with tab_lime:
        st.markdown(
            "LIME, yerel bir yaklaşımla bu örnek için en etkili özellikleri listeler."
        )
        predict_fn = lambda x: pipeline.predict_proba(pd.DataFrame(x, columns=FEATURE_COLUMNS))
        lime_fig = explain_instance_lime(
            pipeline,
            lime_explainer,
            instance_df,
            label_index,
            predict_fn=predict_fn,
        )
        st.pyplot(lime_fig)
        plt.close(lime_fig)


if __name__ == "__main__":
    main()
