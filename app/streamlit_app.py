"""Crop Recommendation — Streamlit karar destek ve XAI arayüzü."""

from __future__ import annotations

import sys
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALL_FEATURE_COLUMNS, FEATURE_COLUMNS, MODEL_PATH
from src.preprocessing import FeatureEngineer
from src.explainability.shap_analysis import supports_tree_explainer
from src.explainability.lime_analysis import explain_instance_lime
from src.explainability.shap_analysis import plot_local_waterfall
from src.models.confidence import get_confidence_level_label, get_confidence_user_message
from src.models.predict import predict_with_confidence


def _on_scenario_change(scenarios: dict[str, dict[str, float]]) -> None:
    """Seçilen senaryo değerlerini slider durumlarına uygular."""
    selected = st.session_state["selected_scenario"]
    if selected != "--- Seçin ---":
        crop_key = selected.lower()
        _apply_scenario(scenarios[crop_key])

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
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {MODEL_PATH}\n"
            "Önce `python -m src.models.train` komutunu çalıştırın."
        )
    with open(MODEL_PATH, "rb") as f:
        return dill.load(f)


@st.cache_resource
def load_lime_explainer(_artifact: dict) -> object:
    """Model artifact'inden LIME explainer nesnesini döndürür."""
    return _artifact["lime_explainer"]


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
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        
        .block-container { 
            padding-top: 1.5rem; 
            padding-bottom: 2rem;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #f0fff4 0%, #e8f8ef 100%);
            border: 1px solid #b7e4c7;
            padding: 1rem;
            border-radius: 12px;
        }
        
        /* İdeal Değer Kartları Tasarımı */
        .ideal-values-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 15px;
            margin-bottom: 25px;
        }
        
        .ideal-val-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            flex-direction: column;
            border-top: 4px solid #cbd5e1;
        }
        
        .ideal-val-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
        }
        
        .npk-card {
            border-top: 4px solid #10b981; /* Canlı yeşil accent */
        }
        .npk-card:hover {
            border-color: #059669;
            box-shadow: 0 12px 20px -3px rgba(16, 185, 129, 0.12), 0 4px 6px -2px rgba(16, 185, 129, 0.04);
        }
        
        .env-card {
            border-top: 4px solid #3b82f6; /* Canlı mavi accent */
        }
        .env-card:hover {
            border-color: #2563eb;
            box-shadow: 0 12px 20px -3px rgba(59, 130, 246, 0.12), 0 4px 6px -2px rgba(59, 130, 246, 0.04);
        }
        
        .card-icon {
            font-size: 1.75rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        
        .card-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 6px;
        }
        
        .card-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 8px;
            line-height: 1.2;
        }
        
        .card-desc {
            font-size: 0.75rem;
            color: #64748b;
            line-height: 1.4;
            margin-top: auto;
        }

        .confidence-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        }
        .confidence-card h4 {
            margin: 0 0 1rem 0;
            color: #0f172a;
            font-size: 1.1rem;
        }
        .confidence-metric {
            margin-bottom: 0.75rem;
        }
        .confidence-metric-label {
            font-size: 0.8rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }
        .confidence-metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌱 Akıllı Tarım Karar Destek ve Mahsul Öneri Sistemi")
    st.markdown(
        "Toprak bileşenleri ve iklim verilerine göre 22 farklı ürün arasından "
        "en uygun mahsulleri öneren; **Confidence Analysis** ile modelin kararına ne kadar "
        "güvendiğini ve **SHAP/LIME** ile neden bu kararı verdiğini açıklayan kontrol paneli."
    )

    _init_session_state()

    try:
        artifact = load_artifact()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    bounds = artifact.get("train_feature_bounds")
    scenarios = artifact.get("train_crop_scenarios")
    if bounds is None or scenarios is None:
        st.error(
            "Model dosyası güncel değil (train_feature_bounds / train_crop_scenarios eksik). "
            "Lütfen `python -m src.models.train` komutunu çalıştırın."
        )
        st.stop()

    with st.sidebar:
        st.header("Saha Ölçüm Değerleri")

        st.slider("Azot (N)", min_value=int(bounds["N"][0]), max_value=int(bounds["N"][1]), key="N")
        st.slider("Fosfor (P)", min_value=int(bounds["P"][0]), max_value=int(bounds["P"][1]), key="P")
        st.slider("Potasyum (K)", min_value=int(bounds["K"][0]), max_value=int(bounds["K"][1]), key="K")
        st.slider(
            "Sıcaklık (Temp °C)",
            min_value=float(bounds["temperature"][0]),
            max_value=float(bounds["temperature"][1]),
            step=0.1,
            key="temperature",
        )
        st.slider(
            "Nem (%)",
            min_value=float(bounds["humidity"][0]),
            max_value=float(bounds["humidity"][1]),
            step=0.1,
            key="humidity",
        )
        st.slider(
            "pH (Toprak Asitliği)",
            min_value=float(bounds["ph"][0]),
            max_value=float(bounds["ph"][1]),
            step=0.1,
            key="ph",
        )
        st.slider(
            "Yağış (Rainfall mm)",
            min_value=float(bounds["rainfall"][0]),
            max_value=float(bounds["rainfall"][1]),
            step=0.1,
            key="rainfall",
        )

        st.markdown("---")
        st.subheader("Bitki Referans Şablonları")
        crop_list = sorted(list(scenarios.keys()))
        
        st.selectbox(
            "Saha Değerlerini Bitkinin İdeal Ortalama Koşullarına Ayarla:",
            ["--- Seçin ---"] + [crop.title() for crop in crop_list],
            key="selected_scenario",
            on_change=_on_scenario_change,
            args=(scenarios,),
        )

    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    lime_explainer = load_lime_explainer(artifact)

    instance_df = _build_instance_df()
    try:
        prediction_result = predict_with_confidence(instance_df, artifact=artifact)
        label_index = int(pipeline.predict(instance_df)[0])
    except ValueError as e:
        st.error(f"⚠️ Girdi Doğrulama Hatası: {str(e)}")
        st.stop()

    top3 = [
        (item["crop"], item["probability"]) for item in prediction_result["top_predictions"]
    ]
    best_crop = prediction_result["prediction"]
    best_score = top3[0][1]
    confidence_score = prediction_result["confidence_score"]
    confidence_level = prediction_result["confidence_level"]
    probability_margin = prediction_result["probability_margin"]

    st.subheader("📊 Tahmin Sonuçları")

    # En uygun mahsulü ve alternatifleri aynı boyutta göstermek için üç metrik kullanıyoruz
    col_best, col2, col3 = st.columns(3)
    with col_best:
        st.metric("En Uygun Mahsul", best_crop.title(), f"{best_score * 100:.2f}%")
    with col2:
        st.metric("2. Alternatif", top3[1][0].title(), f"{top3[1][1] * 100:.2f}%")
    with col3:
        st.metric("3. Alternatif", top3[2][0].title(), f"{top3[2][1] * 100:.2f}%")

    st.markdown("#### Prediction Confidence")
    st.caption(
        "SHAP/LIME → model **neden** bu kararı verdi? · "
        "Confidence Analysis → model bu karardan **ne kadar emin**?"
    )
    conf_col1, conf_col2, conf_col3 = st.columns(3)
    with conf_col1:
        st.markdown(
            f"""
            <div class="confidence-card">
                <div class="confidence-metric">
                    <div class="confidence-metric-label">Confidence Score</div>
                    <div class="confidence-metric-value">{confidence_score}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with conf_col2:
        level_label = get_confidence_level_label(confidence_level)
        st.markdown(
            f"""
            <div class="confidence-card">
                <div class="confidence-metric">
                    <div class="confidence-metric-label">Confidence Level</div>
                    <div class="confidence-metric-value">{level_label}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with conf_col3:
        st.markdown(
            f"""
            <div class="confidence-card">
                <div class="confidence-metric">
                    <div class="confidence-metric-label">Probability Margin</div>
                    <div class="confidence-metric-value">{probability_margin}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    msg_type, msg_text = get_confidence_user_message(confidence_level)
    if msg_type == "success":
        st.success(msg_text)
    elif msg_type == "warning":
        st.warning(msg_text)
    else:
        st.error(msg_text)

    bar_fig = _plot_top3_bar(top3)
    st.pyplot(bar_fig)
    plt.close(bar_fig)

    # Seçilen senaryonun ideal referans değerlerini kutular halinde göster
    selected = st.session_state.get("selected_scenario", "--- Seçin ---")
    if selected != "--- Seçin ---":
        st.markdown("---")
        st.subheader(f"🌾 {selected} İdeal Yetişme Koşulları (Veri Seti Referans Değerleri)")
        st.markdown(
            "Seçtiğiniz bitkinin veri setindeki ideal (ortalama) koşulları aşağıdaki bilgi kartlarında gösterilmektedir. "
            "Sol taraftaki panelden kendi ölçümlerinizi girerek ideal değerlerle karşılaştırabilir ve model tahminini gözlemleyebilirsiniz:"
        )
        crop_key = selected.lower()
        ideal_vals = scenarios[crop_key]
        
        html_content = f"""
        <div class="ideal-values-grid">
            <div class="ideal-val-card npk-card">
                <div class="card-icon">🧪</div>
                <div class="card-title">İdeal Azot (N)</div>
                <div class="card-value">{ideal_vals['N']:.1f}</div>
                <div class="card-desc">Yaprak gelişimi ve klorofil sentezi için ana besin elementidir.</div>
            </div>
            <div class="ideal-val-card npk-card">
                <div class="card-icon">🌾</div>
                <div class="card-title">İdeal Fosfor (P)</div>
                <div class="card-value">{ideal_vals['P']:.1f}</div>
                <div class="card-desc">Kök gelişimi, çiçeklenme ve enerji metabolizması için kritiktir.</div>
            </div>
            <div class="ideal-val-card npk-card">
                <div class="card-icon">🛡️</div>
                <div class="card-title">İdeal Potasyum (K)</div>
                <div class="card-value">{ideal_vals['K']:.1f}</div>
                <div class="card-desc">Su dengesini, ozmotik basıncı ve hastalıklara karşı dayanıklılığı yönetir.</div>
            </div>
            <div class="ideal-val-card env-card">
                <div class="card-icon">🌡️</div>
                <div class="card-title">İdeal Sıcaklık</div>
                <div class="card-value">{ideal_vals['temperature']:.1f} °C</div>
                <div class="card-desc">Enzimatik reaksiyonlar ve fotosentez verimi için en uygun çevre sıcaklığıdır.</div>
            </div>
            <div class="ideal-val-card env-card">
                <div class="card-icon">💧</div>
                <div class="card-title">İdeal Nem</div>
                <div class="card-value">% {ideal_vals['humidity']:.1f}</div>
                <div class="card-desc">Havadaki nem oranı; bitkinin su kaybetme (terleme) hızını belirler.</div>
            </div>
            <div class="ideal-val-card env-card">
                <div class="card-icon">📈</div>
                <div class="card-title">İdeal pH</div>
                <div class="card-value">{ideal_vals['ph']:.1f}</div>
                <div class="card-desc">Toprak asitliği; bitki köklerinin besinleri emebilme kapasitesini kontrol eder.</div>
            </div>
            <div class="ideal-val-card env-card">
                <div class="card-icon">🌧️</div>
                <div class="card-title">İdeal Yağış</div>
                <div class="card-value">{ideal_vals['rainfall']:.1f} mm</div>
                <div class="card-desc">Bitkinin mevsimsel/yıllık olarak ihtiyaç duyduğu ideal su kaynağı miktarıdır.</div>
            </div>
        </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔍 Yapay Zeka Karar Açıklamaları (XAI)")
    st.info(
        "**SHAP / LIME** — Modelin bu tahmine hangi özelliklerle ulaştığını açıklar (neden?). "
        "Yukarıdaki **Confidence Analysis** kartı ise modelin kendi tahminine ne kadar güvendiğini gösterir (ne kadar emin?)."
    )

    tab_shap, tab_lime = st.tabs(["SHAP Waterfall Açıklaması", "LIME Yerel Katkı Açıklaması"])

    with tab_shap:
        st.markdown(
            "SHAP, modelin tahminine her özelliğin ne kadar katkı sağladığını gösterir."
        )
        if supports_tree_explainer(pipeline):
            shap_fig = plot_local_waterfall(pipeline, instance_df, label_index, label_encoder)
            st.pyplot(shap_fig)
            plt.close(shap_fig)
        else:
            st.warning(
                "Seçilen model SHAP TreeExplainer ile uyumlu değil. "
                "KernelExplainer veya Permutation Importance kullanılmalıdır."
            )

    with tab_lime:
        st.markdown(
            "LIME, yerel bir yaklaşımla bu örnek için en etkili özellikleri listeler."
        )
        instance_df_engineered = FeatureEngineer().transform(instance_df)
        predict_fn = lambda x: pipeline[1:].predict_proba(
            pd.DataFrame(x, columns=ALL_FEATURE_COLUMNS)
        )
        lime_fig = explain_instance_lime(
            pipeline,
            lime_explainer,
            instance_df_engineered,
            label_index,
            predict_fn=predict_fn,
        )
        st.pyplot(lime_fig)
        plt.close(lime_fig)


if __name__ == "__main__":
    main()
