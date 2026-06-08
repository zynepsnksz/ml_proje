"""Crop Recommendation — FarmVista tarzı Streamlit karar destek ve XAI arayüzü."""

from __future__ import annotations

import sys
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
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

FEATURE_LABELS_TR = {
    "N": "Azot (N)",
    "P": "Fosfor (P)",
    "K": "Potasyum (K)",
    "temperature": "Sıcaklık",
    "humidity": "Nem",
    "ph": "pH",
    "rainfall": "Yağış",
}

PERCENT_TOLERANCE = 0.15
PH_TOLERANCE = 0.5

NAV_PAGES = [
    ("overview", "Dashboard Overview"),
    ("recommendation", "Mahsul Önerisi"),
    ("feasibility", "Ürün Uygunluk Analizi"),
    ("ideal", "İdeal Yetişme Koşulları"),
    ("xai", "SHAP / LIME"),
]

NAV_ICONS = {
    "overview": "📊",
    "recommendation": "🌾",
    "feasibility": "🎯",
    "ideal": "🌱",
    "xai": "🔍",
}

PLOTLY_COLORS = {
    "primary": "#16a34a",
    "secondary": "#22c55e",
    "accent": "#3b82f6",
    "muted": "#94a3b8",
    "warning": "#f59e0b",
    "danger": "#ef4444",
}

FEASIBILITY_ACTIONS: dict[tuple[str, str], str] = {
    ("N", "Düşük"): "Azot artırılmalı",
    ("N", "Yüksek"): "Fazla azot/gübreleme kontrol edilmeli",
    ("P", "Düşük"): "Fosfor artırılmalı",
    ("P", "Yüksek"): "Fazla fosfor kontrol edilmeli",
    ("K", "Düşük"): "Potasyum artırılmalı",
    ("K", "Yüksek"): "Fazla gübreleme kontrol edilmeli",
    ("temperature", "Düşük"): "Sıcaklık koşulları iyileştirilmeli",
    ("temperature", "Yüksek"): "Serinletme veya gölgeleme gerekebilir",
    ("humidity", "Düşük"): "Nem artırılmalı",
    ("humidity", "Yüksek"): "Havalandırma veya nem kontrolü gerekli",
    ("ph", "Düşük"): "pH yükseltilmeli",
    ("ph", "Yüksek"): "pH düşürülmeli",
    ("rainfall", "Düşük"): "Sulama artırılmalı",
    ("rainfall", "Yüksek"): "Drenaj veya fazla su kontrolü gerekli",
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


def _format_feature_value(feature: str, value: float) -> str:
    """Özellik değerini tablo gösterimi için biçimlendirir."""
    if feature in {"N", "P", "K"}:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _evaluate_feature_status(current: float, ideal: float, feature: str) -> str:
    """Mevcut değeri ideal referansa göre Düşük / Uygun / Yüksek olarak sınıflandırır."""
    if feature == "ph":
        low, high = ideal - PH_TOLERANCE, ideal + PH_TOLERANCE
    else:
        low, high = ideal * (1 - PERCENT_TOLERANCE), ideal * (1 + PERCENT_TOLERANCE)

    if current < low:
        return "Düşük"
    if current > high:
        return "Yüksek"
    return "Uygun"


def _get_feasibility_action(feature: str, status: str) -> str:
    """Duruma göre yapılması gereken aksiyon metnini döndürür."""
    if status == "Uygun":
        return "Mevcut değer uygun"
    return FEASIBILITY_ACTIONS[(feature, status)]


def _build_feasibility_table(
    current_values: dict[str, float],
    ideal_values: dict[str, float],
) -> pd.DataFrame:
    """Seçilen ürün için mevcut ve ideal değer karşılaştırma tablosunu oluşturur."""
    rows = []
    for feature in FEATURE_COLUMNS:
        current = float(current_values[feature])
        ideal = float(ideal_values[feature])
        status = _evaluate_feature_status(current, ideal, feature)
        rows.append(
            {
                "Özellik": FEATURE_LABELS_TR[feature],
                "Mevcut": _format_feature_value(feature, current),
                "İdeal": _format_feature_value(feature, ideal),
                "Durum": status,
                "Yapılması Gereken": _get_feasibility_action(feature, status),
            }
        )
    return pd.DataFrame(rows)


def _style_feasibility_status(val: str) -> str:
    """Durum sütunu için arka plan rengi döndürür."""
    colors = {
        "Uygun": "background-color: #d1fae5; color: #065f46",
        "Düşük": "background-color: #fef3c7; color: #92400e",
        "Yüksek": "background-color: #fee2e2; color: #991b1b",
    }
    return colors.get(val, "")


def _on_scenario_change(scenarios: dict[str, dict[str, float]]) -> None:
    """Seçilen senaryo değerlerini slider durumlarına uygular."""
    selected = st.session_state["selected_scenario"]
    if selected != "--- Seçin ---":
        crop_key = selected.lower()
        _apply_scenario(scenarios[crop_key])


def _init_session_state() -> None:
    """Slider ve navigasyon durumunu başlatır."""
    for key, value in SLIDER_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "overview"


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


def _get_current_values() -> dict[str, float]:
    """Session state'ten mevcut ölçüm değerlerini döndürür."""
    return {
        "N": float(st.session_state["N"]),
        "P": float(st.session_state["P"]),
        "K": float(st.session_state["K"]),
        "temperature": float(st.session_state["temperature"]),
        "humidity": float(st.session_state["humidity"]),
        "ph": float(st.session_state["ph"]),
        "rainfall": float(st.session_state["rainfall"]),
    }


def _normalize_profile(
    values: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> list[float]:
    """Radar grafiği için özellik değerlerini 0-100 aralığına normalize eder."""
    normalized = []
    for feature in FEATURE_COLUMNS:
        low, high = bounds[feature]
        if high <= low:
            normalized.append(50.0)
        else:
            normalized.append(max(0.0, min(100.0, (values[feature] - low) / (high - low) * 100)))
    return normalized


def _plot_top3_donut(top3: list[tuple[str, float]], center_label: str = "En Uygun") -> go.Figure:
    """Top-3 mahsul olasılıkları için Plotly donut grafiği oluşturur."""
    labels = [crop.title() for crop, _ in top3]
    values = [prob * 100 for _, prob in top3]
    colors = [PLOTLY_COLORS["primary"], PLOTLY_COLORS["accent"], PLOTLY_COLORS["muted"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,
                marker={"colors": colors, "line": {"color": "#ffffff", "width": 2}},
                textinfo="percent",
                textposition="outside",
                hovertemplate="<b>%{label}</b><br>Olasılık: %{value:.1f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=340,
        margin={"t": 24, "b": 24, "l": 24, "r": 24},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "x": 0.5, "xanchor": "center"},
        annotations=[
            {
                "text": f"<b>{values[0]:.1f}%</b><br><span style='font-size:11px'>{center_label}</span>",
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 18, "color": "#0f172a"},
                "showarrow": False,
            }
        ],
    )
    return fig


def _plot_profile_radar(
    current_values: dict[str, float],
    ideal_values: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    current_label: str = "Mevcut Ölçümler",
    ideal_label: str = "İdeal Referans",
) -> go.Figure:
    """Mevcut ve ideal profilleri karşılaştıran Plotly radar grafiği oluşturur."""
    categories = [FEATURE_LABELS_TR[feature] for feature in FEATURE_COLUMNS]
    current_r = _normalize_profile(current_values, bounds)
    ideal_r = _normalize_profile(ideal_values, bounds)
    closed_categories = categories + [categories[0]]
    closed_current = current_r + [current_r[0]]
    closed_ideal = ideal_r + [ideal_r[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=closed_current,
            theta=closed_categories,
            fill="toself",
            name=current_label,
            line={"color": PLOTLY_COLORS["primary"], "width": 2},
            fillcolor="rgba(22, 163, 74, 0.18)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=closed_ideal,
            theta=closed_categories,
            fill="toself",
            name=ideal_label,
            line={"color": PLOTLY_COLORS["accent"], "width": 2, "dash": "dot"},
            fillcolor="rgba(59, 130, 246, 0.12)",
        )
    )
    fig.update_layout(
        height=420,
        margin={"t": 48, "b": 48, "l": 64, "r": 64},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar={
            "bgcolor": "rgba(248, 250, 252, 0.6)",
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "tickvals": [25, 50, 75, 100],
                "gridcolor": "#e2e8f0",
                "linecolor": "#cbd5e1",
            },
            "angularaxis": {"gridcolor": "#e2e8f0", "linecolor": "#cbd5e1"},
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.12, "x": 0.5, "xanchor": "center"},
    )
    return fig


def _render_kpi_card(icon: str, label: str, value: str, subtext: str = "", accent: str = "green") -> None:
    """Tek bir KPI kartını HTML olarak render eder."""
    accent_colors = {
        "green": "#16a34a",
        "blue": "#2563eb",
        "amber": "#d97706",
        "slate": "#475569",
    }
    color = accent_colors.get(accent, accent_colors["green"])
    st.markdown(
        f"""
        <div class="fv-kpi-card" style="border-top-color: {color};">
            <div class="fv-kpi-icon">{icon}</div>
            <div class="fv-kpi-label">{label}</div>
            <div class="fv-kpi-value">{value}</div>
            <div class="fv-kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_page_header(title: str, subtitle: str) -> None:
    """Sayfa başlığı ve açıklama metnini render eder."""
    st.markdown(
        f"""
        <div class="fv-page-header">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_panel(title: str) -> None:
    """Panel başlığı render eder."""
    st.markdown(f'<div class="fv-panel-title">{title}</div>', unsafe_allow_html=True)


def _render_ideal_cards(ideal_vals: dict[str, float]) -> None:
    """İdeal yetişme koşulları kartlarını render eder."""
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


def _inject_dashboard_css() -> None:
    """FarmVista tarzı dashboard teması için global CSS enjekte eder."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        :root {
            --fv-bg: #f4f7f5;
            --fv-surface: #ffffff;
            --fv-border: #e2e8f0;
            --fv-text: #0f172a;
            --fv-muted: #64748b;
            --fv-primary: #16a34a;
            --fv-primary-soft: #dcfce7;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--fv-border);
            box-shadow: 4px 0 24px rgba(15, 23, 42, 0.04);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1rem;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }

        .fv-brand {
            padding: 0.25rem 0.25rem 1rem 0.25rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--fv-border);
        }

        .fv-brand-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.65rem;
        }

        .fv-brand-icon {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            box-shadow: 0 8px 20px rgba(22, 163, 74, 0.25);
        }

        .fv-brand h1 {
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0;
            color: var(--fv-text);
            line-height: 1.2;
        }

        .fv-brand p {
            margin: 0.15rem 0 0 0;
            font-size: 0.72rem;
            color: var(--fv-muted);
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .fv-nav-label {
            font-size: 0.7rem;
            font-weight: 700;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 1rem 0 0.5rem 0.25rem;
        }

        div[data-testid="stSidebar"] .stRadio > div {
            gap: 0.35rem;
        }

        div[data-testid="stSidebar"] .stRadio label {
            background: #f8fafc;
            border: 1px solid transparent;
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
            font-weight: 600;
            color: #334155;
            transition: all 0.2s ease;
        }

        div[data-testid="stSidebar"] .stRadio label:hover {
            background: #f0fdf4;
            border-color: #bbf7d0;
        }

        div[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
            background: linear-gradient(135deg, #ecfdf5 0%, #dcfce7 100%);
            border-color: #86efac;
            color: #166534;
            box-shadow: 0 4px 12px rgba(22, 163, 74, 0.12);
        }

        .fv-sidebar-section {
            font-size: 0.7rem;
            font-weight: 700;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 1.25rem 0 0.75rem 0.25rem;
        }

        .fv-page-header h2 {
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--fv-text);
            margin: 0 0 0.35rem 0;
        }

        .fv-page-header p {
            color: var(--fv-muted);
            margin: 0 0 1.25rem 0;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .fv-panel-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #1e293b;
            margin: 0 0 0.75rem 0;
        }

        .fv-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin-bottom: 1.25rem;
        }

        @media (max-width: 1100px) {
            .fv-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        .fv-kpi-card {
            background: var(--fv-surface);
            border: 1px solid var(--fv-border);
            border-top: 4px solid var(--fv-primary);
            border-radius: 16px;
            padding: 1.1rem 1.15rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
            min-height: 128px;
        }

        .fv-kpi-icon {
            font-size: 1.35rem;
            margin-bottom: 0.35rem;
        }

        .fv-kpi-label {
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--fv-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }

        .fv-kpi-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--fv-text);
            line-height: 1.2;
            margin-bottom: 0.25rem;
        }

        .fv-kpi-sub {
            font-size: 0.78rem;
            color: #64748b;
        }

        .fv-chart-card, .fv-content-card {
            background: var(--fv-surface);
            border: 1px solid var(--fv-border);
            border-radius: 18px;
            padding: 1.1rem 1.15rem 0.5rem 1.15rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
        }

        .fv-content-card {
            padding: 1.15rem 1.25rem;
        }

        .confidence-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
            min-height: 108px;
        }

        .confidence-metric-label {
            font-size: 0.72rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .confidence-metric-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #0f172a;
        }

        .ideal-values-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 8px;
            margin-bottom: 8px;
        }

        .ideal-val-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
            display: flex;
            flex-direction: column;
            border-top: 4px solid #cbd5e1;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .ideal-val-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.08);
        }

        .npk-card { border-top-color: #10b981; }
        .env-card { border-top-color: #3b82f6; }

        .card-icon { font-size: 1.6rem; margin-bottom: 8px; }
        .card-title {
            font-size: 0.78rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }
        .card-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 8px;
        }
        .card-desc {
            font-size: 0.75rem;
            color: #64748b;
            line-height: 1.45;
            margin-top: auto;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #f8fafc;
            border-radius: 14px;
            padding: 6px;
            border: 1px solid #e2e8f0;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(bounds: dict, scenarios: dict) -> None:
    """Sol menü, navigasyon ve saha ölçüm kontrollerini render eder."""
    st.markdown(
        """
        <div class="fv-brand">
            <div class="fv-brand-badge">
                <div class="fv-brand-icon">🌿</div>
                <div>
                    <h1>FarmVista</h1>
                    <p>AgTech Analytics</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fv-nav-label">Navigasyon</div>', unsafe_allow_html=True)
    nav_labels = [f"{NAV_ICONS[key]}  {label}" for key, label in NAV_PAGES]
    nav_keys = [key for key, _ in NAV_PAGES]
    selected_label = st.radio(
        "Sayfa",
        nav_labels,
        index=nav_keys.index(st.session_state.nav_page),
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.session_state.nav_page = nav_keys[nav_labels.index(selected_label)]

    st.markdown('<div class="fv-sidebar-section">Saha Ölçüm Değerleri</div>', unsafe_allow_html=True)
    st.slider("Azot (N)", min_value=int(bounds["N"][0]), max_value=int(bounds["N"][1]), key="N")
    st.slider("Fosfor (P)", min_value=int(bounds["P"][0]), max_value=int(bounds["P"][1]), key="P")
    st.slider("Potasyum (K)", min_value=int(bounds["K"][0]), max_value=int(bounds["K"][1]), key="K")
    st.slider(
        "Sıcaklık (°C)",
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
        "pH",
        min_value=float(bounds["ph"][0]),
        max_value=float(bounds["ph"][1]),
        step=0.1,
        key="ph",
    )
    st.slider(
        "Yağış (mm)",
        min_value=float(bounds["rainfall"][0]),
        max_value=float(bounds["rainfall"][1]),
        step=0.1,
        key="rainfall",
    )

    st.markdown('<div class="fv-sidebar-section">Referans Şablonları</div>', unsafe_allow_html=True)
    crop_list = sorted(scenarios.keys())
    st.selectbox(
        "İdeal koşullara ayarla",
        ["--- Seçin ---"] + [crop.title() for crop in crop_list],
        key="selected_scenario",
        on_change=_on_scenario_change,
        args=(scenarios,),
        label_visibility="collapsed",
    )


def _render_overview_page(
    *,
    top3: list[tuple[str, float]],
    best_crop: str,
    best_score: float,
    confidence_score: float,
    confidence_level: str,
    current_values: dict[str, float],
    ideal_values: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    feasibility_df: pd.DataFrame,
) -> None:
    """Dashboard Overview sayfasını render eder."""
    _render_page_header(
        "Dashboard Overview",
        "Saha ölçümlerinizin anlık özeti, model önerisi ve profil analizi tek ekranda.",
    )

    suitable_count = int((feasibility_df["Durum"] == "Uygun").sum())
    level_label = get_confidence_level_label(confidence_level)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        _render_kpi_card("🌾", "En Uygun Mahsul", best_crop.title(), f"{best_score * 100:.1f}% olasılık")
    with kpi2:
        _render_kpi_card("✅", "Confidence Score", f"{confidence_score}%", level_label, accent="blue")
    with kpi3:
        _render_kpi_card("🧪", "Uygun Özellik", f"{suitable_count}/7", "Seçilen ürün için", accent="amber")
    with kpi4:
        _render_kpi_card("🌡️", "Sıcaklık", f"{current_values['temperature']:.1f}°C", f"pH {current_values['ph']:.1f}", accent="slate")

    chart_left, chart_right = st.columns([1, 1.15])
    with chart_left:
        st.markdown('<div class="fv-chart-card">', unsafe_allow_html=True)
        _render_panel("Top-3 Mahsul Dağılımı")
        st.plotly_chart(_plot_top3_donut(top3), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_right:
        st.markdown('<div class="fv-chart-card">', unsafe_allow_html=True)
        _render_panel(f"Profil Radarı — {best_crop.title()} Referansı")
        st.plotly_chart(
            _plot_profile_radar(
                current_values,
                ideal_values,
                bounds,
                ideal_label=f"{best_crop.title()} İdeal",
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="fv-content-card">', unsafe_allow_html=True)
    _render_panel("Hızlı Saha Özeti")
    summary_cols = st.columns(7)
    for col, feature in zip(summary_cols, FEATURE_COLUMNS):
        with col:
            st.metric(FEATURE_LABELS_TR[feature], _format_feature_value(feature, current_values[feature]))
    st.markdown("</div>", unsafe_allow_html=True)


def _render_recommendation_page(
    *,
    top3: list[tuple[str, float]],
    best_crop: str,
    confidence_score: float,
    confidence_level: str,
    probability_margin: float,
) -> None:
    """Mahsul önerisi sayfasını render eder."""
    _render_page_header(
        "Mahsul Önerisi",
        "Modelin Top-3 önerileri ve tahmin güvenilirliği analizi.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("En Uygun Mahsul", best_crop.title(), f"{top3[0][1] * 100:.2f}%")
    with col2:
        st.metric("2. Alternatif", top3[1][0].title(), f"{top3[1][1] * 100:.2f}%")
    with col3:
        st.metric("3. Alternatif", top3[2][0].title(), f"{top3[2][1] * 100:.2f}%")

    st.markdown('<div class="fv-content-card">', unsafe_allow_html=True)
    st.markdown("#### Prediction Confidence")
    st.caption("Confidence Analysis → model bu karardan ne kadar emin?")
    conf_col1, conf_col2, conf_col3 = st.columns(3)
    with conf_col1:
        st.markdown(
            f"""
            <div class="confidence-card">
                <div class="confidence-metric-label">Confidence Score</div>
                <div class="confidence-metric-value">{confidence_score}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with conf_col2:
        level_label = get_confidence_level_label(confidence_level)
        st.markdown(
            f"""
            <div class="confidence-card">
                <div class="confidence-metric-label">Confidence Level</div>
                <div class="confidence-metric-value">{level_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with conf_col3:
        st.markdown(
            f"""
            <div class="confidence-card">
                <div class="confidence-metric-label">Probability Margin</div>
                <div class="confidence-metric-value">{probability_margin}%</div>
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
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="fv-chart-card">', unsafe_allow_html=True)
    _render_panel("Olasılık Dağılımı (Donut)")
    st.plotly_chart(_plot_top3_donut(top3, center_label="1. Öneri"), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


def _render_feasibility_page(
    *,
    scenarios: dict[str, dict[str, float]],
    best_crop: str,
    current_values: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> tuple[str, pd.DataFrame]:
    """Ürün uygunluk analizi sayfasını render eder."""
    _render_page_header(
        "Ürün Uygunluk Analizi",
        "Seçtiğiniz ürünü mevcut koşullarda yetiştirebilir misiniz? Referans değerlerle karşılaştırın.",
    )

    crop_options = sorted(scenarios.keys())
    default_crop_index = crop_options.index(best_crop) if best_crop in crop_options else 0
    selected_feasibility_crop = st.selectbox(
        "Yetiştirmek istediğiniz ürün:",
        crop_options,
        index=default_crop_index,
        format_func=lambda crop: crop.title(),
        key="feasibility_crop",
    )

    ideal_values = scenarios[selected_feasibility_crop]
    feasibility_df = _build_feasibility_table(current_values, ideal_values)
    unsuitable_count = int((feasibility_df["Durum"] != "Uygun").sum())

    if unsuitable_count == 0:
        st.success(
            f"Mevcut koşullarınız **{selected_feasibility_crop.title()}** yetiştirmek için "
            "referans aralıkları içinde görünüyor."
        )
    else:
        st.warning(
            f"**{selected_feasibility_crop.title()}** için {unsuitable_count} özellikte "
            "iyileştirme gerekebilir."
        )

    chart_col, table_col = st.columns([1.05, 1.2])
    with chart_col:
        st.markdown('<div class="fv-chart-card">', unsafe_allow_html=True)
        _render_panel(f"Radar Karşılaştırma — {selected_feasibility_crop.title()}")
        st.plotly_chart(
            _plot_profile_radar(
                current_values,
                ideal_values,
                bounds,
                ideal_label=f"{selected_feasibility_crop.title()} İdeal",
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with table_col:
        st.markdown('<div class="fv-content-card">', unsafe_allow_html=True)
        _render_panel("Uygunluk Tablosu")
        st.caption("Eşikler: N, P, K, sıcaklık, nem ve yağış için ±%15; pH için ±0.5 birim.")
        styled_feasibility = feasibility_df.style.map(_style_feasibility_status, subset=["Durum"])
        st.dataframe(styled_feasibility, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    return selected_feasibility_crop, feasibility_df


def _render_ideal_page(scenarios: dict[str, dict[str, float]]) -> None:
    """İdeal yetişme koşulları sayfasını render eder."""
    _render_page_header(
        "İdeal Yetişme Koşulları",
        "Veri seti referans ortalamalarına göre seçilen ürünün ideal toprak ve iklim profili.",
    )

    selected = st.session_state.get("selected_scenario", "--- Seçin ---")
    crop_list = sorted(scenarios.keys())

    if selected == "--- Seçin ---":
        selected = st.selectbox(
            "İncelemek istediğiniz ürün:",
            crop_list,
            format_func=lambda crop: crop.title(),
            key="ideal_page_crop",
        )
        crop_key = selected
        display_name = selected.title()
    else:
        crop_key = selected.lower()
        display_name = selected
        st.info(f"Sol menüdeki referans şablonundan **{display_name}** seçili.")

    ideal_vals = scenarios[crop_key]
    st.markdown('<div class="fv-content-card">', unsafe_allow_html=True)
    _render_panel(f"{display_name} — Veri Seti Referans Değerleri")
    _render_ideal_cards(ideal_vals)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_xai_page(
    *,
    pipeline,
    label_encoder,
    lime_explainer,
    instance_df: pd.DataFrame,
    label_index: int,
    confidence_score: float,
) -> None:
    """SHAP ve LIME açıklama sayfasını render eder."""
    _render_page_header(
        "SHAP / LIME",
        "Modelin kararını açıklayan yapay zeka analizleri.",
    )

    st.info(
        "**SHAP / LIME** — Modelin bu tahmine hangi özelliklerle ulaştığını açıklar. "
        f"Confidence Score: **{confidence_score}%**"
    )

    tab_shap, tab_lime = st.tabs(["SHAP Waterfall Açıklaması", "LIME Yerel Katkı Açıklaması"])

    with tab_shap:
        st.markdown('<div class="fv-content-card">', unsafe_allow_html=True)
        st.markdown("SHAP, modelin tahminine her özelliğin ne kadar katkı sağladığını gösterir.")
        if supports_tree_explainer(pipeline):
            shap_fig = plot_local_waterfall(pipeline, instance_df, label_index, label_encoder)
            st.pyplot(shap_fig)
            plt.close(shap_fig)
        else:
            st.warning(
                "Seçilen model SHAP TreeExplainer ile uyumlu değil. "
                "KernelExplainer veya Permutation Importance kullanılmalıdır."
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_lime:
        st.markdown('<div class="fv-content-card">', unsafe_allow_html=True)
        st.markdown("LIME, yerel bir yaklaşımla bu örnek için en etkili özellikleri listeler.")
        instance_df_engineered = FeatureEngineer().transform(instance_df)
        predict_fn = lambda x: pipeline[1:].predict_proba(pd.DataFrame(x, columns=ALL_FEATURE_COLUMNS))
        lime_fig = explain_instance_lime(
            pipeline,
            lime_explainer,
            instance_df_engineered,
            label_index,
            predict_fn=predict_fn,
        )
        st.pyplot(lime_fig)
        plt.close(lime_fig)
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    """Streamlit uygulamasını çalıştırır."""
    st.set_page_config(
        page_title="FarmVista | Akıllı Tarım Analytics",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_dashboard_css()
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
        _render_sidebar(bounds, scenarios)

    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    lime_explainer = load_lime_explainer(artifact)

    instance_df = _build_instance_df()
    try:
        prediction_result = predict_with_confidence(instance_df, artifact=artifact)
        label_index = int(pipeline.predict(instance_df)[0])
    except ValueError as exc:
        st.error(f"⚠️ Girdi Doğrulama Hatası: {str(exc)}")
        st.stop()

    top3 = [(item["crop"], item["probability"]) for item in prediction_result["top_predictions"]]
    best_crop = prediction_result["prediction"]
    best_score = top3[0][1]
    confidence_score = prediction_result["confidence_score"]
    confidence_level = prediction_result["confidence_level"]
    probability_margin = prediction_result["probability_margin"]

    current_values = _get_current_values()
    feasibility_crop = st.session_state.get("feasibility_crop", best_crop)
    if feasibility_crop not in scenarios:
        feasibility_crop = best_crop
    overview_feasibility_df = _build_feasibility_table(current_values, scenarios[feasibility_crop])

    page = st.session_state.nav_page
    if page == "overview":
        _render_overview_page(
            top3=top3,
            best_crop=best_crop,
            best_score=best_score,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            current_values=current_values,
            ideal_values=scenarios[best_crop],
            bounds=bounds,
            feasibility_df=overview_feasibility_df,
        )
    elif page == "recommendation":
        _render_recommendation_page(
            top3=top3,
            best_crop=best_crop,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            probability_margin=probability_margin,
        )
    elif page == "feasibility":
        _render_feasibility_page(
            scenarios=scenarios,
            best_crop=best_crop,
            current_values=current_values,
            bounds=bounds,
        )
    elif page == "ideal":
        _render_ideal_page(scenarios)
    elif page == "xai":
        _render_xai_page(
            pipeline=pipeline,
            label_encoder=label_encoder,
            lime_explainer=lime_explainer,
            instance_df=instance_df,
            label_index=label_index,
            confidence_score=confidence_score,
        )


if __name__ == "__main__":
    main()
