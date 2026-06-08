"""EDA notebook: IQR outlier hücresi, politika metni ve boxplot yorumu düzeltmesi."""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "01_eda.ipynb"

BOXPLOT_YORUM = (
    "## 4. Feature Boxplots\n\n"
    "**Yorum:** K ve rainfall en fazla aykırı değere sahip (IQR kutularının dışındaki noktalar). "
    "Nem ve pH dar aralıkta — aykırı değer riski düşük. "
    "StandardScaler tek başına aykırı değerleri yönetmez; "
    "modelleme pipeline'ında EDA ile uyumlu **1.5×IQR Winsorize (`OutlierClipper`)** uygulanır, "
    "ardından `StandardScaler` kırpılmış dağılım üzerinde fit edilir.\n"
)

POLICY_MD = """### Aykırı Değer Politikası (EDA → Modelleme)

IQR analizi (1.5×IQR) sonucunda **K** ve **rainfall** en yüksek aykırı değer sayısına sahip; **humidity** ve **ph** dar aralıkta.

**Karar:**
- **Silme (drop):** Uygulanmadı — dengeli sınıf yapısı korunur.
- **Kırpma (clip):** Uygulandı — `OutlierClipper` eğitim setinde `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` sınırlarına göre Winsorize eder.
- **RobustScaler:** Tercih edilmedi — aykırı değerler doğrudan sınırlandırıldıktan sonra `StandardScaler` kullanılır.

> **Teorik not:** `StandardScaler` aykırı değerleri "yönetmez"; ortalama/varyans uç değerlerden etkilenir. Fit yalnızca eğitim fold'unda yapılır (data leakage yok).
"""

OUTLIER_CODE = """# IQR tabanlı aykırı değer tespiti (EDA — Tukey 1.5×IQR)
IQR_MULTIPLIER = 1.5
outlier_summary = []

for col in FEATURE_COLUMNS:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR_MULTIPLIER * IQR
    upper_bound = Q3 + IQR_MULTIPLIER * IQR
    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_summary.append({
        "feature": col,
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "outlier_count": int(outlier_count),
    })

outlier_summary = pd.DataFrame(outlier_summary).sort_values("outlier_count", ascending=False)
outlier_summary
"""


def _to_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines if lines else [text]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    nb["cells"] = [
        cell
        for cell in nb["cells"]
        if not (
            cell.get("cell_type") == "markdown"
            and "### Aykırı Değer Politikası" in "".join(cell.get("source", []))
        )
    ]

    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        if "## 4. Feature Boxplots" in src and "StandardScaler bu aykırı" in src:
            cell["source"] = _to_source(BOXPLOT_YORUM)
        if "## 0. Veri Kalitesi Kontrolleri" in src:
            cell["source"] = _to_source(
                "## 0. Veri Kalitesi Kontrolleri\n\n"
                "> **Not:** Bu EDA tüm veri seti üzerinde, train/test split **öncesi** yapılmaktadır. "
                "Modelleme aşamasında veri %80/%20 stratified olarak ayrılır (`src/data/loader.py`).\n\n"
                "**Kontrol edilenler:** eksik değer, tekrarlayan satır, veri tipleri, sınıf dengesi, "
                "özellik aralıkları, **IQR (1.5×) aykırı değer sayımı**.\n"
            )

    insert_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code" and "range_checks" in "".join(cell.get("source", [])):
            insert_idx = i + 1
            break

    if insert_idx is None:
        raise RuntimeError("Veri kalitesi kod hücresi bulunamadı.")

    has_outlier = any("outlier_summary" in "".join(c.get("source", [])) for c in nb["cells"])
    if not has_outlier:
        nb["cells"].insert(
            insert_idx,
            {"cell_type": "code", "metadata": {}, "source": _to_source(OUTLIER_CODE), "outputs": []},
        )
        nb["cells"].insert(
            insert_idx + 1,
            {"cell_type": "markdown", "metadata": {}, "source": _to_source(POLICY_MD)},
        )

    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Güncellendi: {NOTEBOOK}")


if __name__ == "__main__":
    main()
