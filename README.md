# Crop Recommendation System

Toprak ve iklim verilerine göre 22 farklı ürün arasından en uygun mahsulü öneren ML projesi.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma Sırası

1. `notebooks/01_eda.ipynb` — EDA (14 grafik → `report/assets/`)
2. `python -m src.models.train` — Model eğitimi, metrikler ve grafikler → `outputs/`
3. `python -m src.validation.robustness_checks` — Permutation test + multi-seed hold-out (→ `outputs/robustness_checks.json`)
4. `streamlit run app/streamlit_app.py` — Demo uygulama (Top-3 + SHAP + LIME)

Tek komut (eğitim + doğrulama):

```bash
python -m src.models.train
```

## Klasör Yapısı

```
ml_proje/
├── data/                   # Ham veri (Crop_recommendation.csv)
├── notebooks/              # EDA notebook
├── report/                 # Rapor ve EDA görselleri
│   ├── assets/             # 14 EDA grafiği
│   └── FINAL_REPORT.md     # Akademik rapor iskeleti
├── src/                    # Kaynak kod
│   ├── config.py
│   ├── data/
│   ├── preprocessing.py
│   ├── models/
│   ├── validation/
│   └── explainability/
├── app/                    # Streamlit demo
├── models/                 # best_model.pkl
├── outputs/                # Metrikler ve model grafikleri
├── tests/                  # Unit testler
└── requirements.txt
```

## Model Seçimi ve Hiperparametre Optimizasyonu

Model seçimi **LazyPredict** ile yapılır; karşılaştırma yalnızca eğitim setinin iç validation bölünmesi (%80/%20) üzerinde gerçekleşir. Test seti model seçimine dahil edilmez.

Validation skorunda GaussianNB veya QDA bazen RandomForest ile eşit/üstün çıkabilir. Ancak proje **SHAP TreeExplainer** ile uyumlu olması için ağaç tabanlı modeller arasından (`RandomForest`, `ExtraTrees`, `XGBoost`, `LightGBM`, `DecisionTree`) en yüksek accuracy'ye sahip olan seçilir. Bu bilinçli bir **performans–açıklanabilirlik trade-off**'udur.

Seçilen en iyi model için **RandomizedSearchCV** kullanılarak modele özel dinamik bir hiperparametre arama gridi (XGBoost, RandomForest, LGBM, DecisionTree vb.) üzerinden optimizasyon gerçekleştirilir. Hiperparametre optimizasyonu ve çapraz doğrulama (cross-validation) süreçlerinde çok sınıflı sınıflandırma problemlerinde doğruluğu en iyi yansıtan **`f1_macro`** metriği skorlama kriteri olarak kullanılır. Overfitting (ezberleme) riskini en aza indirmek amacıyla arama uzayındaki model derinlikleri (`max_depth`) ve dallanma parametreleri (`min_samples_leaf`, `min_samples_split`, `num_leaves`) uygun şekilde regüle edilmiştir.

## Veri Ön İşleme ve Domain Validation

*   **OutlierClipper:** EDA'daki **1.5×IQR (Tukey fences)** analiziyle tespit edilen aykırı değerler silinmez; eğitim setinde hesaplanan IQR sınırlarına göre **Winsorize/kırpma** uygulanır. `StandardScaler` yalnızca kırpılmış dağılım üzerinde fit edilir — tek başına ölçekleme aykırı değerleri yönetmez.
*   **Domain Validation:** Fiziksel kurallara (toprak asitliği pH $\in [0, 14]$, nem $\in [0, 100]$, Azot/Fosfor/Potasyum $\ge 0$, Yağış $\ge 0$) uygunluğu garanti altına almak için hem veri yükleme hem de tahmin girdi katmanlarında **`validate_inputs`** doğrulayıcısı çalıştırılır.

## Dosya Referansı

| Dosya | Amaç |
|---|---|
| `data/Crop_recommendation.csv` | Ham veri — 2200 satır, 7 özellik + label |
| `src/config.py` | Yollar, sütunlar, split/CV sabitleri |
| `src/data/loader.py` | Veri yükleme ve stratified split |
| `src/preprocessing.py` | StandardScaler pipeline, LabelEncoder |
| `src/models/train.py` | LazyPredict seçimi, CV, test, grafikler, kayıt |
| `src/models/predict.py` | predict, predict_top3, predict_proba |
| `src/explainability/shap_analysis.py` | Global summary + local waterfall |
| `src/explainability/lime_analysis.py` | LIME local açıklama |
| `app/streamlit_app.py` | Slider + Top-3 + SHAP/LIME sekmeleri |
| `notebooks/01_eda.ipynb` | EDA — 14 grafik, veri kalitesi kontrolleri |
| `report/FINAL_REPORT.md` | Akademik final raporu |
| `models/best_model.pkl` | Pipeline + LabelEncoder + metadata |
| `outputs/metrics.json` | Tüm metrikler ve hata analizi |
| `src/validation/robustness_checks.py` | Permutation test + multi-seed hold-out analizi |

## Sağlamlık Analizleri

Ana modeli (`models/best_model.pkl`) değiştirmeden çalışır:

```bash
python -m src.validation.robustness_checks
```

- **Permutation test:** `random_state=42` split üzerinde gerçek etiketler vs karıştırılmış etiketler; karıştırılmış durumda skorun ≈ 1/22 ≈ 0.045 şans düzeyine inmesi beklenir.
- **Multi-seed hold-out:** `[42, 123, 777, 2026, 3407]` seed'leriyle stratified split; test accuracy ve f1_macro ortalama ± std.

Çıktılar: `outputs/robustness_checks.json`, `outputs/robustness_checks.csv`

## Çıktılar (`outputs/`)

| Dosya | Açıklama |
|---|---|
| `metrics.json` | CV, test, sınıf bazlı metrikler, hata analizi |
| `confusion_matrix.png` | Ham confusion matrix |
| `confusion_matrix_normalized.png` | Normalize edilmiş CM |
| `roc_curve_ovr.png` | One-vs-Rest ROC |
| `learning_curve.png` | Train/test learning curve |
| `feature_importance.png` | RF özellik önemleri |
| `shap_summary.png` | Global SHAP summary |
| `model_comparison.png` | LazyPredict top-10 |
| `lazypredict_results.csv` | Tüm model karşılaştırması |
| `robustness_checks.json` | Permutation + multi-seed sonuçları ve yorum |
| `robustness_checks.csv` | Aynı sonuçların tablo formatı |

## Testler

```bash
python -m pytest tests/ -v
```
