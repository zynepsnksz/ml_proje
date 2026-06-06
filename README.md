# Crop Recommendation System

Toprak ve iklim verilerine göre 22 farklı ürün arasından en uygun mahsulü öneren ML projesi.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma Sırası

1. `notebooks/01_eda.ipynb` — EDA (14 grafik → `report/assets/`)
2. `python -m src.models.train` — Model eğitimi, metrikler ve grafikler → `outputs/`
3. `streamlit run app/streamlit_app.py` — Demo uygulama (Top-3 + SHAP + LIME)

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
│   └── explainability/
├── app/                    # Streamlit demo
├── models/                 # best_model.pkl
├── outputs/                # Metrikler ve model grafikleri
├── tests/                  # Unit testler
└── requirements.txt
```

## Model Seçimi

Model seçimi **LazyPredict** ile yapılır; karşılaştırma yalnızca eğitim setinin iç validation bölünmesi (%80/%20) üzerinde gerçekleşir. Test seti model seçimine dahil edilmez.

Validation skorunda GaussianNB veya QDA bazen RandomForest ile eşit/üstün çıkabilir. Ancak proje **SHAP TreeExplainer** ile uyumlu olması için ağaç tabanlı modeller arasından (`RandomForest`, `ExtraTrees`, `XGBoost`, `LightGBM`, `DecisionTree`) en yüksek accuracy'ye sahip olan seçilir. Bu bilinçli bir **performans–açıklanabilirlik trade-off**'udur.

Hiperparametre tuning bilinçli olarak yapılmamıştır; sklearn varsayılan parametreleri kullanılır.

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

## Testler

```bash
python -m pytest tests/ -v
```
