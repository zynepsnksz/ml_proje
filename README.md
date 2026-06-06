# Crop Recommendation System

Toprak ve iklim verilerine göre 22 farklı ürün arasından en uygun mahsulü öneren ML projesi.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma Sırası

1. `notebooks/01_eda.ipynb` — EDA
2. `python -m src.models.train` — Model eğitimi
3. `streamlit run app/streamlit_app.py` — Demo uygulama

## Klasör Yapısı

```
ml_proje/
├── data/                   # Ham veri
├── notebooks/              # EDA
├── src/                    # Kaynak kod
├── app/                    # Streamlit
├── models/                 # Eğitilmiş model (.pkl)
├── outputs/                # Grafikler, metrikler
└── requirements.txt
```

## Dosya Referansı

| Dosya | Amaç | İçerik | Bağımlılıklar |
|---|---|---|---|
| `data/Crop_recommendation.csv` | Ham veri | 2200 satır, 7 özellik + label | — |
| `src/config.py` | Sabitler | Yollar, sütunlar, RF parametreleri | pathlib |
| `src/data/loader.py` | Veri yükleme | load, split, feature/target ayırma | pandas, sklearn |
| `src/preprocessing.py` | Ön işleme | StandardScaler pipeline | sklearn |
| `src/models/train.py` | Eğitim | 4 model CV karşılaştırma, RF eğitimi, kaydetme | sklearn, joblib |
| `src/models/predict.py` | Tahmin | predict, predict_top3 | joblib, numpy |
| `src/explainability/shap_analysis.py` | SHAP | summary, waterfall plot | shap, matplotlib |
| `src/explainability/lime_analysis.py` | LIME | local açıklama | lime |
| `app/streamlit_app.py` | UI | Slider + Top-3 + XAI | streamlit |
| `notebooks/01_eda.ipynb` | EDA | 9 grafik (pairplot yok) | pandas, matplotlib, seaborn |
| `models/best_model.pkl` | Model artifact | Eğitilmiş pipeline | — |
| `outputs/` | Çıktılar | metrics.json, grafikler | — |

## Sadeleştirmeler

- PDF generator kaldırıldı
- Hyperparameter tuning yok (sabit RF parametreleri)
- Pairplot kullanılmıyor
- Öncelik: çalışan uçtan uca pipeline
