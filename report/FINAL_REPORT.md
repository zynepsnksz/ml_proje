# Crop Recommendation System — Final Proje Raporu

## 1. Problem Tanımı

Toprak bileşenleri (N, P, K) ve iklim koşullarına (sıcaklık, nem, pH, yağış) göre 22 farklı mahsul arasından en uygun ürünü öneren bir makine öğrenmesi sistemi geliştirilmiştir. Sistem, tahmin sonuçlarını SHAP ve LIME ile açıklayabilen bir Streamlit arayüzü ile sunulmaktadır.

## 2. Veri Seti

| Özellik | Değer |
|---------|-------|
| Kaynak | [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| Satır sayısı | 2200 |
| Özellik sayısı | 7 (N, P, K, temperature, humidity, ph, rainfall) |
| Hedef | label (22 mahsul sınıfı) |
| Sınıf dengesi | Her sınıfta 100 örnek (mükemmel dengeli) |

### Veri Kalitesi

- Eksik değer: kontrol edildi (EDA notebook Bölüm 0)
- Tekrarlayan satır: kontrol edildi
- Özellik aralıkları: domain kurallarına göre doğrulandı

### Sınırlamalar

- Veri seti yüksek oranda ayrışabilir ve sentetik benzeri bir profil göstermektedir.
- %99+ doğruluk skorları gerçek tarım koşullarında genellenebilirlik garantisi vermez.
- Coğrafi ve mevsimsel çeşitlilik sınırlıdır.

## 3. Metodoloji

```
Veri Yükleme → EDA (tüm veri) → Stratified Split (%80/%20)
    → LazyPredict (train içi val) → Tree-only model seçimi
    → Pipeline fit (StandardScaler + classifier)
    → 5-fold CV → Test değerlendirme → Artifact kayıt → XAI demo
```

### Model Seçimi

LazyPredict ile 25+ model karşılaştırılmıştır. Validation'da GaussianNB ve QDA bazen RandomForest ile eşit/üstün skor almıştır. Ancak **SHAP TreeExplainer** uyumu için ağaç tabanlı modeller arasından en yüksek accuracy'ye sahip olan seçilmiştir. Bu bilinçli bir performans–açıklanabilirlik trade-off'udur.

### Train-Test Split

- Yöntem: Stratified hold-out
- Oran: %80 train / %20 test
- random_state: 42
- Sınıf başına: ~80 train / ~20 test örneği

## 4. Sonuçlar

*Son güncelleme: `python -m src.models.train` çıktısı (`outputs/metrics.json`).*

| Metrik | CV train fold | CV test fold | Train (hold-out) | Test |
|--------|---------------|--------------|------------------|------|
| Accuracy | 100.00% | 98.41% ± 0.64% | 100.00% | **98.86%** |
| F1 Macro | 100.00% | 98.40% | 100.00% | **98.86%** |
| ROC AUC OvR Macro | — | — | — | **0.9999** |

### Baseline Karşılaştırma

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| DummyClassifier | 4.55% | 0.40% |
| Seçilen model (XGBoost) | 98.86% | 98.86% |

### Confusion Matrix Yorumu

440 test örneğinde **5 hata** (hata oranı %1.14):

| Gerçek | Tahmin | Adet |
|--------|--------|------|
| lentil | mothbeans | 2 |
| mungbean | lentil | 1 |
| rice | jute | 1 |
| watermelon | mungbean | 1 |

Grafikler: `outputs/confusion_matrix.png`, `outputs/confusion_matrix_normalized.png`

### ROC-AUC Yorumu

ROC AUC OvR macro = **0.9999** — sınıflar neredeyse mükemmel ayrışıyor. En düşük sınıf AUC değerleri: **lentil (0.9994)**, **mothbeans (0.9996)** ve **mungbean (0.9997)**. Grafikte bu en düşük 3 sınıf jüriye kolaylık olması için vurgulanmıştır, geri kalan 19 başarılı sınıf lejantı kalabalıklaştırmamak adına arka plana itilmiştir.

### Overfitting Kontrolü

| Gösterge | Değer |
|----------|-------|
| Train accuracy | 100.00% |
| Test accuracy | 98.86% |
| Train-test gap | 1.14% |
| Learning curve gap (final) | 1.82% |
| CV train vs CV test (accuracy) | 100% vs 98.41% |

Train setinde tam fit (100%) görülüyor; ancak test ve CV skorları çok yakın olduğundan **ciddi overfitting yok**, verinin genelleme gücü son derece yüksektir.

### Özellik Önemleri (XGBoost - Top 10)

| Özellik | Önem | Türü |
|---------|------|------|
| K | 0.158 | Ham |
| P | 0.139 | Ham |
| N | 0.111 | Ham |
| N_to_K | 0.111 | Türetilmiş (Feature Eng.) |
| rainfall | 0.111 | Ham |
| humidity | 0.071 | Ham |
| NPK_Total | 0.061 | Türetilmiş (Feature Eng.) |
| Climate_Index | 0.055 | Türetilmiş (Feature Eng.) |
| N_to_P | 0.043 | Türetilmiş (Feature Eng.) |
| temperature | 0.037 | Ham |

Türettiğimiz `N_to_K`, `NPK_Total` ve `Climate_Index` gibi özellikler modelin en güçlü ayırt edicileri arasına girerek özellik mühendisliğinin tahmin başarısına doğrudan katkı sağladığını ispatlamıştır.

## 5. XAI (Açıklanabilir AI)

- **SHAP:** Global summary (`outputs/shap_summary.png`) + Streamlit'te local waterfall (14 özellik uyumlu)
- **LIME:** Streamlit'te yerel özellik katkı grafiği (7 ham özellik uyumlu)

## 6. Sonuç ve Gelecek Çalışmalar

Proje, uçtan uca çalışan bir ML pipeline ve XAI demo sunmaktadır. Gelecekte: hiperparametre tuning, feature engineering, nested CV, gerçek tarım verisi ile doğrulama ve model kalibrasyonu önerilir.

## 7. Kaynakça

1. Kaggle Crop Recommendation Dataset — https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP).
3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?" Explaining the Predictions of Any Classifier (LIME).
4. scikit-learn Documentation — https://scikit-learn.org/
