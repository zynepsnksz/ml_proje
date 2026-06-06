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
| Accuracy | 100.00% | 99.26% ± 0.58% | 100.00% | **99.55%** |
| F1 Macro | 100.00% | 99.26% | 100.00% | **99.55%** |
| ROC AUC OvR Macro | — | — | — | **1.0000** |

### Baseline Karşılaştırma

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| DummyClassifier | 4.55% | 0.40% |
| Seçilen model (RandomForest) | 99.55% | 99.55% |

### Confusion Matrix Yorumu

440 test örneğinde **2 hata** (hata oranı %0.45):

| Gerçek | Tahmin | Adet |
|--------|--------|------|
| blackgram | maize | 1 |
| rice | jute | 1 |

Grafikler: `outputs/confusion_matrix.png`, `outputs/confusion_matrix_normalized.png`

### ROC-AUC Yorumu

ROC AUC OvR macro = **0.99999** — sınıflar neredeyse mükemmel ayrışıyor. En düşük sınıf AUC: **jute (0.9999)**. Bu profil, veri setinin yüksek ayrışabilirliğini gösterir; gerçek tarım verisinde bu kadar yüksek AUC beklenmemelidir.

### Overfitting Kontrolü

| Gösterge | Değer |
|----------|-------|
| Train accuracy | 100.00% |
| Test accuracy | 99.55% |
| Train-test gap | 0.45% |
| Learning curve gap (final) | 0.40% |
| CV train vs CV test (accuracy) | 100% vs 99.26% |

Train setinde tam fit (100%) görülüyor; ancak test ve CV skorları yakın olduğundan **ciddi overfitting yok**, veri setinin görece kolay ayrışabilir olduğu sonucu çıkar.

### Özellik Önemleri (RandomForest)

| Özellik | Önem |
|---------|------|
| rainfall | 0.230 |
| humidity | 0.224 |
| K | 0.175 |
| P | 0.151 |
| N | 0.096 |
| temperature | 0.072 |
| ph | 0.051 |

EDA bulgularıyla uyumlu: yağış ve nem güçlü ayırt ediciler.

## 5. XAI (Açıklanabilir AI)

- **SHAP:** Global summary (`outputs/shap_summary.png`) + Streamlit'te local waterfall
- **LIME:** Streamlit'te yerel özellik katkı grafiği

## 6. Sonuç ve Gelecek Çalışmalar

Proje, uçtan uca çalışan bir ML pipeline ve XAI demo sunmaktadır. Gelecekte: hiperparametre tuning, feature engineering, nested CV, gerçek tarım verisi ile doğrulama ve model kalibrasyonu önerilir.

## 7. Kaynakça

1. Kaggle Crop Recommendation Dataset — https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP).
3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?" Explaining the Predictions of Any Classifier (LIME).
4. scikit-learn Documentation — https://scikit-learn.org/
