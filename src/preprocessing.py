import inspect
import pandas as pd

from lazypredict.Supervised import CLASSIFIERS
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import FEATURE_COLUMNS, IQR_MULTIPLIER
from src.feature_engineering import add_engineered_features


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Eğitim ve tahmin verilerine sızıntı (leakage) olmadan yeni özellikleri ekleyen transformer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Giriş verisi DataFrame değilse (örn. numpy array), sütun isimleriyle DataFrame'e çevrilir
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        return add_engineered_features(X)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """EDA'daki IQR analizine dayalı Winsorization (kırpma) transformer'ı.

    Politika:
        1. EDA: 1.5×IQR ile aykırı değerler tespit edilir (silinmez).
        2. Modelleme: Eğitim setinde hesaplanan [Q1 - k·IQR, Q3 + k·IQR] sınırlarına
           clip uygulanır; test/tahmin aynı sınırları kullanır (leakage yok).
        3. StandardScaler, kırpılmış dağılım üzerinde fit edilir — tek başına
           StandardScaler aykırı değerleri "yönetmez"; ortalama/varyansı bozar.
    """

    def __init__(self, iqr_multiplier: float = IQR_MULTIPLIER):
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds_: dict[str, float] = {}
        self.upper_bounds_: dict[str, float] = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[col] = float(q1 - self.iqr_multiplier * iqr)
            self.upper_bounds_[col] = float(q3 + self.iqr_multiplier * iqr)
        return self

    def transform(self, X):
        X_copied = X.copy()
        if not isinstance(X_copied, pd.DataFrame):
            X_copied = pd.DataFrame(X_copied)
        for col in X_copied.columns:
            if col in self.lower_bounds_:
                X_copied[col] = X_copied[col].clip(
                    lower=self.lower_bounds_[col],
                    upper=self.upper_bounds_[col],
                )
        return X_copied


def fit_label_encoder(y) -> LabelEncoder:
    """LabelEncoder'ı hedef değişken üzerinde fit eder."""
    encoder = LabelEncoder()
    encoder.fit(y)
    return encoder


def encode_labels(encoder: LabelEncoder, y) -> list[int]:
    """String etiketleri sayısal forma çevirir."""
    return encoder.transform(y).tolist()


def decode_labels(encoder: LabelEncoder, y_encoded) -> list[str]:
    """Sayısal etiketleri string forma çevirir."""
    return encoder.inverse_transform(y_encoded)


def create_classifier(name: str, random_state: int = 42) -> ClassifierMixin:
    """LazyPredict sonucundaki model adından sklearn sınıflandırıcı üretir."""
    for est_name, est_cls in CLASSIFIERS:
        if est_name == name:
            params = {}
            if "random_state" in inspect.signature(est_cls).parameters:
                params["random_state"] = random_state
            return est_cls(**params)
    raise ValueError(f"Bilinmeyen sınıflandırıcı: {name}")


def build_pipeline(classifier: ClassifierMixin) -> Pipeline:
    """FeatureEngineer + imputer + OutlierClipper (IQR) + StandardScaler + classifier pipeline oluşturur."""
    return Pipeline(
        [
            ("engineer", FeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", OutlierClipper()),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
