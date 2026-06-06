"""StandardScaler pipeline ve LabelEncoder yardımcıları."""

import inspect

from lazypredict.Supervised import CLASSIFIERS
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
    """StandardScaler + classifier pipeline oluşturur."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
