"""
Amaç: Proje genelinde kullanılan sabitleri merkezi olarak tanımlamak.

İçerik:
    - Dosya yolları (DATA_PATH, MODEL_PATH, OUTPUT_DIR)
    - Özellik ve hedef sütun isimleri
    - Train/test split oranı ve random_state
    - Model hiperparametreleri (sabit, tuning yok)

Bağımlılıklar:
    - pathlib (stdlib)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Crop_recommendation.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

METRICS_PATH = OUTPUT_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = OUTPUT_DIR / "roc_curve_ovr.png"
LAZYPREDICT_PATH = OUTPUT_DIR / "lazypredict_results.csv"
