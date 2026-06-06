"""
Proje genelinde kullanılan sabitler.

İçerik:
    - Dosya yolları (DATA_PATH, MODEL_PATH, OUTPUT_DIR)
    - Özellik ve hedef sütun isimleri
    - Train/test split oranı, random_state, CV fold sayısı
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Crop_recommendation.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

METRICS_PATH = OUTPUT_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_NORMALIZED_PATH = OUTPUT_DIR / "confusion_matrix_normalized.png"
ROC_CURVE_PATH = OUTPUT_DIR / "roc_curve_ovr.png"
LEARNING_CURVE_PATH = OUTPUT_DIR / "learning_curve.png"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.png"
SHAP_SUMMARY_PATH = OUTPUT_DIR / "shap_summary.png"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison.png"
LAZYPREDICT_PATH = OUTPUT_DIR / "lazypredict_results.csv"


def relative_to_project(path: Path) -> str:
    """Mutlak yolu proje köküne göre relative string'e çevirir."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name
