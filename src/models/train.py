"""Model eğitimi, değerlendirme ve kaydetme."""

import json
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

from src.config import (
    CONFUSION_MATRIX_PATH,
    CV_FOLDS,
    LAZYPREDICT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    ROC_CURVE_PATH,
)
from src.data.loader import get_feature_target, load_data, split_data
from src.preprocessing import (
    build_pipeline,
    create_classifier,
    encode_labels,
    fit_label_encoder,
)

TREE_BASED_MODELS = [
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "DecisionTreeClassifier",
    "XGBClassifier",
    "LGBMClassifier",
]
DEFAULT_TREE_MODEL = "RandomForestClassifier"
LAZYPREDICT_VAL_SIZE = 0.2


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def _select_tree_model(results: pd.DataFrame) -> str:
    """LazyPredict sonuçlarından SHAP TreeExplainer uyumlu ilk modeli seçer."""
    ranked = results.sort_values("Accuracy", ascending=False).index.tolist()
    for model_name in ranked:
        if model_name in TREE_BASED_MODELS:
            return model_name
    return DEFAULT_TREE_MODEL


def select_best_model_with_lazypredict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[str, pd.DataFrame]:
    """LazyPredict ile modelleri karşılaştırır, ağaç tabanlı en iyi modeli seçer.

    Model seçimi yalnızca eğitim setinin iç validation bölünmesi üzerinde yapılır.
    Test seti bu aşamada kullanılmaz (test set leakage önlenir).
    """
    X_lp_train, X_lp_val, y_lp_train, y_lp_val = train_test_split(
        X_train,
        y_train,
        test_size=LAZYPREDICT_VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    scaler = StandardScaler()
    X_lp_train_scaled = scaler.fit_transform(X_lp_train)
    X_lp_val_scaled = scaler.transform(X_lp_val)

    lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, predictions=False)
    results, _ = lazy_clf.fit(X_lp_train_scaled, X_lp_val_scaled, y_lp_train, y_lp_val)
    results = results.sort_values("Accuracy", ascending=False)
    best_model_name = _select_tree_model(results)
    return best_model_name, results


def cross_validate_model(
    pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> dict[str, dict[str, float]]:
    """5-fold stratified cross-validation metriklerini hesaplar."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    metrics = {}
    for metric in scoring:
        values = cv_results[f"test_{metric}"]
        metrics[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return metrics


def evaluate_on_test(
    pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder,
) -> dict[str, Any]:
    """Test seti üzerinde metrikleri hesaplar."""
    y_pred = pipeline.predict(X_test)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }

    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        )
        metrics["roc_auc_ovr_weighted"] = float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        )
    else:
        y_proba = None
        metrics["roc_auc_ovr_macro"] = None
        metrics["roc_auc_ovr_weighted"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["class_labels"] = label_encoder.classes_.tolist()
    metrics["y_true"] = y_test.tolist()
    metrics["y_pred"] = y_pred.tolist()
    metrics["y_proba"] = y_proba.tolist() if y_proba is not None else None
    return metrics


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, class_labels: list[str]) -> None:
    """Confusion matrix grafiğini kaydeder."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()


def plot_roc_curve_ovr(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    class_labels: list[str],
) -> None:
    """One-vs-Rest ROC eğrisini kaydeder."""
    n_classes = len(class_labels)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, alpha=0.35, label=f"{class_labels[i]} (AUC={roc_auc:.3f})")

    # Macro-average OvR
    all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:, i], y_proba[:, i])[0] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color="black", lw=2, label=f"Macro-average (AUC={macro_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — One-vs-Rest")
    plt.legend(loc="lower right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close()


def save_model_artifact(pipeline, label_encoder, best_model_name: str) -> None:
    """Pipeline ve LabelEncoder'ı birlikte kaydeder."""
    artifact = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "best_model_name": best_model_name,
    }
    joblib.dump(artifact, MODEL_PATH)


def save_metrics(metrics: dict[str, Any]) -> None:
    """Metrikleri JSON dosyasına yazar."""
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def train() -> dict[str, Any]:
    """Tam eğitim pipeline'ını çalıştırır."""
    _ensure_output_dir()

    df = load_data()
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    label_encoder = fit_label_encoder(y_train)
    y_train_enc = np.array(encode_labels(label_encoder, y_train))
    y_test_enc = np.array(encode_labels(label_encoder, y_test))

    best_model_name, lazy_results = select_best_model_with_lazypredict(
        X_train, y_train_enc
    )
    lazy_results.to_csv(LAZYPREDICT_PATH)

    classifier = create_classifier(best_model_name, random_state=RANDOM_STATE)
    pipeline = build_pipeline(classifier)
    pipeline.fit(X_train, y_train_enc)

    cv_metrics = cross_validate_model(pipeline, X_train, y_train_enc)
    test_eval = evaluate_on_test(pipeline, X_test, y_test_enc, label_encoder)

    y_pred = np.array(test_eval["y_pred"])
    plot_confusion_matrix(y_test_enc, y_pred, test_eval["class_labels"])

    if test_eval["y_proba"] is not None:
        plot_roc_curve_ovr(
            y_test_enc,
            np.array(test_eval["y_proba"]),
            test_eval["class_labels"],
        )

    metrics = {
        "best_model": best_model_name,
        "model_selection": {
            "method": "lazypredict_on_train_validation_split",
            "validation_size": LAZYPREDICT_VAL_SIZE,
            "tree_based_filter": TREE_BASED_MODELS,
            "default_fallback": DEFAULT_TREE_MODEL,
        },
        "lazypredict_top5": lazy_results.head(5).reset_index().to_dict(orient="records"),
        "cross_validation": cv_metrics,
        "test_metrics": {
            "accuracy": test_eval["accuracy"],
            "precision_macro": test_eval["precision_macro"],
            "recall_macro": test_eval["recall_macro"],
            "f1_macro": test_eval["f1_macro"],
            "roc_auc_ovr_macro": test_eval["roc_auc_ovr_macro"],
            "roc_auc_ovr_weighted": test_eval["roc_auc_ovr_weighted"],
        },
        "artifacts": {
            "model_path": str(MODEL_PATH),
            "metrics_path": str(METRICS_PATH),
            "confusion_matrix_path": str(CONFUSION_MATRIX_PATH),
            "roc_curve_path": str(ROC_CURVE_PATH),
            "lazypredict_path": str(LAZYPREDICT_PATH),
        },
    }

    save_model_artifact(pipeline, label_encoder, best_model_name)
    save_metrics(metrics)

    # JSON boyutunu küçük tutmak için geçici alanları çıkar
    test_eval.pop("y_true", None)
    test_eval.pop("y_pred", None)
    test_eval.pop("y_proba", None)

    print(f"Best model: {best_model_name}")
    print(f"Test accuracy: {metrics['test_metrics']['accuracy']:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    return metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
