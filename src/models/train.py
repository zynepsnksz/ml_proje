"""Model eğitimi, değerlendirme ve kaydetme."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

from src.config import (
    CONFUSION_MATRIX_NORMALIZED_PATH,
    CONFUSION_MATRIX_PATH,
    CV_FOLDS,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    LAZYPREDICT_PATH,
    LEARNING_CURVE_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    ROC_CURVE_PATH,
    SHAP_SUMMARY_PATH,
    TARGET_COLUMN,
    relative_to_project,
)
from src.data.loader import get_feature_target, load_data, split_data
from src.explainability.shap_analysis import plot_summary, supports_tree_explainer
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
PARAM_DIST = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [5, 10, 15, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}


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


def build_split_report(y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    """Sınıf başına train/test örnek sayılarını raporlar."""
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()
    per_class = []
    for label in sorted(y_train.unique()):
        per_class.append(
            {
                "label": label,
                "train": int(train_counts.get(label, 0)),
                "test": int(test_counts.get(label, 0)),
            }
        )
    return {
        "total_train": int(len(y_train)),
        "total_test": int(len(y_test)),
        "per_class": per_class,
    }


def select_best_model_with_lazypredict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[str, pd.DataFrame]:
    """LazyPredict ile modelleri karşılaştırır, ağaç tabanlı en iyi modeli seçer."""
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


def tune_hyperparameters(
    pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """RandomizedSearchCV ile pipeline hiperparametrelerini optimize eder."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DIST,
        n_iter=10,
        cv=cv,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_params = {
        key: (None if value is None else int(value) if isinstance(value, np.integer) else value)
        for key, value in search.best_params_.items()
    }
    return search.best_estimator_, best_params


def cross_validate_model(
    pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """5-fold stratified CV; train ve test fold skorlarını döndürür."""
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
        return_train_score=True,
    )

    metrics: dict[str, Any] = {}
    for metric in scoring:
        test_values = cv_results[f"test_{metric}"]
        train_values = cv_results[f"train_{metric}"]
        metrics[metric] = {
            "train_mean": float(np.mean(train_values)),
            "train_std": float(np.std(train_values)),
            "test_mean": float(np.mean(test_values)),
            "test_std": float(np.std(test_values)),
        }
    return metrics


def evaluate_on_split(
    pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder,
    split_name: str = "test",
) -> dict[str, Any]:
    """Verilen split üzerinde metrikleri hesaplar."""
    y_pred = pipeline.predict(X)
    metrics: dict[str, Any] = {
        "split": split_name,
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
    }

    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        y_proba = pipeline.predict_proba(X)
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
        )
        metrics["roc_auc_ovr_weighted"] = float(
            roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
        )
        metrics["per_class_roc_auc"] = _per_class_roc_auc(y, y_proba, label_encoder.classes_)
    else:
        y_proba = None

    metrics["classification_report"] = classification_report(
        y,
        y_pred,
        target_names=label_encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )
    metrics["confusion_matrix"] = confusion_matrix(y, y_pred).tolist()
    metrics["class_labels"] = label_encoder.classes_.tolist()
    metrics["y_true"] = y.tolist()
    metrics["y_pred"] = y_pred.tolist()
    metrics["y_proba"] = y_proba.tolist() if y_proba is not None else None
    return metrics


def _per_class_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Sınıf bazlı OvR ROC AUC değerlerini hesaplar."""
    n_classes = len(class_labels)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    results = []
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        results.append({"class": str(label), "roc_auc": float(auc(fpr, tpr))})
    return sorted(results, key=lambda item: item["roc_auc"])


def analyze_misclassifications(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    """Yanlış sınıflandırma çiftlerini ve örnek sayılarını çıkarır."""
    cm = confusion_matrix(y_true, y_pred)
    pairs = []
    for i, actual in enumerate(class_labels):
        for j, predicted in enumerate(class_labels):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "actual": actual,
                        "predicted": predicted,
                        "count": int(cm[i, j]),
                    }
                )
    pairs.sort(key=lambda item: item["count"], reverse=True)
    total_errors = int(np.sum(y_true != y_pred))
    return {
        "total_errors": total_errors,
        "total_samples": int(len(y_true)),
        "error_rate": float(total_errors / len(y_true)) if len(y_true) else 0.0,
        "top_confusion_pairs": pairs[:10],
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    path,
    *,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """Confusion matrix grafiğini kaydeder."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmax = None

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc_curve_ovr(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    class_labels: list[str],
) -> list[dict[str, Any]]:
    """One-vs-Rest ROC eğrisini kaydeder; sınıf bazlı AUC listesi döndürür."""
    n_classes = len(class_labels)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    per_class = []

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        per_class.append({"class": class_labels[i], "roc_auc": float(roc_auc)})
        plt.plot(fpr, tpr, alpha=0.35, label=f"{class_labels[i]} (AUC={roc_auc:.3f})")

    all_fpr = np.unique(
        np.concatenate([roc_curve(y_bin[:, i], y_proba[:, i])[0] for i in range(n_classes)])
    )
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
    return sorted(per_class, key=lambda item: item["roc_auc"])


def plot_learning_curve_chart(pipeline, X_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, float]:
    """Learning curve grafiği üretir ve train/test skor farkını döndürür."""
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8),
        shuffle=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    plt.figure(figsize=(9, 6))
    plt.plot(train_sizes, train_mean, "o-", color="#2ecc71", label="Train")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2ecc71")
    plt.plot(train_sizes, test_mean, "o-", color="#3498db", label="CV Test")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color="#3498db")
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LEARNING_CURVE_PATH, dpi=150)
    plt.close()

    return {
        "final_train_mean": float(train_mean[-1]),
        "final_cv_mean": float(test_mean[-1]),
        "gap": float(train_mean[-1] - test_mean[-1]),
    }


def plot_feature_importance(pipeline, class_labels: list[str]) -> dict[str, float] | None:
    """Ağaç tabanlı model için özellik önem grafiği."""
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return None

    importances = classifier.feature_importances_
    order = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    plt.barh(
        [FEATURE_COLUMNS[i] for i in order],
        importances[order],
        color="#27ae60",
    )
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close()
    return dict(zip(FEATURE_COLUMNS, importances.tolist()))


def plot_model_comparison(lazy_results: pd.DataFrame, top_n: int = 10) -> None:
    """LazyPredict top-N accuracy bar chart."""
    top = lazy_results.head(top_n).reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, y="Model", x="Accuracy", hue="Model", palette="viridis", legend=False)
    plt.xlim(0, 1.05)
    plt.title(f"LazyPredict Top-{top_n} Model Comparison")
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PATH, dpi=150)
    plt.close()


def extract_baseline_metrics(lazy_results: pd.DataFrame) -> dict[str, float] | None:
    """DummyClassifier baseline metriklerini LazyPredict sonuçlarından çıkarır."""
    if "DummyClassifier" not in lazy_results.index:
        return None
    row = lazy_results.loc["DummyClassifier"]
    return {
        "accuracy": float(row["Accuracy"]),
        "balanced_accuracy": float(row["Balanced Accuracy"]),
        "f1_macro": float(row["F1 Score"]),
    }


def save_model_artifact(
    pipeline,
    label_encoder,
    best_model_name: str,
    test_metrics: dict[str, Any],
) -> None:
    """Pipeline, LabelEncoder ve metadata ile artifact kaydeder."""
    artifact = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "best_model_name": best_model_name,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "feature_columns": FEATURE_COLUMNS,
            "test_accuracy": test_metrics.get("accuracy"),
            "test_f1_macro": test_metrics.get("f1_macro"),
        },
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
    split_report = build_split_report(y_train, y_test)

    label_encoder = fit_label_encoder(y_train)
    y_train_enc = np.array(encode_labels(label_encoder, y_train))
    y_test_enc = np.array(encode_labels(label_encoder, y_test))

    best_model_name, lazy_results = select_best_model_with_lazypredict(X_train, y_train_enc)
    lazy_results.to_csv(LAZYPREDICT_PATH)
    baseline_metrics = extract_baseline_metrics(lazy_results)

    classifier = create_classifier(best_model_name, random_state=RANDOM_STATE)
    pipeline = build_pipeline(classifier)
    pipeline, best_params = tune_hyperparameters(pipeline, X_train, y_train_enc)

    cv_metrics = cross_validate_model(pipeline, X_train, y_train_enc)
    train_eval = evaluate_on_split(pipeline, X_train, y_train_enc, label_encoder, split_name="train")
    test_eval = evaluate_on_split(pipeline, X_test, y_test_enc, label_encoder, split_name="test")

    y_pred = np.array(test_eval["y_pred"])
    misclassification = analyze_misclassifications(
        y_test_enc, y_pred, test_eval["class_labels"]
    )

    plot_confusion_matrix(
        y_test_enc, y_pred, test_eval["class_labels"], CONFUSION_MATRIX_PATH, title="Confusion Matrix"
    )
    plot_confusion_matrix(
        y_test_enc,
        y_pred,
        test_eval["class_labels"],
        CONFUSION_MATRIX_NORMALIZED_PATH,
        normalize=True,
        title="Normalized Confusion Matrix",
    )

    roc_per_class = []
    if test_eval["y_proba"] is not None:
        roc_per_class = plot_roc_curve_ovr(
            y_test_enc,
            np.array(test_eval["y_proba"]),
            test_eval["class_labels"],
        )

    learning_curve_stats = plot_learning_curve_chart(pipeline, X_train, y_train_enc)
    feature_importance = plot_feature_importance(pipeline, test_eval["class_labels"])
    plot_model_comparison(lazy_results)

    if supports_tree_explainer(pipeline):
        plot_summary(pipeline, X_train, SHAP_SUMMARY_PATH)

    lowest_auc_classes = roc_per_class[:3] if roc_per_class else []

    metrics: dict[str, Any] = {
        "best_model": best_model_name,
        "best_params": best_params,
        "model_selection": {
            "method": "lazypredict_on_train_validation_split",
            "validation_size": LAZYPREDICT_VAL_SIZE,
            "tree_based_filter": TREE_BASED_MODELS,
            "default_fallback": DEFAULT_TREE_MODEL,
            "rationale": (
                "Ağaç tabanlı modeller SHAP TreeExplainer ile uyumludur. "
                "GaussianNB/QDA validation'da eşit/üstün olsa da XAI trade-off nedeniyle elenir."
            ),
        },
        "split_report": split_report,
        "baseline_dummy_classifier": baseline_metrics,
        "lazypredict_top5": lazy_results.head(5).reset_index().to_dict(orient="records"),
        "cross_validation": cv_metrics,
        "train_metrics": {
            "accuracy": train_eval["accuracy"],
            "balanced_accuracy": train_eval["balanced_accuracy"],
            "precision_macro": train_eval["precision_macro"],
            "recall_macro": train_eval["recall_macro"],
            "f1_macro": train_eval["f1_macro"],
        },
        "test_metrics": {
            "accuracy": test_eval["accuracy"],
            "balanced_accuracy": test_eval["balanced_accuracy"],
            "precision_macro": test_eval["precision_macro"],
            "recall_macro": test_eval["recall_macro"],
            "f1_macro": test_eval["f1_macro"],
            "roc_auc_ovr_macro": test_eval["roc_auc_ovr_macro"],
            "roc_auc_ovr_weighted": test_eval["roc_auc_ovr_weighted"],
        },
        "train_test_gap": {
            "accuracy": float(train_eval["accuracy"] - test_eval["accuracy"]),
            "f1_macro": float(train_eval["f1_macro"] - test_eval["f1_macro"]),
        },
        "learning_curve": learning_curve_stats,
        "classification_report": test_eval["classification_report"],
        "confusion_matrix": test_eval["confusion_matrix"],
        "misclassification_analysis": misclassification,
        "lowest_roc_auc_classes": lowest_auc_classes,
        "feature_importance": feature_importance,
        "artifacts": {
            "model_path": relative_to_project(MODEL_PATH),
            "metrics_path": relative_to_project(METRICS_PATH),
            "confusion_matrix_path": relative_to_project(CONFUSION_MATRIX_PATH),
            "confusion_matrix_normalized_path": relative_to_project(CONFUSION_MATRIX_NORMALIZED_PATH),
            "roc_curve_path": relative_to_project(ROC_CURVE_PATH),
            "learning_curve_path": relative_to_project(LEARNING_CURVE_PATH),
            "feature_importance_path": relative_to_project(FEATURE_IMPORTANCE_PATH),
            "shap_summary_path": relative_to_project(SHAP_SUMMARY_PATH),
            "model_comparison_path": relative_to_project(MODEL_COMPARISON_PATH),
            "lazypredict_path": relative_to_project(LAZYPREDICT_PATH),
        },
    }

    save_model_artifact(pipeline, label_encoder, best_model_name, metrics["test_metrics"])
    save_metrics(metrics)

    print(f"Best model: {best_model_name}")
    print(f"Train accuracy: {metrics['train_metrics']['accuracy']:.4f}")
    print(f"Test accuracy: {metrics['test_metrics']['accuracy']:.4f}")
    print(f"Train-test accuracy gap: {metrics['train_test_gap']['accuracy']:.4f}")
    print(f"Total misclassifications: {misclassification['total_errors']}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    return metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
