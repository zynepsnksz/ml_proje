"""Model eğitimi, değerlendirme ve kaydetme."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import dill
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
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize

from src.config import (
    ALL_FEATURE_COLUMNS,
    CALIBRATION_CURVE_PATH,
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
from src.models.confidence import plot_calibration_curve, summarize_test_confidence
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
LAZYPREDICT_CV_FOLDS = 3


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def _select_tree_model(results: pd.DataFrame) -> str:
    """LazyPredict sonuçlarından SHAP TreeExplainer uyumlu ilk modeli seçer."""
    ranked = results.sort_values("F1 Score", ascending=False).index.tolist()
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
    """LazyPredict ile model seçimi; tek split yerine fold ortalaması kullanılır.

    Tek validation split model sıralamasını fold'a özgü şansa bırakır; 3-fold
    ortalama daha kararlı bir karşılaştırma sağlar. F1 macro, sınıf dengesizliğinde
    accuracy'ye göre daha adil bir sıralama metriğidir.
    """
    from src.preprocessing import FeatureEngineer, OutlierClipper

    # Stratified fold'lar azınlık sınıfların her turda temsil edilmesini korur.
    cv = StratifiedKFold(n_splits=LAZYPREDICT_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results: list[pd.DataFrame] = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_lp_train = X_train.iloc[train_idx]
        X_lp_val = X_train.iloc[val_idx]
        y_lp_train = y_train[train_idx]
        y_lp_val = y_train[val_idx]

        # LazyPredict ham veri kabul etmez; nihai pipeline ile tam aynı ön işleme
        # uygulanmazsa model karşılaştırması yanıltıcı olur.
        # Adım sırası: FeatureEngineer → SimpleImputer → OutlierClipper → StandardScaler
        # (build_pipeline() ile birebir eşleşir — imputer burada da zorunludur)
        from sklearn.impute import SimpleImputer

        engineer = FeatureEngineer()
        imputer = SimpleImputer(strategy="median")
        clipper = OutlierClipper()
        scaler = StandardScaler()

        X_lp_train_eng = engineer.transform(X_lp_train)
        X_lp_val_eng = engineer.transform(X_lp_val)

        # imputer ve clipper yalnızca train fold üzerinde fit edilir;
        # val fold'a aynı parametrelerle transform uygulanır (leakage yok).
        X_lp_train_imp = imputer.fit_transform(X_lp_train_eng)
        X_lp_val_imp = imputer.transform(X_lp_val_eng)

        clipper.fit(X_lp_train_imp)
        X_lp_train_clipped = clipper.transform(X_lp_train_imp)
        X_lp_val_clipped = clipper.transform(X_lp_val_imp)

        X_lp_train_scaled = scaler.fit_transform(X_lp_train_clipped)
        X_lp_val_scaled = scaler.transform(X_lp_val_clipped)

        lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, predictions=False)
        results, _ = lazy_clf.fit(X_lp_train_scaled, X_lp_val_scaled, y_lp_train, y_lp_val)
        fold_results.append(results)

    # Fold'lar arası ortalama, tek seferlik şanslı/şanssız split etkisini azaltır.
    metric_columns = [col for col in fold_results[0].columns if col != "Time Taken"]
    aggregated = fold_results[0].copy()
    for model in aggregated.index:
        for metric_col in metric_columns:
            scores = [
                df.loc[model, metric_col]
                for df in fold_results
                if model in df.index
            ]
            if scores:
                aggregated.loc[model, metric_col] = float(np.mean(scores))

    aggregated = aggregated.sort_values("F1 Score", ascending=False)
    # SHAP TreeExplainer yalnızca ağaç tabanlı modellerle uyumlu; XAI trade-off'u.
    best_model_name = _select_tree_model(aggregated)
    return best_model_name, aggregated


def _build_param_dist(classifier_name: str) -> dict[str, Any]:
    """Model tipine özel hiperparametre aralıkları.

    Sınırlar literatürde yaygın aralıklardan seçildi: çok sığ modeller underfitting,
    çok derin/agresif modeller overfitting riski taşır; orta bant genelleme için
    daha güvenilir bir arama alanı bırakır.
    """
    if classifier_name == "XGBClassifier":
        return {
            # Düşük learning_rate + yüksek n_estimators: daha yumuşak öğrenme eğrisi.
            "classifier__n_estimators": [50, 100, 200, 300],
            "classifier__max_depth": [3, 4, 5, 6, 7, 8],
            "classifier__learning_rate": [0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
            # subsample/colsample: ağaçlar arası çeşitlilik; ezberlemeyi sınırlar.
            "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__min_child_weight": [1, 2, 3, 5],
        }
    if classifier_name in ("RandomForestClassifier", "ExtraTreesClassifier"):
        return {
            "classifier__n_estimators": [50, 100, 200, 300, 400],
            # None derinliği overfitting'e açık; üst sınır 20 ile sınırlandı.
            "classifier__max_depth": [5, 8, 10, 12, 15, 20, None],
            # min_samples_* yaprak gürültüsünü filtreler, varyansı düşürür.
            "classifier__min_samples_split": [2, 3, 5, 10, 15],
            "classifier__min_samples_leaf": [1, 2, 4, 6, 8],
            "classifier__max_features": ["sqrt", "log2", None],
        }
    if classifier_name == "LGBMClassifier":
        return {
            "classifier__n_estimators": [50, 100, 200, 300],
            "classifier__max_depth": [3, 5, 7, 10, 15],
            "classifier__learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
            # num_leaves max_depth ile birlikte model kapasitesini dengeler.
            "classifier__num_leaves": [15, 31, 63, 127],
            "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        }
    if classifier_name == "DecisionTreeClassifier":
        return {
            "classifier__max_depth": [3, 5, 7, 10, 15, 20],
            "classifier__min_samples_split": [2, 3, 5, 10, 15],
            "classifier__min_samples_leaf": [1, 2, 4, 6, 8],
        }
    # Bilinmeyen sınıflandırıcılar için dar fallback; geniş grid gereksiz arama maliyeti yaratır.
    return {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [5, 8, 12, None],
    }


def tune_hyperparameters(
    pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """RandomizedSearchCV ile pipeline hiperparametrelerini optimize eder."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    classifier = pipeline.named_steps["classifier"]
    classifier_name = type(classifier).__name__
    param_dist = _build_param_dist(classifier_name)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_params = {}
    for key, value in search.best_params_.items():
        if value is None:
            best_params[key] = None
        elif isinstance(value, (np.integer, int)):
            best_params[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            best_params[key] = float(value)
        else:
            best_params[key] = value

    return search.best_estimator_, best_params


def nested_cross_validate_model(
    pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> dict[str, float]:
    """Nested CV ile tarafsız genelleme tahmini.

    Hiperparametre tuning ve skorlama aynı veride yapılırsa selection bias oluşur;
    dış fold tuning'i içeride, değerlendirmeyi dışarıda tutarak optimistik
    metrikleri engeller. cross_validation ile yan yana raporlanır: biri tuning
    sonrası modelin CV profili, diğeri tuning sürecinin kendisinin tarafsız skoru.

    Scoring tutarlılığı: iç döngü (inner_cv) ve dış değerlendirme metriği f1_macro
    olarak hizalandı — tune_hyperparameters() ve cross_validate_model() ile aynı
    kriter; çok-sınıflı sınıflandırmada accuracy'ye göre daha kararlı karşılaştırma
    sağlar.
    """
    classifier = pipeline.named_steps["classifier"]
    classifier_name = type(classifier).__name__
    param_dist = _build_param_dist(classifier_name)

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    outer_f1_scores: list[float] = []

    for train_idx, val_idx in outer_cv.split(X_train, y_train):
        X_outer_train = X_train.iloc[train_idx]
        X_outer_val = X_train.iloc[val_idx]
        y_outer_train = y_train[train_idx]
        y_outer_val = y_train[val_idx]

        # clone: önceki outer fold'un en iyi parametreleri sonraki fold'a sızmasın.
        # scoring="f1_macro": iç ve dış döngü aynı metriği optimize eder;
        # tutarsız metrik seçimi nested CV'nin bias-free özelliğini zayıflatır.
        search = RandomizedSearchCV(
            estimator=clone(pipeline),
            param_distributions=param_dist,
            n_iter=50,
            cv=inner_cv,
            scoring="f1_macro",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_outer_train, y_outer_train)
        y_pred = search.best_estimator_.predict(X_outer_val)
        outer_f1_scores.append(
            float(f1_score(y_outer_val, y_pred, average="macro", zero_division=0))
        )

    return {
        "f1_macro_mean": float(np.mean(outer_f1_scores)),
        "f1_macro_std": float(np.std(outer_f1_scores)),
    }


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
    """One-vs-Rest ROC eğrisini kaydeder; görsel karmaşayı önlemek için en düşük 3 sınıfı vurgular."""
    n_classes = len(class_labels)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    per_class = []

    # Önce tüm sınıfların AUC değerlerini hesaplayalım
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        per_class.append({
            "index": i,
            "class": class_labels[i],
            "roc_auc": float(roc_auc),
            "fpr": fpr,
            "tpr": tpr
        })

    # AUC değerlerine göre sıralayalım (en düşükler önce gelir)
    sorted_classes = sorted(per_class, key=lambda item: item["roc_auc"])
    lowest_3_indices = {item["index"] for item in sorted_classes[:3]}

    plt.figure(figsize=(10, 8))
    
    # Renkli çizilecek en düşük 3 sınıf için palet
    colors = ["#e74c3c", "#e67e22", "#f1c40f"]  # Kırmızı, Turuncu, Sarı tonları
    color_idx = 0

    for item in sorted_classes:
        i = item["index"]
        fpr = item["fpr"]
        tpr = item["tpr"]
        roc_auc = item["roc_auc"]
        
        if i in lowest_3_indices:
            color = colors[color_idx]
            color_idx += 1
            plt.plot(
                fpr, 
                tpr, 
                color=color, 
                lw=2.5, 
                zorder=3, 
                label=f"{item['class']} (En Düşük AUC = {roc_auc:.4f})"
            )
        else:
            plt.plot(
                fpr, 
                tpr, 
                color="#cbd5e1", 
                alpha=0.25, 
                lw=1, 
                zorder=1,
                label=None  # Lejantta kalabalık yapmaması için gizliyoruz
            )

    # Macro-average ROC hesaplama ve çizme
    all_fpr = np.unique(
        np.concatenate([sorted_classes[idx]["fpr"] for idx in range(n_classes)])
    )
    mean_tpr = np.zeros_like(all_fpr)
    for idx in range(n_classes):
        mean_tpr += np.interp(all_fpr, sorted_classes[idx]["fpr"], sorted_classes[idx]["tpr"])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    
    plt.plot(
        all_fpr, 
        mean_tpr, 
        color="#1e293b", 
        lw=3, 
        zorder=4, 
        label=f"Macro-average (AUC = {macro_auc:.4f})"
    )

    plt.plot([0, 1], [0, 1], "k--", lw=1, zorder=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve — One-vs-Rest (Vurgulanmış En Düşük 3 Sınıf)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close()
    
    # Sonuç listesini (sadece metadata ile) döndürelim
    return [{"class": item["class"], "roc_auc": item["roc_auc"]} for item in sorted_classes]


def plot_learning_curve_chart(pipeline, X_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, float]:
    """Learning curve grafiği üretir ve train/test skor farkını döndürür."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        cv=cv,
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
    
    # Özellik isimlerini pipeline yapısına göre belirle
    engineer = pipeline.named_steps.get("engineer", None)
    feature_names = ALL_FEATURE_COLUMNS if engineer is not None else FEATURE_COLUMNS
    
    order = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(
        [feature_names[i] for i in order],
        importances[order],
        color="#27ae60",
    )
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close()
    return dict(zip(feature_names, importances.tolist()))


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
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    lime_explainer,
) -> None:
    """Pipeline, LabelEncoder, LIME explainer, train-only UI metadata ve metadata ile artifact kaydeder."""
    train_feature_bounds = {
        col: (
            float(X_train[col].quantile(0.01)),
            float(X_train[col].quantile(0.99)),
        )
        for col in FEATURE_COLUMNS
    }

    train_labels = label_encoder.inverse_transform(y_train)
    train_profile = X_train[FEATURE_COLUMNS].copy()
    train_profile["label"] = train_labels
    crop_means = train_profile.groupby("label")[FEATURE_COLUMNS].mean()
    train_crop_scenarios = {
        str(crop): {col: float(crop_means.loc[crop, col]) for col in FEATURE_COLUMNS}
        for crop in crop_means.index
    }

    artifact = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "best_model_name": best_model_name,
        "lime_explainer": lime_explainer,
        "train_feature_bounds": train_feature_bounds,
        "train_crop_scenarios": train_crop_scenarios,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "feature_columns": FEATURE_COLUMNS,
            "test_accuracy": test_metrics.get("accuracy"),
            "test_f1_macro": test_metrics.get("f1_macro"),
        },
    }
    with open(MODEL_PATH, "wb") as f:
        dill.dump(artifact, f)


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

    nested_cv_metrics = nested_cross_validate_model(pipeline, X_train, y_train_enc)
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
        plot_summary(pipeline, X_train, SHAP_SUMMARY_PATH, class_names=label_encoder.classes_.tolist())

    lowest_auc_classes = roc_per_class[:3] if roc_per_class else []

    confidence_analysis: dict[str, Any] | None = None
    calibration_analysis: dict[str, Any] | None = None
    if test_eval["y_proba"] is not None:
        y_proba_arr = np.array(test_eval["y_proba"])
        y_true_arr = np.array(test_eval["y_true"])
        confidence_analysis = summarize_test_confidence(y_proba_arr)
        calibration_analysis = plot_calibration_curve(y_true_arr, y_proba_arr)

    from src.explainability.lime_analysis import create_lime_explainer
    from src.preprocessing import FeatureEngineer

    X_train_eng = FeatureEngineer().transform(X_train)
    lime_explainer = create_lime_explainer(
        X_train_eng,
        ALL_FEATURE_COLUMNS,
        label_encoder.classes_.tolist(),
    )

    metrics: dict[str, Any] = {
        "best_model": best_model_name,
        "best_params": best_params,
        "model_selection": {
            "method": "lazypredict_stratified_kfold",
            "cv_folds": LAZYPREDICT_CV_FOLDS,
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
        "nested_cv": nested_cv_metrics,
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
        "confidence_analysis": confidence_analysis,
        "calibration_analysis": calibration_analysis,
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
            "calibration_curve_path": relative_to_project(CALIBRATION_CURVE_PATH),
        },
    }

    save_model_artifact(
        pipeline,
        label_encoder,
        best_model_name,
        metrics["test_metrics"],
        X_train,
        y_train_enc,
        lime_explainer,
    )
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
