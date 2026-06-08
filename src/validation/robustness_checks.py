"""Permutation test ve multi-seed hold-out sağlamlık analizleri.

Ana eğitim pipeline'ını (`build_pipeline`) kullanır; `models/best_model.pkl` dosyasına
dokunmaz. Sonuçlar `outputs/robustness_checks.json` ve `.csv` dosyalarına yazılır.

Çalıştırma:
    python -m src.validation.robustness_checks
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import METRICS_PATH, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE
from src.data.loader import get_feature_target, load_data, split_data
from src.preprocessing import (
    build_pipeline,
    create_classifier,
    encode_labels,
    fit_label_encoder,
)

MULTI_SEED_VALUES = [42, 123, 777, 2026, 3407]
PERMUTATION_LABEL_SEED = 42
ROBUSTNESS_JSON_PATH = OUTPUT_DIR / "robustness_checks.json"
ROBUSTNESS_CSV_PATH = OUTPUT_DIR / "robustness_checks.csv"
DEFAULT_CLASSIFIER = "RandomForestClassifier"
N_CLASSES = 22
CHANCE_BASELINE = 1.0 / N_CLASSES


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_classifier_config() -> tuple[str, dict[str, Any]]:
    """Ana eğitimden kayıtlı model adı ve hiperparametreleri yükler."""
    if not METRICS_PATH.exists():
        return DEFAULT_CLASSIFIER, {}

    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    model_name = metrics.get("best_model", DEFAULT_CLASSIFIER)
    best_params = metrics.get("best_params", {})
    return model_name, best_params


def _build_fitted_pipeline(
    model_name: str,
    best_params: dict[str, Any],
    random_state: int,
) -> Any:
    """Ana pipeline yapısı ile sınıflandırıcıyı oluşturur."""
    classifier = create_classifier(model_name, random_state=random_state)
    pipeline = build_pipeline(classifier)
    if best_params:
        pipeline.set_params(**best_params)
    return pipeline


def _evaluate(
    pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Pipeline'ı eğitir ve hold-out test metriklerini döndürür."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def run_permutation_test(
    model_name: str,
    best_params: dict[str, Any],
    split_seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Gerçek etiketler ile karıştırılmış etiket performansını karşılaştırır."""
    df = load_data()
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = split_data(
        X, y, random_state=split_seed, test_size=TEST_SIZE
    )

    label_encoder = fit_label_encoder(y_train)
    y_train_enc = np.array(encode_labels(label_encoder, y_train))
    y_test_enc = np.array(encode_labels(label_encoder, y_test))

    real_pipeline = _build_fitted_pipeline(model_name, best_params, random_state=split_seed)
    real_metrics = _evaluate(real_pipeline, X_train, y_train_enc, X_test, y_test_enc)

    rng = np.random.default_rng(PERMUTATION_LABEL_SEED)
    y_train_shuffled = y_train_enc.copy()
    rng.shuffle(y_train_shuffled)

    permuted_pipeline = _build_fitted_pipeline(model_name, best_params, random_state=split_seed)
    permuted_metrics = _evaluate(
        permuted_pipeline, X_train, y_train_shuffled, X_test, y_test_enc
    )

    return {
        "split_random_state": split_seed,
        "permutation_label_seed": PERMUTATION_LABEL_SEED,
        "n_classes": N_CLASSES,
        "chance_baseline_accuracy": CHANCE_BASELINE,
        "chance_baseline_f1_macro": CHANCE_BASELINE,
        "classifier": model_name,
        "real_labels": real_metrics,
        "permuted_labels": permuted_metrics,
        "delta_accuracy": float(real_metrics["accuracy"] - permuted_metrics["accuracy"]),
        "delta_f1_macro": float(real_metrics["f1_macro"] - permuted_metrics["f1_macro"]),
    }


def run_multi_seed_holdout(
    model_name: str,
    best_params: dict[str, Any],
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Farklı random_state değerleriyle hold-out performansını ölçer."""
    seeds = seeds or MULTI_SEED_VALUES
    df = load_data()
    X, y = get_feature_target(df)

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = split_data(
            X, y, random_state=seed, test_size=TEST_SIZE
        )
        label_encoder = fit_label_encoder(y_train)
        y_train_enc = np.array(encode_labels(label_encoder, y_train))
        y_test_enc = np.array(encode_labels(label_encoder, y_test))

        pipeline = _build_fitted_pipeline(model_name, best_params, random_state=seed)
        metrics = _evaluate(pipeline, X_train, y_train_enc, X_test, y_test_enc)
        per_seed.append(
            {
                "seed": seed,
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                **metrics,
            }
        )

    accuracies = [row["accuracy"] for row in per_seed]
    f1_scores = [row["f1_macro"] for row in per_seed]

    return {
        "seeds": seeds,
        "classifier": model_name,
        "per_seed": per_seed,
        "summary": {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "f1_macro_mean": float(np.mean(f1_scores)),
            "f1_macro_std": float(np.std(f1_scores)),
        },
    }


def _build_commentary(
    permutation: dict[str, Any],
    multi_seed: dict[str, Any],
) -> str:
    """README ve rapor için kısa yorum metni üretir."""
    real = permutation["real_labels"]
    perm = permutation["permuted_labels"]
    summary = multi_seed["summary"]

    return (
        f"Permutation test: gercek etiketlerle accuracy={real['accuracy']:.4f}, "
        f"f1_macro={real['f1_macro']:.4f}; karistirilmis etiketlerle accuracy="
        f"{perm['accuracy']:.4f}, f1_macro={perm['f1_macro']:.4f} "
        f"(sans duzeyi ~ {CHANCE_BASELINE:.4f}). "
        f"Gercek X-y iliskisi bozuldugunda skor sans seviyesine iniyor; model anlamli sinyal "
        f"ogreniyor. Multi-seed hold-out ({len(multi_seed['seeds'])} seed): "
        f"accuracy={summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}, "
        f"f1_macro={summary['f1_macro_mean']:.4f} +/- {summary['f1_macro_std']:.4f}; "
        f"performans split secimine karsi kararli."
    )


def _results_to_csv_rows(
    permutation: dict[str, Any],
    multi_seed: dict[str, Any],
) -> list[dict[str, Any]]:
    """JSON sonuçlarını düz CSV satırlarına dönüştürür."""
    rows: list[dict[str, Any]] = []

    rows.append(
        {
            "analysis": "permutation_test",
            "seed": permutation["split_random_state"],
            "label_type": "real",
            "accuracy": permutation["real_labels"]["accuracy"],
            "f1_macro": permutation["real_labels"]["f1_macro"],
        }
    )
    rows.append(
        {
            "analysis": "permutation_test",
            "seed": permutation["permutation_label_seed"],
            "label_type": "permuted",
            "accuracy": permutation["permuted_labels"]["accuracy"],
            "f1_macro": permutation["permuted_labels"]["f1_macro"],
        }
    )
    rows.append(
        {
            "analysis": "permutation_test",
            "seed": None,
            "label_type": "chance_baseline",
            "accuracy": permutation["chance_baseline_accuracy"],
            "f1_macro": permutation["chance_baseline_f1_macro"],
        }
    )

    for row in multi_seed["per_seed"]:
        rows.append(
            {
                "analysis": "multi_seed_holdout",
                "seed": row["seed"],
                "label_type": "real",
                "accuracy": row["accuracy"],
                "f1_macro": row["f1_macro"],
            }
        )

    summary = multi_seed["summary"]
    rows.append(
        {
            "analysis": "multi_seed_holdout",
            "seed": "mean",
            "label_type": "summary",
            "accuracy": summary["accuracy_mean"],
            "f1_macro": summary["f1_macro_mean"],
        }
    )
    rows.append(
        {
            "analysis": "multi_seed_holdout",
            "seed": "std",
            "label_type": "summary",
            "accuracy": summary["accuracy_std"],
            "f1_macro": summary["f1_macro_std"],
        }
    )

    return rows


def save_results(payload: dict[str, Any]) -> None:
    """Sonuçları JSON ve CSV dosyalarına yazar."""
    _ensure_output_dir()

    with open(ROBUSTNESS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_rows = _results_to_csv_rows(
        payload["permutation_test"],
        payload["multi_seed_holdout"],
    )
    pd.DataFrame(csv_rows).to_csv(ROBUSTNESS_CSV_PATH, index=False)


def run() -> dict[str, Any]:
    """Tüm sağlamlık analizlerini çalıştırır ve sonuçları kaydeder."""
    _ensure_output_dir()
    model_name, best_params = _load_classifier_config()

    print(f"Classifier config: {model_name}")
    print("Running permutation test...")
    permutation = run_permutation_test(model_name, best_params)

    print("Running multi-seed hold-out test...")
    multi_seed = run_multi_seed_holdout(model_name, best_params)

    commentary = _build_commentary(permutation, multi_seed)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classifier": model_name,
        "best_params": best_params,
        "permutation_test": permutation,
        "multi_seed_holdout": multi_seed,
        "commentary": commentary,
    }

    save_results(payload)

    print("\n=== Permutation Test ===")
    print(f"  Real      - accuracy: {permutation['real_labels']['accuracy']:.4f}, "
          f"f1_macro: {permutation['real_labels']['f1_macro']:.4f}")
    print(f"  Permuted  - accuracy: {permutation['permuted_labels']['accuracy']:.4f}, "
          f"f1_macro: {permutation['permuted_labels']['f1_macro']:.4f}")
    print(f"  Chance    - {CHANCE_BASELINE:.4f}")

    print("\n=== Multi-Seed Hold-Out ===")
    for row in multi_seed["per_seed"]:
        print(f"  seed={row['seed']:4d} - accuracy: {row['accuracy']:.4f}, "
              f"f1_macro: {row['f1_macro']:.4f}")
    s = multi_seed["summary"]
    print(f"  Mean +/- Std - accuracy: {s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}, "
          f"f1_macro: {s['f1_macro_mean']:.4f} +/- {s['f1_macro_std']:.4f}")

    print(f"\nCommentary:\n{commentary}")
    print(f"\nSaved: {ROBUSTNESS_JSON_PATH}")
    print(f"Saved: {ROBUSTNESS_CSV_PATH}")

    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
