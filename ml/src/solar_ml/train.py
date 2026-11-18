import hashlib
import json
import platform
import sys
import time
import warnings
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from solar_ml import cnn
from solar_ml.constants import CLASSES, HEIGHT, SEED, WIDTH
from solar_ml.dataset import load_arrays
from solar_ml.export import export_cnn, export_dense, forward, write_artifact

CNN_SEEDS = (42, 43, 44)
CNN_WIDTHS = (8, 16)
DEPLOY_SEED = 42
TIE_MARGIN = 0.01
BOOTSTRAP_RESAMPLES = 1000
NORMAL_CLASS = "No-Anomaly"


def protocol(epochs: int = cnn.RECIPE["epochs"]) -> dict:
    return {
        "registered_before_training": True,
        "candidates": {
            "logistic_regression": "LogisticRegression(C=1, max_iter=3000, lbfgs) on raw "
            "960 pixels, no normalization; baseline identical to the historical model",
            "cnn_w8": "SmallCNN width 8",
            "cnn_w16": "SmallCNN width 16",
        },
        "cnn_recipe": {**cnn.RECIPE, "epochs": epochs},
        "cnn_seeds": list(CNN_SEEDS),
        "summary_statistics": "mean and sample standard deviation (ddof=1) over seeds",
        "selection_metric": "highest mean validation F1 macro",
        "tie_rule": f"if cnn_w8 mean validation F1 macro is within {TIE_MARGIN} of cnn_w16, "
        "select cnn_w8 (cheaper inference)",
        "deployed_model": f"seed-{DEPLOY_SEED} run of the selected architecture "
        "(fixed in advance, not the best seed)",
        "test_policy": "fresh test evaluated exactly once, only for the selected model and the "
        "logistic regression baseline (reference)",
        "test_statistics": f"percentile bootstrap 95% CI, {BOOTSTRAP_RESAMPLES} resamples, "
        f"numpy default_rng({SEED})",
        "binary_derivation": f"anomaly = any predicted class other than {NORMAL_CLASS}",
    }


def summary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "auc_macro_ovr": float(
            roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
        ),
    }


def bootstrap_ci(y_true: np.ndarray, probabilities: np.ndarray, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    predicted = probabilities.argmax(axis=1)
    samples = {"accuracy": [], "f1_macro": [], "auc_macro_ovr": []}
    n = len(y_true)
    for _ in range(BOOTSTRAP_RESAMPLES):
        index = rng.integers(0, n, n)
        truth = y_true[index]
        if len(np.unique(truth)) != len(CLASSES):
            continue
        samples["accuracy"].append(accuracy_score(truth, predicted[index]))
        samples["f1_macro"].append(f1_score(truth, predicted[index], average="macro"))
        samples["auc_macro_ovr"].append(
            roc_auc_score(truth, probabilities[index], multi_class="ovr", average="macro")
        )
    return {
        key: {
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
            "resamples": len(values),
        }
        for key, values in samples.items()
    }


def full_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, records: list[dict] | None = None
) -> dict:
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=range(len(CLASSES)), zero_division=0
    )
    roc = {}
    for index, label in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve((y_true == index).astype(int), probabilities[:, index])
        roc[label] = {"auc": float(auc(fpr, tpr)), "fpr": fpr.tolist(), "tpr": tpr.tolist()}
    normal = CLASSES.index(NORMAL_CLASS)
    true_anomaly = y_true != normal
    predicted_anomaly = predicted != normal
    result = {
        "count": int(len(y_true)),
        **summary_metrics(y_true, probabilities),
        "ci95_bootstrap": bootstrap_ci(y_true, probabilities),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "auc_ovr": roc[label]["auc"],
            }
            for index, label in enumerate(CLASSES)
        },
        "binary_anomaly_vs_normal": {
            "sensitivity": float((predicted_anomaly & true_anomaly).sum() / true_anomaly.sum()),
            "specificity": float(
                (~predicted_anomaly & ~true_anomaly).sum() / (~true_anomaly).sum()
            ),
            "positives": int(true_anomaly.sum()),
            "negatives": int((~true_anomaly).sum()),
        },
        "confusion_matrix": confusion_matrix(
            y_true, predicted, labels=range(len(CLASSES))
        ).tolist(),
        "roc_curves": roc,
    }
    if records is not None:
        wrong = np.flatnonzero(predicted != y_true)
        wrong = sorted(wrong, key=lambda i: -probabilities[i, predicted[i]])
        result["misclassified"] = [
            {
                "image_id": records[i]["image_id"],
                "path": records[i]["path"],
                "true": CLASSES[y_true[i]],
                "predicted": CLASSES[predicted[i]],
                "probabilities": dict(
                    zip(CLASSES, [round(float(p), 6) for p in probabilities[i]], strict=True)
                ),
            }
            for i in wrong[:30]
        ]
        result["misclassified_total"] = int(len(wrong))
    return result


def select_architecture(candidates: dict) -> str:
    means = {
        name: model["validation_mean"]["f1_macro"]
        for name, model in candidates.items()
        if name.startswith("cnn_")
    }
    best = max(means, key=means.get)
    if best == "cnn_w16" and "cnn_w8" in means and means["cnn_w8"] >= means[best] - TIE_MARGIN:
        return "cnn_w8"
    return best


def plot_results(results: dict, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    test = results["selected_model"]["test"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(np.asarray(test["confusion_matrix"]), display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set(xlabel="Clase predicha", ylabel="Clase real")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for label, curve in test["roc_curves"].items():
        ax.plot(curve["fpr"], curve["tpr"], label=f"{label} (AUC {curve['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=180)
    plt.close(fig)

    names = list(results["models"])
    display = {
        "logistic_regression": "Regresión logística",
        "cnn_w8": "CNN w8",
        "cnn_w16": "CNN w16",
    }
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(names))
    for offset, key, title in ((-0.18, "accuracy", "Accuracy"), (0.18, "f1_macro", "F1 macro")):
        means = [results["models"][n]["validation_mean"][key] for n in names]
        errors = [results["models"][n]["validation_sd"][key] for n in names]
        ax.bar(x + offset, means, 0.36, yerr=errors, capsize=3, label=title)
    ax.set_xticks(x, [display[n] for n in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validación")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison.png", dpi=180)
    plt.close(fig)


def _mean_sd(runs: list[dict]) -> tuple[dict, dict]:
    keys = runs[0].keys()
    mean = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    sd = {k: float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0 for k in keys}
    return mean, sd


def train(
    archive: Path,
    manifest: dict,
    artifacts_dir: Path,
    figures_dir: Path | None = None,
    epochs: int = cnn.RECIPE["epochs"],
    seeds: tuple[int, ...] = CNN_SEEDS,
    widths: tuple[int, ...] = CNN_WIDTHS,
    arrays: dict | None = None,
) -> dict:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "metrics.json"
    registered = protocol(epochs)
    registered["cnn_seeds"] = list(seeds)
    metrics_path.write_text(
        json.dumps(
            {"status": "protocol registered; training not started", "protocol": registered},
            indent=2,
        ),
        encoding="utf-8",
    )
    arrays = arrays or load_arrays(archive, manifest)
    train_x, train_y = arrays["train"]
    valid_x, valid_y = arrays["validation"]
    test_x, test_y = arrays["test"]
    test_records = [item for item in manifest["records"] if item["split"] == "test"]
    results = {
        "status": "complete",
        "protocol": registered,
        "classes": list(CLASSES),
        "split_counts": {split: int(len(labels)) for split, (_, labels) in arrays.items()},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "models": {},
    }

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        baseline = LogisticRegression(C=1, max_iter=3000, random_state=SEED).fit(train_x, train_y)
    baseline_val = summary_metrics(valid_y, baseline.predict_proba(valid_x))
    results["models"]["logistic_regression"] = {
        "parameters": int(baseline.coef_.size + baseline.intercept_.size),
        "training_seconds": time.perf_counter() - start,
        "iterations": int(np.max(baseline.n_iter_)),
        "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "validation_runs": {"deterministic": baseline_val},
        "validation_mean": baseline_val,
        "validation_sd": {k: 0.0 for k in baseline_val},
    }

    deploy_models = {}
    for width in widths:
        name = f"cnn_w{width}"
        runs, times = {}, {}
        for seed in seeds:
            model, seconds = cnn.train_cnn(train_x, train_y, width, seed, epochs=epochs)
            runs[str(seed)] = summary_metrics(valid_y, cnn.predict_proba(model, valid_x))
            times[str(seed)] = seconds
            if seed == DEPLOY_SEED or (DEPLOY_SEED not in seeds and seed == seeds[0]):
                deploy_models[name] = model
        mean, sd = _mean_sd(list(runs.values()))
        results["models"][name] = {
            "parameters": cnn.parameter_count(deploy_models[name]),
            "training_seconds": times,
            "validation_runs": runs,
            "validation_mean": mean,
            "validation_sd": sd,
        }

    selected = select_architecture(results["models"])
    model = deploy_models[selected]
    probabilities = cnn.predict_proba(model, test_x)
    test = full_metrics(test_y, probabilities, test_records)
    baseline_test = full_metrics(test_y, baseline.predict_proba(test_x), test_records)
    summary_keys = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_macro_ovr")
    metrics_ref = {
        "split": "fresh test (never in the historical manifest), evaluated once",
        "count": test["count"],
        **{key: round(test[key], 4) for key in summary_keys},
        "ci95_bootstrap": {
            key: [round(v["low"], 4), round(v["high"], 4)]
            for key, v in test["ci95_bootstrap"].items()
        },
    }
    artifact = export_cnn(model, {"type": "per_image", "eps": cnn.PER_IMAGE_EPS}, metrics_ref)
    artifact_bytes = write_artifact(artifact, artifacts_dir / "model.json")
    exported = forward(artifact, test_x)
    baseline_artifact = export_dense(
        baseline, {key: round(baseline_test[key], 4) for key in summary_keys}
    )
    baseline_bytes = write_artifact(baseline_artifact, artifacts_dir / "baseline_model.json")
    baseline_parity = float(
        np.max(np.abs(forward(baseline_artifact, test_x) - baseline.predict_proba(test_x)))
    )

    results["selected_model"] = {
        "name": selected,
        "selection_reason": registered["selection_metric"] + "; " + registered["tie_rule"],
        "deployed_seed": DEPLOY_SEED if DEPLOY_SEED in seeds else seeds[0],
        "parameters": cnn.parameter_count(model),
        "artifact": "model.json",
        "artifact_bytes": artifact_bytes,
        "artifact_sha256": artifact["sha256"],
        "numpy_export_parity_max_abs_prob_error_full_test": float(
            np.max(np.abs(exported - probabilities))
        ),
        "numpy_export_prediction_agreement_full_test": float(
            np.mean(exported.argmax(1) == probabilities.argmax(1))
        ),
        "test": test,
    }
    results["baseline_reference"] = {
        "name": "logistic_regression",
        "artifact": "baseline_model.json",
        "artifact_bytes": baseline_bytes,
        "numpy_export_parity_max_abs_prob_error_full_test": baseline_parity,
        "test": baseline_test,
    }

    _write_reference(archive, artifacts_dir, test_records, test_x, test_y, exported, artifact)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if figures_dir is not None:
        plot_results(results, figures_dir)
    return results


def _write_reference(archive, artifacts_dir, test_records, test_x, test_y, probs, artifact):
    """First correctly classified demo image (2 per class, Cell-Multi preferred)."""
    order = ["Cell-Multi"] + [label for label in CLASSES if label != "Cell-Multi"]
    demo = [
        index
        for label in order
        for index in [i for i, r in enumerate(test_records) if r["label"] == label][:2]
    ]
    correct = [i for i in demo if probs[i].argmax() == test_y[i]]
    if not correct:
        correct = [i for i in range(len(test_y)) if probs[i].argmax() == test_y[i]]
    index = correct[0]
    record = test_records[index]
    with zipfile.ZipFile(archive) as source:
        checksum = hashlib.sha256(source.read(record["path"])).hexdigest()
    (artifacts_dir / "reference_prediction.json").write_text(
        json.dumps(
            {
                "image_id": record["image_id"],
                "label": record["label"],
                "split": "test",
                "source": f"ml/demo/{record['label']}_{record['image_id']}.jpg",
                "source_sha256": checksum,
                "preprocessing": "rounded_bt601_luma_255",
                "width": WIDTH,
                "height": HEIGHT,
                "pixels": test_x[index].tolist(),
                "expected_probabilities": dict(zip(CLASSES, probs[index].tolist(), strict=True)),
                "expected_class": CLASSES[int(probs[index].argmax())],
                "model_sha256": artifact["sha256"],
                "reference": "solar_ml.export.forward on artifacts/model.json (version 2.0.0)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
