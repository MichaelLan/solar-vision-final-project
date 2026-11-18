import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier

from solar_ml.constants import CLASSES, FEATURE_COUNT, HEIGHT, SEED, WIDTH
from solar_ml.dataset import load_arrays


def runtime_probabilities(artifact: dict, features: np.ndarray) -> np.ndarray:
    backend_src = Path(__file__).resolve().parents[3] / "backend" / "src"
    if not backend_src.is_dir():
        raise FileNotFoundError(f"Backend runtime unavailable: {backend_src}")
    sys.path.insert(0, str(backend_src))
    try:
        from infra.classifier import DenseClassifier

        prediction = DenseClassifier(artifact).predict(features.tolist())
        return np.asarray([prediction.probabilities[label] for label in CLASSES])
    finally:
        sys.path.remove(str(backend_src))


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=range(len(CLASSES)), zero_division=0
    )
    macro = precision_recall_fscore_support(y_true, predicted, average="macro", zero_division=0)
    roc = {}
    for index, label in enumerate(CLASSES):
        binary = (y_true == index).astype(int)
        if len(np.unique(binary)) != 2:
            roc[label] = {"auc": None, "fpr": [], "tpr": []}
        else:
            fpr, tpr, _ = roc_curve(binary, probabilities[:, index])
            roc[label] = {"auc": float(auc(fpr, tpr)), "fpr": fpr.tolist(), "tpr": tpr.tolist()}
    auc_values = [value["auc"] for value in roc.values() if value["auc"] is not None]
    return {
        "count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "auc_macro_ovr": float(np.mean(auc_values)) if len(auc_values) == len(CLASSES) else None,
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
        "confusion_matrix": confusion_matrix(
            y_true, predicted, labels=range(len(CLASSES))
        ).tolist(),
        "roc_curves": roc,
    }


def export_model(model: LogisticRegression | MLPClassifier, destination: Path) -> dict:
    if isinstance(model, LogisticRegression):
        layers = [
            {
                "weights": model.coef_.tolist(),
                "bias": model.intercept_.tolist(),
                "activation": "softmax",
            }
        ]
    else:
        layers = []
        for index, (weights, biases) in enumerate(
            zip(model.coefs_, model.intercepts_, strict=True)
        ):
            layers.append(
                {
                    "weights": weights.T.tolist(),
                    "bias": biases.tolist(),
                    "activation": "softmax" if index == len(model.coefs_) - 1 else "relu",
                }
            )
    artifact = {
        "version": "1.0.0",
        "classes": list(CLASSES),
        "input": {
            "width": WIDTH,
            "height": HEIGHT,
            "channels": 1,
            "normalization": "rounded_bt601_luma_255",
        },
        "layers": layers,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    return artifact


def infer_exported(artifact: dict, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (FEATURE_COUNT,):
        raise ValueError(f"Expected {FEATURE_COUNT} features")
    for layer in artifact["layers"]:
        values = np.asarray(layer["weights"]) @ values + np.asarray(layer["bias"])
        if layer["activation"] == "relu":
            values = np.maximum(values, 0)
        elif layer["activation"] == "softmax":
            values = np.exp(values - values.max())
            values /= values.sum()
        else:
            raise ValueError("Unknown activation")
    return values


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
        if curve["auc"] is not None:
            ax.plot(curve["fpr"], curve["tpr"], label=f"{label} (AUC {curve['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(results["models"])
    x = np.arange(len(names))
    ax.bar(
        x - 0.18,
        [results["models"][name]["validation"]["accuracy"] for name in names],
        0.36,
        label="Accuracy",
    )
    ax.bar(
        x + 0.18,
        [results["models"][name]["validation"]["f1_macro"] for name in names],
        0.36,
        label="F1 macro",
    )
    display_names = {"logistic_regression": "Regresión logística", "mlp_32": "MLP (32)"}
    ax.set_xticks(x, [display_names[name] for name in names])
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison.png", dpi=180)
    plt.close(fig)


def train(
    archive: Path, manifest: dict, artifacts_dir: Path, figures_dir: Path, seed: int = SEED
) -> dict:
    arrays = load_arrays(archive, manifest)
    train_x, train_y = arrays["train"]
    valid_x, valid_y = arrays["validation"]
    test_x, test_y = arrays["test"]
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=300, random_state=seed),
        "mlp_32": MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=60,
            early_stopping=False,
            random_state=seed,
        ),
    }
    results = {
        "seed": seed,
        "classes": list(CLASSES),
        "split_counts": {split: len(labels) for split, (_, labels) in arrays.items()},
        "configuration": {
            "logistic_max_iter": 300,
            "mlp_hidden_units": 32,
            "mlp_max_iter": 60,
            "batch_size": 64,
        },
        "models": {},
    }
    for name, model in candidates.items():
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_x, train_y)
        elapsed = time.perf_counter() - start
        results["models"][name] = {
            "training_seconds": elapsed,
            "iterations": int(model.n_iter_),
            "convergence_warning": any(
                issubclass(warning.category, ConvergenceWarning) for warning in caught
            ),
            "validation": metrics(valid_y, model.predict_proba(valid_x)),
        }
    selected = max(candidates, key=lambda name: results["models"][name]["validation"]["f1_macro"])
    model = candidates[selected]
    artifact = export_model(model, artifacts_dir / "model.json")
    parity_errors = []
    backend_parity_errors = []
    for features, expected in zip(test_x[:32], model.predict_proba(test_x[:32]), strict=True):
        parity_errors.append(float(np.max(np.abs(infer_exported(artifact, features) - expected))))
        backend_parity_errors.append(
            float(np.max(np.abs(runtime_probabilities(artifact, features) - expected)))
        )
    results["selected_model"] = {
        "name": selected,
        "test": metrics(test_y, model.predict_proba(test_x)),
        "export_parity_max_error_first_32_test": max(parity_errors),
        "backend_parity_max_error_first_32_test": max(backend_parity_errors),
        "artifact_bytes": (artifacts_dir / "model.json").stat().st_size,
    }
    if max(parity_errors) > 1e-5:
        raise ValueError(f"Export parity failed: {max(parity_errors)}")
    if max(backend_parity_errors) > 1e-5:
        raise ValueError(f"Backend parity failed: {max(backend_parity_errors)}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_results(results, figures_dir)
    return results
