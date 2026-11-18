"""Generate the data-dependent parts of the report from the experiment artifacts.

Default mode uses only the standard library and writes ``report/results.typ``: every number
quoted in ``main.typ`` comes from there, and the qualitative claims of the text are checked
against the artifacts (the script fails if a claim stops being true).

``--figures`` additionally renders the report figures into ``report/figures/`` (requires
matplotlib and Pillow, e.g. the ``ml`` environment). Confusion matrices and ROC curves are
drawn from ``artifacts/metrics.json``; image examples are read from the original ZIP.
"""

import argparse
import json
import math
import random
import re
import zipfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parent / "artifacts"
FIGURES = ROOT / "figures"
OUTPUT = ROOT / "results.typ"
DEFAULT_ARCHIVE = ROOT.parent / "dataset" / "2020-02-14_InfraredSolarModules.zip"
CLASSES = ("No-Anomaly", "Cell", "Cell-Multi", "Diode")
NNBSP = " "  # narrow no-break space: thousands separator and number-unit glue


def load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def dec(value: float, places: int = 3) -> str:
    """Spanish decimal comma with half-up rounding (0.7567 -> 0,757)."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite metric")
    quantum = Decimal(1).scaleb(-places)
    return str(Decimal(str(result)).quantize(quantum, rounding=ROUND_HALF_UP)).replace(".", ",")


def integer(value: int) -> str:
    """Spanish style: no separator up to four digits, narrow no-break space from 10 000."""
    value = int(value)
    if abs(value) < 10_000:
        return str(value)
    return f"{value:,}".replace(",", NNBSP)


def ci(block: dict) -> str:
    return f"[{dec(block['low'])}; {dec(block['high'])}]"


def sci(value: float) -> tuple[str, str]:
    mantissa, exponent = f"{value:.1e}".split("e")
    return mantissa.replace(".", ","), str(int(exponent))


def typst_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def recipe_value(text: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=([0-9.e-]+)", text)
    if not match:
        raise ValueError(f"Recipe entry not found: {key} in {text!r}")
    return match.group(1)


def check(condition: bool, claim: str) -> None:
    if not condition:
        raise ValueError(f"Artifact no longer supports the report claim: {claim}")


def build_values(metrics: dict, manifest: dict, info: dict, historical: dict) -> dict:
    protocol = metrics["protocol"]
    recipe = protocol["cnn_recipe"]
    models = metrics["models"]
    chosen = metrics["selected_model"]
    test = chosen["test"]
    base = metrics["baseline_reference"]["test"]
    lr_val = models["logistic_regression"]["validation_mean"]
    w8, w16 = models["cnn_w8"], models["cnn_w16"]
    per = test["per_class"]
    cm = test["confusion_matrix"]
    split = manifest["split_counts_by_class"][CLASSES[0]]
    totals = manifest["split_totals"]
    dedup = manifest["dedup"]
    rejected = manifest["rejected_near_duplicates_by_stage_and_class"]
    binary = test["binary_anomaly_vs_normal"]
    base_binary = base["binary_anomaly_vs_normal"]
    original = info["original_counts"]
    four_total = sum(original[c] for c in CLASSES)

    # Claims made in the prose of main.typ.
    check(chosen["name"] == "cnn_w16", "the selected model is CNN w16")
    check(
        w16["validation_mean"]["f1_macro"] - w8["validation_mean"]["f1_macro"]
        > float(re.search(r"within ([0-9.]+)", protocol["tie_rule"]).group(1)),
        "w16 beats w8 by more than the tie margin",
    )
    check(
        all(manifest["split_counts_by_class"][c] == split for c in CLASSES),
        "balanced splits",
    )
    check(manifest["old_manifest_excluded_from_test"], "fresh test")
    check(manifest["old_manifest_images_by_split"]["test"] == 0, "fresh test")
    f1_order = sorted(CLASSES, key=lambda c: per[c]["f1"])
    check(f1_order == ["Cell-Multi", "Cell", "No-Anomaly", "Diode"], "per-class F1 ordering")
    index = {c: i for i, c in enumerate(CLASSES)}
    off_diagonal = {
        (CLASSES[i], CLASSES[j]): cm[i][j] for i in range(4) for j in range(4) if i != j
    }
    top_two = sorted(off_diagonal, key=off_diagonal.get, reverse=True)[:2]
    check(set(top_two) == {("Cell", "Cell-Multi"), ("Cell-Multi", "Cell")}, "main confusion")
    third = sorted(off_diagonal, key=off_diagonal.get, reverse=True)[2:4]
    check(
        set(third) == {("Cell", "No-Anomaly"), ("Cell-Multi", "No-Anomaly")},
        "small anomalies predicted as normal are the next error",
    )
    check(per["No-Anomaly"]["recall"] > per["No-Anomaly"]["precision"], "normal recall > precision")
    check(test["f1_macro"] > base["f1_macro"], "CNN beats the baseline")
    check(
        base["ci95_bootstrap"]["f1_macro"]["high"] < test["ci95_bootstrap"]["f1_macro"]["low"],
        "bootstrap intervals do not overlap",
    )
    check(
        base_binary["specificity"] < binary["specificity"]
        and base_binary["sensitivity"] < binary["sensitivity"],
        "binary view improves over baseline",
    )
    check(
        all(per[c]["auc_ovr"] > base["per_class"][c]["auc_ovr"] for c in CLASSES),
        "CNN AUC above baseline in every class",
    )
    check(
        base["confusion_matrix"][0][1] + base["confusion_matrix"][0][2] > 50,
        "baseline confuses many normal images with cell anomalies",
    )
    check(
        abs(test["f1_macro"] - w16["validation_runs"][str(chosen["deployed_seed"])]["f1_macro"])
        < 0.03,
        "test close to validation",
    )
    check(chosen["numpy_export_prediction_agreement_full_test"] == 1.0, "100 % agreement")

    prevalence = original["No-Anomaly"] / four_total
    sens, spec = binary["sensitivity"], binary["specificity"]
    ppv = sens * (1 - prevalence) / (sens * (1 - prevalence) + (1 - spec) * prevalence)
    npv = spec * prevalence / (spec * prevalence + (1 - sens) * (1 - prevalence))
    cell_confusion = cm[index["Cell"]][index["Cell-Multi"]] + cm[index["Cell-Multi"]][index["Cell"]]
    missed = sum(cm[i][index["No-Anomaly"]] for i in range(1, 4))
    false_alarms = sum(cm[index["No-Anomaly"]][1:])
    v = {
        "zip_total": integer(info["metadata_count"]),
        "zip_classes": str(len(original)),
        "n_noanomaly": integer(original["No-Anomaly"]),
        "n_cell": integer(original["Cell"]),
        "n_cellmulti": integer(original["Cell-Multi"]),
        "n_diode": integer(original["Diode"]),
        "n_four": integer(four_total),
        "n_vegetation": integer(original["Vegetation"]),
        "n_shadowing": integer(original["Shadowing"]),
        "n_soiling": integer(original["Soiling"]),
        "n_diode_multi": integer(original["Diode-Multi"]),
        "n_cracking": integer(original["Cracking"]),
        "n_offline": integer(original["Offline-Module"]),
        "prev_all": str(round(100 * original["No-Anomaly"] / info["metadata_count"])),
        "prev_four": str(round(100 * original["No-Anomaly"] / four_total)),
        "exact_dups": str(manifest["exact_duplicates_removed_before_sampling"]),
        "rms_threshold": dec(dedup["threshold_rms"], 2),
        "rms_threshold_gray": str(round(dedup["threshold_rms"] * 255)),
        "rejected_total": str(sum(sum(stage.values()) for stage in rejected.values())),
        "rejected_noanomaly": str(sum(stage["No-Anomaly"] for stage in rejected.values())),
        "rejected_cross": str(sum(manifest["rejected_near_duplicates_of_other_class"].values())),
        "min_cross_rms": dec(manifest["min_cross_split_rms"], 4),
        "train_pc": integer(split["train"]),
        "val_pc": integer(split["validation"]),
        "test_pc": integer(split["test"]),
        "train_total": integer(totals["train"]),
        "val_total": integer(totals["validation"]),
        "test_total": integer(totals["test"]),
        "all_total": integer(sum(totals.values())),
        "old_images": integer(manifest["old_manifest"]["images"]),
        "old_in_train": integer(manifest["old_manifest_images_by_split"]["train"]),
        "old_in_val": integer(manifest["old_manifest_images_by_split"]["validation"]),
        "seed": str(metrics["protocol"]["cnn_seeds"][0]),
        "seeds": ", ".join(str(s) for s in protocol["cnn_seeds"]),
        "n_seeds": str(len(protocol["cnn_seeds"])),
        "tie_margin": dec(float(re.search(r"within ([0-9.]+)", protocol["tie_rule"]).group(1)), 2),
        "resamples": integer(int(re.search(r"(\d+) resamples", protocol["test_statistics"])[1])),
        "params_lr": integer(models["logistic_regression"]["parameters"]),
        "params_w8": integer(w8["parameters"]),
        "params_w16": integer(w16["parameters"]),
        "lr_iters": str(models["logistic_regression"]["iterations"]),
        "lr_val_f1": dec(lr_val["f1_macro"]),
        "w8_val_f1": dec(w8["validation_mean"]["f1_macro"]),
        "w16_val_f1": dec(w16["validation_mean"]["f1_macro"]),
        "w16_w8_gap": dec(w16["validation_mean"]["f1_macro"] - w8["validation_mean"]["f1_macro"]),
        "w16_seed_f1": dec(w16["validation_runs"][str(chosen["deployed_seed"])]["f1_macro"]),
        "w16_seconds": str(round(sum(w16["training_seconds"].values()) / 3)),
        "acc": dec(test["accuracy"]),
        "acc_ci": ci(test["ci95_bootstrap"]["accuracy"]),
        "prec": dec(test["precision_macro"]),
        "rec": dec(test["recall_macro"]),
        "f1": dec(test["f1_macro"]),
        "f1_ci": ci(test["ci95_bootstrap"]["f1_macro"]),
        "auc": dec(test["auc_macro_ovr"]),
        "auc_ci": ci(test["ci95_bootstrap"]["auc_macro_ovr"]),
        "base_acc": dec(base["accuracy"]),
        "base_f1": dec(base["f1_macro"]),
        "base_f1_ci": ci(base["ci95_bootstrap"]["f1_macro"]),
        "base_auc": dec(base["auc_macro_ovr"]),
        "ppv_field": str(round(100 * ppv)),
        "npv_field": str(round(100 * npv)),
        "detected": str(binary["positives"] - missed),
        "sens": dec(binary["sensitivity"]),
        "spec": dec(binary["specificity"]),
        "base_sens": dec(base_binary["sensitivity"]),
        "base_spec": dec(base_binary["specificity"]),
        "positives": str(binary["positives"]),
        "negatives": str(binary["negatives"]),
        "missed": str(missed),
        "false_alarms": str(false_alarms),
        "cell_to_multi": str(cm[index["Cell"]][index["Cell-Multi"]]),
        "multi_to_cell": str(cm[index["Cell-Multi"]][index["Cell"]]),
        "cell_confusion": str(cell_confusion),
        "cell_to_normal": str(cm[index["Cell"]][index["No-Anomaly"]]),
        "multi_to_normal": str(cm[index["Cell-Multi"]][index["No-Anomaly"]]),
        "errors_total": str(test["misclassified_total"]),
        "cell_confusion_share": str(round(100 * cell_confusion / test["misclassified_total"])),
        "diode_recall": dec(per["Diode"]["recall"]),
        "diode_f1": dec(per["Diode"]["f1"]),
        "normal_recall": dec(per["No-Anomaly"]["recall"]),
        "normal_prec": dec(per["No-Anomaly"]["precision"]),
        "cell_f1": dec(per["Cell"]["f1"]),
        "multi_f1": dec(per["Cell-Multi"]["f1"]),
        "cell_recall": dec(per["Cell"]["recall"]),
        "multi_recall": dec(per["Cell-Multi"]["recall"]),
        "cell_auc": dec(per["Cell"]["auc_ovr"]),
        "multi_auc": dec(per["Cell-Multi"]["auc_ovr"]),
        "diode_auc": dec(per["Diode"]["auc_ovr"]),
        "normal_auc": dec(per["No-Anomaly"]["auc_ovr"]),
        "normal_f1": dec(per["No-Anomaly"]["f1"]),
        "base_normal_f1": dec(base["per_class"]["No-Anomaly"]["f1"]),
        "base_cell_f1": dec(base["per_class"]["Cell"]["f1"]),
        "base_multi_f1": dec(base["per_class"]["Cell-Multi"]["f1"]),
        "parity_m": sci(chosen["numpy_export_parity_max_abs_prob_error_full_test"])[0],
        "parity_e": sci(chosen["numpy_export_parity_max_abs_prob_error_full_test"])[1],
        "agreement": str(round(100 * chosen["numpy_export_prediction_agreement_full_test"])),
        "artifact_kb": str(round(chosen["artifact_bytes"] / 1000)),
        "hist_acc": dec(historical["selected_model"]["test"]["accuracy"]),
        "hist_test": str(historical["split_counts"]["test"]),
        "py": metrics["environment"]["python"],
        "torch": metrics["environment"]["torch"].split("+")[0],
        "sklearn": metrics["environment"]["scikit_learn"],
        "lr": dec(float(recipe_value(recipe["optimizer"], "lr")), 3),
        "wd": dec(float(recipe_value(recipe["optimizer"], "weight_decay")), 3),
        "smoothing": dec(float(recipe_value(recipe["loss"], "label_smoothing")), 2),
        "dropout": dec(float(re.search(r"dropout\(([0-9.]+)", recipe["architecture"])[1]), 1),
        "batch": str(recipe["batch_size"]),
        "epochs": str(recipe["epochs"]),
        "eps": recipe["input_normalization"].split("+ ")[1].split(")")[0],
    }
    return v


def table_rows(rows: list[list[str]]) -> str:
    return ", ".join(f"[{cell}]" for row in rows for cell in row)


def build_tables(metrics: dict, manifest: dict) -> dict:
    models = metrics["models"]
    test = metrics["selected_model"]["test"]
    base = metrics["baseline_reference"]["test"]
    names = {
        "logistic_regression": "Reg. logística",
        "cnn_w8": "CNN w8",
        "cnn_w16": "CNN w16",
    }
    validation = []
    for key, label in names.items():
        mean, sd = models[key]["validation_mean"], models[key]["validation_sd"]
        runs = len(models[key]["validation_runs"])

        def cell(metric: str, mean: dict = mean, sd: dict = sd, runs: int = runs) -> str:
            if runs == 1:
                return dec(mean[metric])
            return f"{dec(mean[metric])}{NNBSP}±{NNBSP}{dec(sd[metric])}"

        name = f"*{label}*" if key == metrics["selected_model"]["name"] else label
        validation.append(
            [name, integer(models[key]["parameters"]), str(runs), cell("f1_macro"),
             cell("auc_macro_ovr")]
        )
    per_class = []
    for c in CLASSES:
        p = test["per_class"][c]
        per_class.append(
            [c, dec(p["precision"]), dec(p["recall"]), dec(p["f1"]), dec(p["auc_ovr"]),
             dec(base["per_class"][c]["f1"])]
        )
    comparison = []
    for label, block in (("CNN w16", test), ("Reg. logística", base)):
        b = block["binary_anomaly_vs_normal"]
        comparison.append(
            [label, dec(block["accuracy"]), dec(block["f1_macro"]), dec(block["auc_macro_ovr"]),
             dec(b["sensitivity"]), dec(b["specificity"])]
        )
    return {
        "validation_rows": table_rows(validation),
        "class_rows": table_rows(per_class),
        "comparison_rows": table_rows(comparison),
    }


def render_typst(values: dict, tables: dict) -> str:
    lines = [
        "// Generated by report/build_report.py from artifacts/*.json. Do not edit by hand.",
        "#let v = (",
    ]
    lines += [f"  {key}: {typst_string(value)}," for key, value in values.items()]
    lines.append(")")
    for key, rows in tables.items():
        lines.append(f"#let {key} = ({rows})")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- figures (optional)


def plot_figures(metrics: dict, manifest: dict, archive: Path) -> None:
    import io

    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "savefig.dpi": 300})
    FIGURES.mkdir(exist_ok=True)
    test = metrics["selected_model"]["test"]
    base = metrics["baseline_reference"]["test"]

    def comma(x: float, places: int = 1) -> str:
        return f"{x:.{places}f}".replace(".", ",")

    # Confusion matrices: selected CNN and baseline, counts and row percentage.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    for ax, block, title in (
        (axes[0], test, "(a) CNN w16"),
        (axes[1], base, "(b) Regresión logística"),
    ):
        matrix = np.asarray(block["confusion_matrix"])
        rows = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(
                    j, i, f"{matrix[i, j]}\n{comma(100 * rows[i, j], 0)} %",
                    ha="center", va="center", fontsize=7,
                    color="white" if rows[i, j] > 0.55 else "black",
                )
        ax.set_xticks(range(4), CLASSES, rotation=20)
        ax.set_yticks(range(4), CLASSES)
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        ax.set_title(title, fontsize=9)
    fig.savefig(FIGURES / "confusion_matrix.png")
    plt.close(fig)

    # ROC curves (one-vs-rest) of the selected CNN; baseline macro AUC in the title.
    colors = {"No-Anomaly": "#4c72b0", "Cell": "#dd8452", "Cell-Multi": "#55a868",
              "Diode": "#c44e52"}
    fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
    for label in CLASSES:
        curve = test["roc_curves"][label]
        ax.plot(curve["fpr"], curve["tpr"], color=colors[label], lw=1.3,
                label=f"{label} ({comma(curve['auc'], 3)})")
        ref = base["roc_curves"][label]
        ax.plot(ref["fpr"], ref["tpr"], color=colors[label], lw=0.8, ls=":", alpha=0.8)
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.8)
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
           xlim=(0, 1), ylim=(0, 1.01))
    ticks = matplotlib.ticker.FuncFormatter(lambda x, _pos: comma(x))
    ax.xaxis.set_major_formatter(ticks)
    ax.yaxis.set_major_formatter(ticks)
    ax.legend(title="CNN w16, clase (AUC)", fontsize=7, title_fontsize=7, loc="lower right")
    fig.savefig(FIGURES / "roc_curves.png")
    plt.close(fig)

    if not archive.is_file():
        print(f"Archive not found ({archive}); image figures were not regenerated.")
        return
    with zipfile.ZipFile(archive) as source:

        def image(path: str) -> np.ndarray:
            with Image.open(io.BytesIO(source.read(path))) as handle:
                return np.asarray(handle.convert("L"), dtype=np.float64)

        def show(ax, pixels: np.ndarray) -> None:
            low, high = pixels.min(), pixels.max()
            ax.imshow((pixels - low) / max(high - low, 1.0), cmap="inferno",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

        # Class examples: five random training images per class.
        rng = random.Random(42)
        per_row = 6
        fig, axes = plt.subplots(4, per_row, figsize=(3.4, 4.2), constrained_layout=True)
        for row, label in enumerate(CLASSES):
            pool = sorted(
                (r for r in manifest["records"] if r["label"] == label and r["split"] == "train"),
                key=lambda r: r["image_id"],
            )
            for col, record in enumerate(rng.sample(pool, per_row)):
                show(axes[row, col], image(record["path"]))
            axes[row, 0].set_ylabel(label, fontsize=8)
        fig.savefig(FIGURES / "class_examples.png")
        plt.close(fig)

        # Most confident test errors of the selected model, per error type.
        wanted = [("Cell", "Cell-Multi")] * 2 + [("Cell-Multi", "Cell")] * 2 + [
            ("Cell", "No-Anomaly"), ("Cell-Multi", "No-Anomaly"), ("Diode", "No-Anomaly"),
            ("No-Anomaly", "Diode"),
        ]
        pool = list(test["misclassified"])
        chosen = []
        for pair in wanted:
            item = next(e for e in pool if (e["true"], e["predicted"]) == pair)
            pool.remove(item)
            chosen.append(item)
        short = {"No-Anomaly": "NoA", "Cell": "Cell", "Cell-Multi": "C-M", "Diode": "Diode"}
        fig, axes = plt.subplots(1, len(chosen), figsize=(3.4, 1.12), constrained_layout=True)
        for ax, item in zip(axes, chosen, strict=True):
            show(ax, image(item["path"]))
            p = item["probabilities"][item["predicted"]]
            ax.set_title(f"{short[item['true']]}\n→{short[item['predicted']]}\n{comma(p, 2)}",
                         fontsize=6.5)
        fig.savefig(FIGURES / "error_examples.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true", help="also render report figures")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    metrics = load("metrics.json")
    manifest = load("manifest.json")
    info = load("dataset_info.json")
    historical = load("historical/metrics.json")
    if metrics.get("status") != "complete":
        raise ValueError("metrics.json is not complete")
    values = build_values(metrics, manifest, info, historical)
    tables = build_tables(metrics, manifest)
    OUTPUT.write_text(render_typst(values, tables), encoding="utf-8")
    print(f"Generated {OUTPUT}")
    if args.figures:
        plot_figures(metrics, manifest, args.archive)
        print(f"Rendered figures in {FIGURES}")


if __name__ == "__main__":
    main()
