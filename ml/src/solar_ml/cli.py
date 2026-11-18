import argparse
import json
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from solar_ml.constants import ARCHIVE_NAME, CLASSES, SEED
from solar_ml.dataset import DEDUP_THRESHOLD_RMS, inspect_archive, make_manifest
from solar_ml.train import train


def download(destination: Path) -> None:
    url = f"https://github.com/RaptorMaps/InfraredSolarModules/raw/refs/heads/master/{ARCHIVE_NAME}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=90) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    with zipfile.ZipFile(destination) as source:
        if source.testzip() is not None:
            raise ValueError("Downloaded ZIP failed integrity check")


def plot_dataset(archive: Path, manifest: dict, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(item["label"] for item in manifest["records"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(CLASSES, [counts[label] for label in CLASSES], color="#167a84")
    ax.set(ylabel="Imágenes seleccionadas")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(9, 2.7))
    with zipfile.ZipFile(archive) as source:
        for axis, label in zip(axes, CLASSES, strict=True):
            item = next(item for item in manifest["records"] if item["label"] == label)
            from io import BytesIO

            with Image.open(BytesIO(source.read(item["path"]))) as image:
                axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axis.set_title(label, fontsize=9)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_examples.png", dpi=180)
    plt.close(fig)


def export_demo(archive: Path, manifest: dict, destination: Path, archive_sha256: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    selected = []
    with zipfile.ZipFile(archive) as source:
        for label in CLASSES:
            examples = [
                item
                for item in manifest["records"]
                if item["label"] == label and item["split"] == "test"
            ][:2]
            for item in examples:
                filename = f"{label}_{item['image_id']}.jpg"
                (destination / filename).write_bytes(source.read(item["path"]))
                selected.append({"file": filename, "image_id": item["image_id"], "class": label})
    provenance = {
        "source": "https://github.com/RaptorMaps/InfraredSolarModules",
        "license": "MIT (source repository)",
        "archive_sha256": archive_sha256,
        "split": "test",
        "images": selected,
    }
    selected_names = {item["file"] for item in selected}
    for old_image in destination.glob("*.jpg"):
        if old_image.name not in selected_names:
            old_image.unlink()
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and train on InfraredSolarModules")
    parser.add_argument("--archive", type=Path, default=Path("data") / ARCHIVE_NAME)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=150)
    parser.add_argument("--dedup-threshold", type=float, default=DEDUP_THRESHOLD_RMS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--old-manifest",
        type=Path,
        default=Path("../artifacts/historical/manifest.json"),
        help="Historical manifest whose images are excluded from the fresh test split",
    )
    parser.add_argument("--output", type=Path, default=Path("../artifacts"))
    parser.add_argument("--figures", type=Path, default=Path("../artifacts/figures"))
    parser.add_argument("--demo", type=Path, default=Path("demo"))
    parser.add_argument("--fixture-output", type=Path, default=None)
    args = parser.parse_args()
    if min(args.train_per_class, args.validation_per_class, args.test_per_class) < 1:
        parser.error("per-class split sizes must be positive")
    if args.download and not args.archive.exists():
        download(args.archive)
    if not args.archive.is_file():
        parser.error(f"Dataset ZIP unavailable: {args.archive}; pass --archive or --download")
    if not args.old_manifest.is_file():
        parser.error(f"Historical manifest unavailable: {args.old_manifest}")
    old_ids = {
        item["image_id"]
        for item in json.loads(args.old_manifest.read_text(encoding="utf-8"))["records"]
    }
    info, dataset = inspect_archive(args.archive)
    manifest = make_manifest(
        dataset["records"],
        dataset["pixels"],
        old_ids,
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
        test_per_class=args.test_per_class,
        threshold=args.dedup_threshold,
        seed=args.seed,
    )
    manifest["old_manifest"] = {"path": str(args.old_manifest), "images": len(old_ids)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_dataset(args.archive, manifest, args.figures)
    export_demo(args.archive, manifest, args.demo, info["archive_sha256"])
    results = train(args.archive, manifest, args.output, args.figures, epochs=args.epochs)
    reference = args.output / "reference_prediction.json"
    if args.fixture_output is not None and reference.exists():
        args.fixture_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference, args.fixture_output)
    summary_keys = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_macro_ovr")
    print(
        json.dumps(
            {
                "splits": manifest["split_counts_by_class"],
                "validation": {
                    name: {
                        "mean": model["validation_mean"],
                        "sd": model["validation_sd"],
                    }
                    for name, model in results["models"].items()
                },
                "selected_model": results["selected_model"]["name"],
                "test": {
                    key: results["selected_model"]["test"][key]
                    for key in (*summary_keys, "ci95_bootstrap", "binary_anomaly_vs_normal")
                },
                "baseline_test": {
                    key: results["baseline_reference"]["test"][key] for key in summary_keys
                },
                "parity": results["selected_model"][
                    "numpy_export_parity_max_abs_prob_error_full_test"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
