"""Recompute backend ConvClassifier parity on the test split and record it in metrics.json.

Usage (from ml/): uv run python -m solar_ml.verify_backend --archive <zip>
Only ``selected_model.backend_convclassifier_parity`` in metrics.json is (re)written.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from solar_ml.constants import ARCHIVE_NAME
from solar_ml.dataset import load_arrays
from solar_ml.export import forward

ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    "backend/src/infra/cnn.py ConvClassifier vs solar_ml.export.forward, checked after training"
)


def backend_parity(artifact: dict, features: np.ndarray, backend_src: Path) -> dict:
    sys.path.insert(0, str(backend_src))
    try:
        from infra.cnn import ConvClassifier

        classifier = ConvClassifier(artifact)
    finally:
        sys.path.remove(str(backend_src))
    reference = forward(artifact, features)
    error = 0.0
    for row, expected in zip(features, reference, strict=True):
        prediction = classifier.predict(row.tolist())
        actual = np.asarray([prediction.probabilities[label] for label in artifact["classes"]])
        error = max(error, float(np.max(np.abs(actual - expected))))
    return {
        "max_abs_prob_error_full_test": error,
        "source": SOURCE,
        "sha256_match": classifier.sha256 == artifact["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=Path("data") / ARCHIVE_NAME)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--backend-src", type=Path, default=ROOT / "backend" / "src")
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Dataset ZIP unavailable: {args.archive}")
    metrics_path = args.artifacts / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    artifact = json.loads((args.artifacts / "model.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.artifacts / "manifest.json").read_text(encoding="utf-8"))
    test_x, _ = load_arrays(args.archive, manifest)["test"]
    block = backend_parity(artifact, test_x, args.backend_src)
    metrics["selected_model"]["backend_convclassifier_parity"] = block
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(block, indent=2))


if __name__ == "__main__":
    main()
