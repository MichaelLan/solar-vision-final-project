import hashlib
import json
import zipfile
from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from sklearn.linear_model import LogisticRegression

from solar_ml.cli import export_demo
from solar_ml.constants import CLASSES, FEATURE_COUNT
from solar_ml.dataset import Record, inspect_archive, make_manifest
from solar_ml.images import decode_pixels
from solar_ml.train import export_model, infer_exported, runtime_probabilities


def test_luminance_and_dimensions():
    rgb = np.zeros((40, 24, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    image = Image.fromarray(rgb)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    pixels = decode_pixels(buffer.getvalue())
    assert pixels.shape == (FEATURE_COUNT,)
    assert pixels[0] == np.float32(76 / 255)


def test_export_matches_training_predictions(tmp_path):
    rng = np.random.default_rng(42)
    features = rng.random((20, FEATURE_COUNT))
    labels = np.tile(np.arange(4), 5)
    model = LogisticRegression(max_iter=50).fit(features, labels)
    artifact = export_model(model, tmp_path / "model.json")
    assert artifact["classes"] == list(CLASSES)
    actual = np.stack([infer_exported(artifact, row) for row in features])
    np.testing.assert_allclose(actual, model.predict_proba(features), atol=1e-9)
    deployed = np.stack([runtime_probabilities(artifact, row) for row in features[:3]])
    np.testing.assert_allclose(deployed, model.predict_proba(features[:3]), atol=1e-9)
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-12)


def test_manifest_keeps_similar_images_together():
    records = [
        Record(
            str(index),
            f"images/{index}.png",
            CLASSES[index // 20],
            str(index),
            hashlib.sha256(str(index // 2).encode()).hexdigest()[:16],
            str(index // 2),
        )
        for index in range(80)
    ]
    manifest = make_manifest(records, per_class=10)
    assert len(manifest["records"]) == 40
    assert manifest["dhash_cross_split_pairs_hamming_le_2"] == 0
    split_by_quantized = {}
    for item in manifest["records"]:
        split_by_quantized.setdefault(item["quantized_hash"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in split_by_quantized.values())
    for counts in manifest["split_counts_by_class"].values():
        assert all(count > 0 for count in counts.values())


def test_manifest_rejects_insufficient_distinct_images():
    records = [
        Record(
            str(index),
            f"images/{index}.png",
            CLASSES[index // 10],
            str(index),
            f"{index // 2:016x}",
            "same-quantized-image",
        )
        for index in range(40)
    ]
    with pytest.raises(ValueError, match="Cannot select"):
        make_manifest(records, per_class=10)


def test_archive_inspection_detects_metadata_and_images(tmp_path):
    archive = tmp_path / "sample.zip"
    image = Image.new("RGB", (24, 40), (10, 20, 30))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    metadata = {"1": {"image_filepath": "images/1.jpg", "anomaly_class": "Cell"}}
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("module_metadata.json", json.dumps(metadata))
        output.writestr("images/1.jpg", buffer.getvalue())
    info, dataset = inspect_archive(archive)
    assert info["original_counts"]["Cell"] == 1
    assert (
        dataset["records"][0].sha256
        == hashlib.sha256(decode_pixels(buffer.getvalue()).tobytes()).hexdigest()
    )


def test_demo_export_removes_stale_jpeg_only(tmp_path):
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("images/1.jpg", b"image")
    destination = tmp_path / "demo"
    destination.mkdir()
    (destination / "Cell_old.jpg").write_bytes(b"old")
    (destination / "notes.txt").write_text("keep", encoding="utf-8")
    manifest = {
        "records": [{"label": "Cell", "split": "test", "path": "images/1.jpg", "image_id": "1"}]
    }
    export_demo(archive, manifest, destination, "archive-hash")
    assert {item.name for item in destination.glob("*.jpg")} == {"Cell_1.jpg"}
    assert (destination / "notes.txt").read_text(encoding="utf-8") == "keep"
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["images"] == [{"file": "Cell_1.jpg", "image_id": "1", "class": "Cell"}]
