import hashlib
import json
import zipfile
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from solar_ml import cnn
from solar_ml.cli import export_demo
from solar_ml.constants import CLASSES, FEATURE_COUNT, HEIGHT, WIDTH
from solar_ml.dataset import DEDUP_THRESHOLD_RMS, Record, inspect_archive, make_manifest
from solar_ml.export import export_cnn, export_dense, forward, model_sha256
from solar_ml.images import decode_pixels
from solar_ml.train import select_architecture, train


def test_luminance_and_dimensions():
    rgb = np.zeros((40, 24, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    image = Image.fromarray(rgb)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    pixels = decode_pixels(buffer.getvalue())
    assert pixels.shape == (FEATURE_COUNT,)
    assert pixels[0] == np.float32(76 / 255)


def synthetic_dataset(per_class=30, seed=0):
    """Distinct random images plus near duplicates (RMS ~0.005) and one exact duplicate."""
    rng = np.random.default_rng(seed)
    records, pixels = [], []
    for c, label in enumerate(CLASSES):
        for k in range(per_class):
            index = c * 1000 + k
            image = rng.random(FEATURE_COUNT).astype(np.float32)
            records.append(Record(str(index), f"images/{index}.jpg", label, f"h{index}"))
            pixels.append(image)
            if k % 5 == 0:
                near = np.clip(image + rng.normal(0, 0.005, FEATURE_COUNT), 0, 1)
                records.append(Record(f"{index}n", f"images/{index}n.jpg", label, f"h{index}n"))
                pixels.append(near.astype(np.float32))
        records.append(Record(f"{c}dup", f"images/{c}dup.jpg", label, f"h{c * 1000}"))
        pixels.append(pixels[[r.image_id for r in records].index(str(c * 1000))])
    return records, np.stack(pixels)


def pairwise_rms(values):
    values = values.astype(np.float64)
    difference = values[:, None, :] - values[None, :, :]
    distances = np.sqrt((difference**2).mean(axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances


def test_manifest_split_invariants():
    records, pixels = synthetic_dataset()
    old = {str(c * 1000 + k) for c in range(4) for k in range(0, 30, 2)}
    manifest = make_manifest(
        records, pixels, old, train_per_class=8, validation_per_class=3, test_per_class=4
    )
    items = manifest["records"]
    ids = [item["image_id"] for item in items]
    assert len(ids) == len(set(ids)) == 4 * 15
    for label in CLASSES:
        assert manifest["split_counts_by_class"][label] == {
            "train": 8,
            "validation": 3,
            "test": 4,
        }
    test_ids = {item["image_id"] for item in items if item["split"] == "test"}
    assert test_ids.isdisjoint(old)
    assert manifest["old_manifest_excluded_from_test"] is True
    assert manifest["old_manifest_images_by_split"]["test"] == 0
    assert all(item["in_old_manifest"] == (item["image_id"] in old) for item in items)
    index = {record.image_id: k for k, record in enumerate(records)}
    chosen = pixels[[index[i] for i in ids]]
    assert pairwise_rms(chosen).min() > DEDUP_THRESHOLD_RMS
    assert manifest["min_cross_split_rms"] > DEDUP_THRESHOLD_RMS
    assert manifest["exact_duplicates_removed_before_sampling"] == 4
    assert manifest["dedup"]["threshold_rms"] == DEDUP_THRESHOLD_RMS
    again = make_manifest(
        records, pixels, old, train_per_class=8, validation_per_class=3, test_per_class=4
    )
    assert again["records"] == items


def test_manifest_rejects_conflicts_and_shortages():
    records, pixels = synthetic_dataset(per_class=10)
    with pytest.raises(ValueError, match="Cannot select"):
        make_manifest(records, pixels, set(), train_per_class=8, validation_per_class=3)
    conflicting = records + [Record("x", "images/x.jpg", "Diode", records[0].sha256)]
    with pytest.raises(ValueError, match="conflicting labels"):
        make_manifest(conflicting, np.vstack([pixels, pixels[:1]]), set())


def test_cnn_export_matches_torch_and_hash():
    torch.manual_seed(0)
    model = cnn.SmallCNN(8).eval()
    features = np.random.default_rng(1).random((16, FEATURE_COUNT))
    artifact = export_cnn(model, {"type": "per_image", "eps": cnn.PER_IMAGE_EPS})
    assert artifact["version"] == "2.0.0" and artifact["architecture"] == "cnn"
    assert artifact["classes"] == list(CLASSES)
    assert [layer["type"] for layer in artifact["layers"]] == [
        "conv2d", "relu", "maxpool2d", "conv2d", "relu", "maxpool2d", "conv2d", "relu",
        "global_avg_pool", "dense", "softmax",
    ]  # fmt: skip
    assert artifact["sha256"] == model_sha256(json.loads(json.dumps(artifact)))
    expected = cnn.predict_proba(model, features)
    np.testing.assert_allclose(forward(artifact, features), expected, atol=1e-5)
    np.testing.assert_allclose(forward(artifact, features[0]), expected[0], atol=1e-5)


def test_dense_export_matches_logistic_regression():
    rng = np.random.default_rng(42)
    features = rng.random((20, FEATURE_COUNT))
    labels = np.tile(np.arange(4), 5)
    model = LogisticRegression(max_iter=50).fit(features, labels)
    artifact = export_dense(model)
    assert artifact["architecture"] == "dense"
    assert artifact["input"]["normalization"] == {"type": "none"}
    np.testing.assert_allclose(
        forward(artifact, features), model.predict_proba(features), atol=1e-5
    )


def test_cnn_training_is_reproducible():
    rng = np.random.default_rng(3)
    features = rng.random((40, FEATURE_COUNT))
    labels = np.tile(np.arange(4), 10)
    first, _ = cnn.train_cnn(features, labels, width=8, seed=42, epochs=2, batch_size=16)
    second, _ = cnn.train_cnn(features, labels, width=8, seed=42, epochs=2, batch_size=16)
    for a, b in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_selection_rule_prefers_cheaper_model_within_margin():
    def candidate(f1):
        return {"validation_mean": {"f1_macro": f1}}

    models = {"logistic_regression": candidate(0.9), "cnn_w8": candidate(0.70)}
    assert select_architecture({**models, "cnn_w16": candidate(0.705)}) == "cnn_w8"
    assert select_architecture({**models, "cnn_w16": candidate(0.72)}) == "cnn_w16"


def test_tiny_end_to_end_run(tmp_path):
    rng = np.random.default_rng(7)
    archive = tmp_path / "tiny.zip"
    records = []
    with zipfile.ZipFile(archive, "w") as output:
        for c, label in enumerate(CLASSES):
            for k in range(8):
                base = np.full((HEIGHT, WIDTH), 40 + 40 * c, dtype=np.float64)
                image = np.clip(base + rng.normal(0, 25, base.shape), 0, 255).astype(np.uint8)
                buffer = BytesIO()
                Image.fromarray(image).save(buffer, format="PNG")
                path = f"images/{c}_{k}.png"
                output.writestr(path, buffer.getvalue())
                split = "train" if k < 4 else "validation" if k < 6 else "test"
                records.append(
                    {"image_id": f"{c}{k}", "path": path, "label": label, "split": split}
                )
    results = train(archive, {"records": records}, tmp_path / "out", None, epochs=1, seeds=(42, 43))
    saved = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert saved["protocol"]["registered_before_training"] is True
    assert set(saved["models"]) == {"logistic_regression", "cnn_w8", "cnn_w16"}
    assert results["selected_model"]["numpy_export_parity_max_abs_prob_error_full_test"] < 1e-5
    artifact = json.loads((tmp_path / "out" / "model.json").read_text())
    assert artifact["sha256"] == model_sha256(artifact)
    assert artifact["input"]["normalization"]["type"] == "per_image"
    reference = json.loads((tmp_path / "out" / "reference_prediction.json").read_text())
    probabilities = forward(artifact, np.asarray(reference["pixels"]))
    assert CLASSES[int(probabilities.argmax())] == reference["label"]
    test = saved["selected_model"]["test"]
    assert {"ci95_bootstrap", "binary_anomaly_vs_normal", "misclassified"} <= set(test)


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
    assert dataset["pixels"].shape == (1, FEATURE_COUNT)


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
