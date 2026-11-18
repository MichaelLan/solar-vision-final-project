import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from solar_ml.constants import ARCHIVE_NAME, CLASSES, HEIGHT, SEED, SOURCE_URL, WIDTH
from solar_ml.images import decode_pixels


@dataclass
class Record:
    image_id: str
    path: str
    label: str
    sha256: str
    split: str = ""


def inspect_archive(archive: Path) -> tuple[dict, dict]:
    with zipfile.ZipFile(archive) as source:
        corrupt = source.testzip()
        if corrupt:
            raise ValueError(f"Corrupt ZIP member: {corrupt}")
        names = source.namelist()
        metadata_names = [name for name in names if name.endswith("module_metadata.json")]
        if len(metadata_names) != 1:
            raise ValueError("Expected exactly one module_metadata.json")
        metadata_path = PurePosixPath(metadata_names[0])
        metadata = json.loads(source.read(metadata_names[0]))
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be an object")
        counts = Counter()
        records = []
        arrays = []
        missing = []
        dimensions = Counter()
        formats = Counter()
        modes = Counter()
        for image_id, item in metadata.items():
            if (
                not isinstance(item, dict)
                or "anomaly_class" not in item
                or "image_filepath" not in item
            ):
                raise ValueError(f"Invalid metadata entry: {image_id}")
            label = item["anomaly_class"]
            counts[label] += 1
            if label not in CLASSES:
                continue
            relative = PurePosixPath(item["image_filepath"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe image path: {relative}")
            full_path = str(metadata_path.parent / relative)
            if full_path not in names:
                missing.append(full_path)
                continue
            data = source.read(full_path)
            from io import BytesIO

            with Image.open(BytesIO(data)) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                formats[str(image.format)] += 1
                modes[image.mode] += 1
            pixels = decode_pixels(data)
            arrays.append(pixels)
            records.append(
                Record(
                    image_id=str(image_id),
                    path=full_path,
                    label=label,
                    sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                )
            )
        if missing:
            raise ValueError(f"Missing {len(missing)} images; first: {missing[0]}")
        with archive.open("rb") as archive_stream:
            archive_sha256 = hashlib.file_digest(archive_stream, "sha256").hexdigest()
        info = {
            "source": SOURCE_URL,
            "archive_name": ARCHIVE_NAME,
            "archive_sha256": archive_sha256,
            "metadata_count": len(metadata),
            "original_counts": dict(sorted(counts.items())),
            "selected_class_counts": dict(Counter(record.label for record in records)),
            "validated_dimensions": dict(dimensions),
            "observed_formats": dict(formats),
            "observed_modes": dict(modes),
            "selected_image_count": len(records),
        }
        pixels = np.stack(arrays) if arrays else np.zeros((0, WIDTH * HEIGHT), dtype=np.float32)
        return info, {"records": records, "metadata": metadata, "pixels": pixels}


DEDUP_THRESHOLD_RMS = 0.02
SPLITS = ("train", "validation", "test")


def rms_distances(candidate: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Root-mean-square pixel distance between one image and each image of a pool."""
    if len(pool) == 0:
        return np.zeros(0)
    difference = pool.astype(np.float64) - candidate.astype(np.float64)
    return np.sqrt(np.mean(difference * difference, axis=1))


def min_cross_split_rms(pixels: np.ndarray, splits: list[str]) -> float:
    """Smallest RMS distance between two selected images that belong to different splits."""
    values = pixels.astype(np.float64)
    norms = (values * values).sum(axis=1)
    labels = np.asarray(splits)
    best = np.inf
    for split in SPLITS:
        left = values[labels == split]
        right = values[labels != split]
        if len(left) == 0 or len(right) == 0:
            continue
        squared = (
            norms[labels == split][:, None] + norms[labels != split][None] - 2 * left @ right.T
        )
        best = min(best, float(np.sqrt(max(squared.min(), 0.0) / values.shape[1])))
    return best


def make_manifest(
    records: list[Record],
    pixels: np.ndarray,
    old_manifest_ids: set[str],
    train_per_class: int = 1000,
    validation_per_class: int = 100,
    test_per_class: int = 150,
    threshold: float = DEDUP_THRESHOLD_RMS,
    seed: int = SEED,
) -> dict:
    """Balanced split with exact and pixel-RMS near-duplicate removal across all splits.

    Test images are drawn first and only from images absent from the previous manifest.
    Every accepted image must be farther than ``threshold`` (RMS on [0, 1] grayscale) from
    every image accepted before it, regardless of class or split.
    """
    if len(records) != len(pixels):
        raise ValueError("records and pixels must be aligned")
    rng = random.Random(seed)
    by_hash = defaultdict(list)
    for index, record in enumerate(records):
        by_hash[record.sha256].append(index)
    conflicting = [
        group for group in by_hash.values() if len({records[i].label for i in group}) > 1
    ]
    if conflicting:
        raise ValueError(f"{len(conflicting)} identical images have conflicting labels")
    unique = [min(group, key=lambda i: records[i].image_id) for group in by_hash.values()]
    unique.sort(key=lambda i: records[i].image_id)
    capacity = len(CLASSES) * (train_per_class + validation_per_class + test_per_class)
    buffer = np.empty((capacity, pixels.shape[1]), dtype=np.float64)
    buffer_norms = np.empty(capacity, dtype=np.float64)
    count = 0
    selected: list[tuple[int, str]] = []
    rejected = {stage: Counter() for stage in ("test", "train_validation")}
    rejected_cross_class = Counter()
    labels_of_selected: list[str] = []

    def accept(index: int, stage: str) -> bool:
        nonlocal count
        if count:
            candidate = pixels[index].astype(np.float64)
            squared = buffer_norms[:count] + candidate @ candidate - 2 * buffer[:count] @ candidate
            distances = np.sqrt(np.maximum(squared, 0.0) / len(candidate))
            close = distances <= threshold
            if close.any():
                label = records[index].label
                rejected[stage][label] += 1
                if any(labels_of_selected[j] != label for j in np.flatnonzero(close)):
                    rejected_cross_class[label] += 1
                return False
        buffer[count] = pixels[index]
        buffer_norms[count] = buffer[count] @ buffer[count]
        count += 1
        labels_of_selected.append(records[index].label)
        return True

    stages = (
        ("test", test_per_class, True),
        ("train_validation", train_per_class + validation_per_class, False),
    )
    taken = set()
    for stage, quota, fresh_only in stages:
        for label in CLASSES:
            candidates = [
                i
                for i in unique
                if records[i].label == label
                and i not in taken
                and not (fresh_only and records[i].image_id in old_manifest_ids)
            ]
            rng.shuffle(candidates)
            chosen = []
            for index in candidates:
                if accept(index, stage):
                    chosen.append(index)
                    taken.add(index)
                    if len(chosen) == quota:
                        break
            if len(chosen) < quota:
                raise ValueError(
                    f"Cannot select {quota} separated images for {label}/{stage}; "
                    f"found {len(chosen)}"
                )
            if stage == "test":
                selected.extend((index, "test") for index in chosen)
            else:
                selected.extend((index, "validation") for index in chosen[:validation_per_class])
                selected.extend((index, "train") for index in chosen[validation_per_class:])
    items = []
    for index, split in selected:
        item = asdict(records[index])
        item["split"] = split
        item["in_old_manifest"] = records[index].image_id in old_manifest_ids
        items.append(item)
    order = sorted(
        range(len(items)), key=lambda k: (CLASSES.index(items[k]["label"]), items[k]["image_id"])
    )
    items = [items[k] for k in order]
    chosen_pixels = pixels[[selected[k][0] for k in order]]
    counts = {label: {split: 0 for split in SPLITS} for label in CLASSES}
    old_by_split = {split: 0 for split in SPLITS}
    for item in items:
        counts[item["label"]][item["split"]] += 1
        old_by_split[item["split"]] += item["in_old_manifest"]
    available = Counter(records[i].label for i in unique)
    return {
        "seed": seed,
        "classes": list(CLASSES),
        "width": WIDTH,
        "height": HEIGHT,
        "selection_method": "greedy_global_pixel_rms_dedup",
        "dedup": {
            "metric": "rms_pixel_distance_on_[0,1]_bt601_luma",
            "threshold_rms": threshold,
            "duplicate_if": "rms <= threshold",
            "scope": "all selected images, across classes and splits",
            "order": "test (fresh images only) first, then train+validation; classes in CLASSES "
            "order; candidates shuffled with random.Random(seed)",
        },
        "exact_duplicates_removed_before_sampling": len(records) - len(unique),
        "unique_images_by_class": {label: available[label] for label in CLASSES},
        "rejected_near_duplicates_by_stage_and_class": {
            stage: {label: counter[label] for label in CLASSES}
            for stage, counter in rejected.items()
        },
        "rejected_near_duplicates_of_other_class": {
            label: rejected_cross_class[label] for label in CLASSES
        },
        "requested_per_class": {
            "train": train_per_class,
            "validation": validation_per_class,
            "test": test_per_class,
        },
        "split_counts_by_class": counts,
        "split_totals": {split: sum(counts[c][split] for c in CLASSES) for split in SPLITS},
        "old_manifest_excluded_from_test": True,
        "old_manifest_images_by_split": old_by_split,
        "min_cross_split_rms": min_cross_split_rms(
            chosen_pixels, [item["split"] for item in items]
        ),
        "records": items,
    }


def load_arrays(archive: Path, manifest: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    groups = {split: ([], []) for split in ("train", "validation", "test")}
    with zipfile.ZipFile(archive) as source:
        for item in manifest["records"]:
            features, labels = groups[item["split"]]
            features.append(decode_pixels(source.read(item["path"])))
            labels.append(CLASSES.index(item["label"]))
    return {
        split: (np.stack(features), np.asarray(labels, dtype=np.int64))
        for split, (features, labels) in groups.items()
    }
