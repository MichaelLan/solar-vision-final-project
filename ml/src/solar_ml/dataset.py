import hashlib
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from solar_ml.constants import ARCHIVE_NAME, CLASSES, HEIGHT, SEED, SOURCE_URL, WIDTH
from solar_ml.images import decode_pixels, perceptual_hash


@dataclass
class Record:
    image_id: str
    path: str
    label: str
    sha256: str
    perceptual_hash: str
    quantized_hash: str
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
            records.append(
                Record(
                    image_id=str(image_id),
                    path=full_path,
                    label=label,
                    sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                    perceptual_hash=f"{perceptual_hash(pixels):016x}",
                    quantized_hash=hashlib.sha256(
                        np.floor(pixels * 15).astype(np.uint8).tobytes()
                    ).hexdigest(),
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
        return info, {"records": records, "metadata": metadata}


def make_manifest(records: list[Record], per_class: int = 400, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)
    conflicting = [group for group in by_hash.values() if len({r.label for r in group}) > 1]
    if conflicting:
        raise ValueError(f"{len(conflicting)} identical images have conflicting labels")
    unique = [sorted(group, key=lambda r: r.image_id)[0] for group in by_hash.values()]
    selected = []
    selected_hashes = []
    rejected_near_duplicates = Counter()
    priority = ("Diode", "Cell-Multi", "Cell", "No-Anomaly")
    for label in priority:
        candidates = sorted((r for r in unique if r.label == label), key=lambda r: r.image_id)
        if len(candidates) < per_class:
            raise ValueError(f"Class {label} has only {len(candidates)} unique images")
        rng.shuffle(candidates)
        chosen = 0
        for record in candidates:
            candidate_hash = int(record.perceptual_hash, 16)
            if any((candidate_hash ^ prior).bit_count() <= 2 for prior in selected_hashes):
                rejected_near_duplicates[label] += 1
                continue
            selected.append(record)
            selected_hashes.append(candidate_hash)
            chosen += 1
            if chosen == per_class:
                break
        if chosen < per_class:
            raise ValueError(
                f"Cannot select {per_class} dHash-separated images for {label}; found {chosen}"
            )
    groups = defaultdict(list)
    for record in selected:
        groups[record.quantized_hash].append(record)
    grouped = list(groups.values())
    rng.shuffle(grouped)
    desired = {
        label: {
            "train": round(per_class * 0.70),
            "validation": round(per_class * 0.15),
            "test": per_class - round(per_class * 0.70) - round(per_class * 0.15),
        }
        for label in CLASSES
    }
    current = {label: {split: 0 for split in ("train", "validation", "test")} for label in CLASSES}
    for group in grouped:
        group_counts = Counter(record.label for record in group)
        split = min(
            ("train", "validation", "test"),
            key=lambda candidate: (
                sum(
                    (current[label][candidate] + count - desired[label][candidate]) ** 2
                    - (current[label][candidate] - desired[label][candidate]) ** 2
                    for label, count in group_counts.items()
                ),
                -sum(current[label][candidate] == 0 for label in group_counts),
            ),
        )
        for record in group:
            record.split = split
            current[record.label][split] += 1
    tolerance = max(2, math.ceil(per_class * 0.05))
    for label in CLASSES:
        for split in ("train", "validation", "test"):
            observed = current[label][split]
            if observed == 0 or abs(observed - desired[label][split]) > tolerance:
                raise ValueError(
                    f"Quantized-image grouping prevents a stratified split for {label}/{split}: "
                    f"observed {observed}, desired {desired[label][split]}"
                )
    near_pair_count = 0
    near_cross_split_count = 0
    hashes = [(int(record.perceptual_hash, 16), record.split) for record in selected]
    for left in range(len(hashes)):
        left_hash, left_split = hashes[left]
        for right_hash, right_split in hashes[left + 1 :]:
            if (left_hash ^ right_hash).bit_count() <= 2:
                near_pair_count += 1
                near_cross_split_count += left_split != right_split
    selected.sort(key=lambda r: (CLASSES.index(r.label), r.image_id))
    return {
        "seed": seed,
        "per_class": per_class,
        "selection_priority": list(priority),
        "selection_method": "greedy_global_dhash_hamming_gt_2",
        "rejected_near_duplicates_by_class": dict(rejected_near_duplicates),
        "classes": list(CLASSES),
        "width": WIDTH,
        "height": HEIGHT,
        "exact_duplicates_removed_before_sampling": len(records) - len(unique),
        "dhash_pairs_hamming_le_2": near_pair_count,
        "dhash_cross_split_pairs_hamming_le_2": near_cross_split_count,
        "quantized_groups": len(grouped),
        "quantized_duplicate_images": len(selected) - len(grouped),
        "largest_quantized_group": max(map(len, grouped)),
        "split_counts_by_class": current,
        "records": [asdict(record) for record in selected],
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
