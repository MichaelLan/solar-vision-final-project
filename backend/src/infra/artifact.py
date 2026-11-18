"""Validation shared by every model artifact format (metadata, classes, input spec)."""

import hashlib
import json
import math

from domain.model import CHANNELS, HEIGHT, LABELS, PIXEL_ENCODING, WIDTH

NORMALIZATION_TYPES = {"none", "global", "per_image"}


def finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def parse_classes(model: dict) -> tuple[str, ...]:
    classes = model.get("classes")
    if not isinstance(classes, list) or len(classes) != len(LABELS) or set(classes) != set(LABELS):
        raise ValueError("Unsupported model classes")
    return tuple(classes)


def parse_normalization(spec: object) -> dict:
    """Return a canonical normalization spec.

    The legacy v1 string ``rounded_bt601_luma_255`` means no server-side normalization.
    ``global``: (x - mean) / std. ``per_image``: (x - mean(x)) / (std(x) + eps), with the
    population standard deviation (ddof=0, as numpy and torch ``std(unbiased=False)``).
    """
    if spec == PIXEL_ENCODING:
        return {"type": "none"}
    if not isinstance(spec, dict) or spec.get("type") not in NORMALIZATION_TYPES:
        raise ValueError("Unsupported input normalization")
    kind = spec["type"]
    if kind == "none":
        return {"type": "none"}
    if kind == "global":
        mean, std = spec.get("mean"), spec.get("std")
        if isinstance(mean, list) and len(mean) == CHANNELS:
            mean = mean[0]
        if isinstance(std, list) and len(std) == CHANNELS:
            std = std[0]
        if not finite_number(mean) or not finite_number(std) or std <= 0:
            raise ValueError("Global normalization requires finite mean and positive std")
        return {"type": "global", "mean": float(mean), "std": float(std)}
    eps = spec.get("eps", 1e-6)
    if not finite_number(eps) or eps < 0:
        raise ValueError("Per-image normalization requires a non-negative eps")
    return {"type": "per_image", "eps": float(eps)}


def parse_input(model: dict) -> dict:
    spec = model.get("input")
    if not isinstance(spec, dict):
        raise ValueError("Missing model input specification")  # noqa: TRY004
    if (spec.get("width"), spec.get("height"), spec.get("channels")) != (WIDTH, HEIGHT, CHANNELS):
        raise ValueError("Unsupported model input specification")
    return parse_normalization(spec.get("normalization"))


def canonical_sha256(model: dict) -> str:
    """sha256 of the canonical JSON of ``{"input", "layers"}`` (as ``solar_ml.export``)."""
    canonical = json.dumps(
        {"input": model.get("input"), "layers": model.get("layers")},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def parse_sha256(model: dict) -> str | None:
    """Return the declared sha256 after checking it against the weights, or None if absent."""
    value = model.get("sha256")
    if value is None:
        return None
    if not isinstance(value, str) or value.lower() != canonical_sha256(model):
        raise ValueError("Model sha256 does not match its input and layers")
    return value.lower()


def normalize_list(pixels: list[float], spec: dict) -> list[float]:
    if spec["type"] == "global":
        return [(value - spec["mean"]) / spec["std"] for value in pixels]
    if spec["type"] == "per_image":
        mean = sum(pixels) / len(pixels)
        std = math.sqrt(sum((value - mean) ** 2 for value in pixels) / len(pixels))
        return [(value - mean) / (std + spec["eps"]) for value in pixels]
    return pixels


def softmax_list(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]
