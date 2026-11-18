"""JSON model contract (version 2.0.0) and a pure-numpy reference forward pass."""

import hashlib
import json
from pathlib import Path

import numpy as np

from solar_ml.constants import CLASSES, FEATURE_COUNT, HEIGHT, WIDTH

VERSION = "2.0.0"
SIGNIFICANT_DIGITS = 7


def _round(values) -> list[float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(f"{value:.{SIGNIFICANT_DIGITS}g}") for value in flat]


def input_spec(normalization: dict) -> dict:
    return {
        "width": WIDTH,
        "height": HEIGHT,
        "channels": 1,
        "layout": "HW row-major",
        "range": "[0,1] BT.601 luma/255",
        "normalization": normalization,
    }


def model_sha256(artifact: dict) -> str:
    """sha256 of json.dumps({"input", "layers"}, sort_keys=True, separators=(",", ":"))."""
    canonical = json.dumps(
        {"input": artifact["input"], "layers": artifact["layers"]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _finish(architecture: str, normalization: dict, layers: list, metrics_ref: dict | None) -> dict:
    artifact = {
        "version": VERSION,
        "architecture": architecture,
        "classes": list(CLASSES),
        "input": input_spec(normalization),
        "layers": layers,
    }
    artifact["sha256"] = model_sha256(artifact)
    artifact["metrics_ref"] = metrics_ref or {}
    return artifact


def export_cnn(model, normalization: dict, metrics_ref: dict | None = None) -> dict:
    """Export a SmallCNN (no batch normalization, so there is nothing to fold)."""
    layers = []
    convs = [model.conv1, model.conv2, model.conv3]
    for index, conv in enumerate(convs):
        layers.append(
            {
                "type": "conv2d",
                "in_channels": conv.in_channels,
                "out_channels": conv.out_channels,
                "kernel": conv.kernel_size[0],
                "stride": conv.stride[0],
                "padding": conv.padding[0],
                "weight": _round(conv.weight.detach().numpy()),
                "bias": _round(conv.bias.detach().numpy()),
            }
        )
        layers.append({"type": "relu"})
        if index < 2:
            layers.append({"type": "maxpool2d", "kernel": 2, "stride": 2})
    layers.append({"type": "global_avg_pool"})
    layers.append(
        {
            "type": "dense",
            "in_features": model.dense.in_features,
            "out_features": model.dense.out_features,
            "weight": _round(model.dense.weight.detach().numpy()),
            "bias": _round(model.dense.bias.detach().numpy()),
        }
    )
    layers.append({"type": "softmax"})
    return _finish("cnn", normalization, layers, metrics_ref)


def export_dense(model, metrics_ref: dict | None = None) -> dict:
    """Export a multinomial LogisticRegression as dense + softmax on raw pixels."""
    coefficients = np.asarray(model.coef_)
    layers = [
        {
            "type": "dense",
            "in_features": coefficients.shape[1],
            "out_features": coefficients.shape[0],
            "weight": _round(coefficients),
            "bias": _round(model.intercept_),
        },
        {"type": "softmax"},
    ]
    return _finish("dense", {"type": "none"}, layers, metrics_ref)


def write_artifact(artifact: dict, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(artifact, separators=(",", ":"), allow_nan=False)
    destination.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _normalize(values: np.ndarray, spec: dict) -> np.ndarray:
    kind = spec["type"]
    if kind == "none":
        return values
    if kind == "global":
        return (values - spec["mean"]) / spec["std"]
    if kind == "per_image":
        mean = values.mean(axis=1, keepdims=True)
        std = values.std(axis=1, keepdims=True)
        return (values - mean) / (std + spec["eps"])
    raise ValueError(f"Unknown normalization: {kind}")


def _conv2d(x: np.ndarray, layer: dict) -> np.ndarray:
    """x: (N, C, H, W); weight order (out, in, kh, kw)."""
    k, stride, pad = layer["kernel"], layer["stride"], layer["padding"]
    weight = np.asarray(layer["weight"]).reshape(layer["out_channels"], layer["in_channels"], k, k)
    n, c, h, w = x.shape
    if c != layer["in_channels"]:
        raise ValueError("conv2d channel mismatch")
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out_h = (h + 2 * pad - k) // stride + 1
    out_w = (w + 2 * pad - k) // stride + 1
    patches = np.empty((n, c, k, k, out_h, out_w))
    for dy in range(k):
        for dx in range(k):
            patches[:, :, dy, dx] = padded[
                :, :, dy : dy + stride * out_h : stride, dx : dx + stride * out_w : stride
            ]
    out = np.einsum("ncyxhw,ocyx->nohw", patches, weight, optimize=True)
    return out + np.asarray(layer["bias"])[None, :, None, None]


def forward(artifact: dict, pixels: np.ndarray) -> np.ndarray:
    """Reference forward pass. pixels: (N, 960) or (960,) in [0, 1]; returns probabilities."""
    values = np.asarray(pixels, dtype=np.float64)
    single = values.ndim == 1
    values = values.reshape(-1, FEATURE_COUNT)
    values = _normalize(values, artifact["input"]["normalization"])
    x = values
    if artifact["architecture"] == "cnn":
        x = values.reshape(-1, 1, HEIGHT, WIDTH)
    for layer in artifact["layers"]:
        kind = layer["type"]
        if kind == "conv2d":
            x = _conv2d(x, layer)
        elif kind == "relu":
            x = np.maximum(x, 0.0)
        elif kind == "maxpool2d":
            k, s = layer["kernel"], layer["stride"]
            if k != s:
                raise ValueError("Only non-overlapping max pooling is supported")
            n, c, h, w = x.shape
            x = x[:, :, : h // k * k, : w // k * k].reshape(n, c, h // k, k, w // k, k)
            x = x.max(axis=(3, 5))
        elif kind == "global_avg_pool":
            x = x.mean(axis=(2, 3))
        elif kind == "dense":
            weight = np.asarray(layer["weight"]).reshape(
                layer["out_features"], layer["in_features"]
            )
            x = x.reshape(len(x), -1) @ weight.T + np.asarray(layer["bias"])
        elif kind == "softmax":
            x = np.exp(x - x.max(axis=1, keepdims=True))
            x = x / x.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown layer type: {kind}")
    return x[0] if single else x
