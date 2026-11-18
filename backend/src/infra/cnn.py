"""numpy forward pass for the ``cnn`` artifact format (sequential layers, CHW layout).

Conventions match PyTorch: weights are flat row-major ``(out, in, kh, kw)`` for conv2d
(cross-correlation, zero padding) and ``(out, in)`` for dense; max pooling uses floor mode;
a dense layer applied to a CHW tensor flattens it in C, H, W order.
"""

import numpy as np

from domain.model import CHANNELS, HEIGHT, PIXELS, WIDTH, Prediction
from infra.artifact import finite_number, parse_classes, parse_input, parse_sha256

LAYER_TYPES = {"conv2d", "relu", "maxpool2d", "global_avg_pool", "flatten", "dense", "softmax"}


def _pair(value: object, name: str, minimum: int) -> tuple[int, int]:
    if type(value) is int:
        value = [value, value]
    if (
        not isinstance(value, list | tuple)
        or len(value) != 2
        or any(type(item) is not int or item < minimum for item in value)
    ):
        raise ValueError(f"Invalid {name}")
    return value[0], value[1]


def _array(values: object, size: int, name: str) -> np.ndarray:
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f"Invalid {name} size")
    if not all(finite_number(value) for value in values):
        raise ValueError(f"Invalid {name} values")
    return np.asarray(values, dtype=np.float64)


def _windows(x: np.ndarray, kernel: tuple[int, int], stride: tuple[int, int]) -> np.ndarray:
    view = np.lib.stride_tricks.sliding_window_view(x, kernel, axis=(1, 2))
    return view[:, :: stride[0], :: stride[1]]  # (C, Ho, Wo, kh, kw)


def _out_size(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


class ConvClassifier:
    """Sequential typed-layer network; also runs v2 ``dense`` artifacts (dense + softmax)."""

    def __init__(self, model: dict):
        if model.get("architecture") not in {"cnn", "dense"}:
            raise ValueError("ConvClassifier supports the cnn and typed dense architectures")
        self.architecture = model["architecture"]
        self.normalization = parse_input(model)
        self.classes = parse_classes(model)
        self.version = str(model.get("version", "unknown"))
        self.sha256 = parse_sha256(model)
        layers = model.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("Missing layers")
        self._ops = []
        shape: tuple[int, ...] = (CHANNELS, HEIGHT, WIDTH)
        for index, layer in enumerate(layers):
            kind = layer.get("type") if isinstance(layer, dict) else None
            if kind not in LAYER_TYPES:
                raise ValueError(f"Unsupported layer {index}")
            if (kind == "softmax") != (index == len(layers) - 1):
                raise ValueError("Softmax must be the last layer")
            shape = self._compile(index, kind, layer, shape)
        if shape != (len(self.classes),):
            raise ValueError("Output size does not match classes")

    def _compile(self, index: int, kind: str, layer: dict, shape: tuple[int, ...]):
        if kind in {"conv2d", "maxpool2d", "global_avg_pool"} and len(shape) != 3:
            raise ValueError(f"Layer {index} requires a CHW tensor")
        if kind == "conv2d":
            channels, height, width = shape
            if layer.get("in_channels") != channels:
                raise ValueError(f"Layer {index}: in_channels mismatch")
            out_channels = layer.get("out_channels")
            if type(out_channels) is not int or out_channels < 1:
                raise ValueError(f"Layer {index}: invalid out_channels")
            kernel = _pair(layer.get("kernel"), "kernel", 1)
            stride = _pair(layer.get("stride", 1), "stride", 1)
            padding = _pair(layer.get("padding", 0), "padding", 0)
            size = out_channels * channels * kernel[0] * kernel[1]
            weight = _array(layer.get("weight"), size, f"layer {index} weight")
            bias = _array(layer.get("bias"), out_channels, f"layer {index} bias")
            out_h = _out_size(height, kernel[0], stride[0], padding[0])
            out_w = _out_size(width, kernel[1], stride[1], padding[1])
            if out_h < 1 or out_w < 1:
                raise ValueError(f"Layer {index}: kernel larger than input")
            self._ops.append(
                (kind, weight.reshape(out_channels, -1), bias[:, None], kernel, stride, padding)
            )
            return out_channels, out_h, out_w
        if kind == "maxpool2d":
            channels, height, width = shape
            kernel = _pair(layer.get("kernel"), "kernel", 1)
            stride = _pair(layer.get("stride", list(kernel)), "stride", 1)
            out_h = _out_size(height, kernel[0], stride[0], 0)
            out_w = _out_size(width, kernel[1], stride[1], 0)
            if out_h < 1 or out_w < 1:
                raise ValueError(f"Layer {index}: pool larger than input")
            self._ops.append((kind, kernel, stride))
            return channels, out_h, out_w
        if kind == "global_avg_pool":
            self._ops.append((kind,))
            return (shape[0],)
        if kind == "flatten":
            self._ops.append((kind,))
            return (int(np.prod(shape)),)
        if kind == "dense":
            features = int(np.prod(shape))
            if layer.get("in_features") != features:
                raise ValueError(f"Layer {index}: in_features mismatch")
            out_features = layer.get("out_features")
            if type(out_features) is not int or out_features < 1:
                raise ValueError(f"Layer {index}: invalid out_features")
            weight = _array(layer.get("weight"), out_features * features, f"layer {index} weight")
            bias = _array(layer.get("bias"), out_features, f"layer {index} bias")
            self._ops.append((kind, weight.reshape(out_features, features), bias))
            return (out_features,)
        self._ops.append((kind,))  # relu, softmax keep the shape
        return shape

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        spec = self.normalization
        if spec["type"] == "global":
            return (x - spec["mean"]) / spec["std"]
        if spec["type"] == "per_image":
            return (x - x.mean()) / (x.std() + spec["eps"])
        return x

    def forward(self, pixels: list[float]) -> np.ndarray:
        x = np.asarray(pixels, dtype=np.float64)
        if x.shape != (PIXELS,):
            raise ValueError(f"Expected {PIXELS} pixels")
        x = self._normalize(x).reshape(CHANNELS, HEIGHT, WIDTH)
        for op in self._ops:
            kind = op[0]
            if kind == "conv2d":
                _, weight, bias, kernel, stride, (pad_h, pad_w) = op
                if pad_h or pad_w:
                    x = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)))
                windows = _windows(x, kernel, stride)
                out_h, out_w = windows.shape[1:3]
                cols = windows.transpose(0, 3, 4, 1, 2).reshape(weight.shape[1], out_h * out_w)
                x = (weight @ cols + bias).reshape(weight.shape[0], out_h, out_w)
            elif kind == "relu":
                x = np.maximum(x, 0.0)
            elif kind == "maxpool2d":
                x = _windows(x, op[1], op[2]).max(axis=(3, 4))
            elif kind == "global_avg_pool":
                x = x.mean(axis=(1, 2))
            elif kind == "flatten":
                x = x.reshape(-1)
            elif kind == "dense":
                x = op[1] @ x.reshape(-1) + op[2]
            else:  # softmax
                x = np.exp(x - x.max())
                x = x / x.sum()
        return x

    def predict(self, pixels: list[float]) -> Prediction:
        probabilities = [float(value) for value in self.forward(pixels)]
        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        return Prediction(
            class_name=self.classes[winner],
            confidence=probabilities[winner],
            probabilities=dict(zip(self.classes, probabilities, strict=True)),
        )
