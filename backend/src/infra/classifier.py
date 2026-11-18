"""Pure-Python fully connected classifier for the v1 ``dense`` artifact format."""

from domain.model import PIXELS, Prediction
from infra.artifact import (
    finite_number,
    normalize_list,
    parse_classes,
    parse_input,
    parse_sha256,
    softmax_list,
)


class DenseClassifier:
    architecture = "dense"

    def __init__(self, model: dict):
        if model.get("architecture", "dense") != "dense":
            raise ValueError("DenseClassifier only supports the dense architecture")
        self.normalization = parse_input(model)
        self.classes = parse_classes(model)
        layers = model.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("Missing layers")
        input_size = PIXELS
        for index, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError("Invalid layer")  # noqa: TRY004
            weights = layer.get("weights")
            biases = layer.get("bias")
            if (
                not isinstance(weights, list)
                or not isinstance(biases, list)
                or len(weights) != len(biases)
                or not weights
            ):
                raise ValueError("Invalid layer shape")
            if any(not isinstance(row, list) or len(row) != input_size for row in weights):
                raise ValueError("Invalid weight shape")
            if not all(finite_number(value) for row in weights for value in row):
                raise ValueError("Invalid model weights")
            if not all(finite_number(value) for value in biases):
                raise ValueError("Invalid model biases")
            if layer.get("activation") != ("softmax" if index == len(layers) - 1 else "relu"):
                raise ValueError("Invalid layer activation")
            input_size = len(biases)
        if input_size != len(self.classes):
            raise ValueError("Output size does not match classes")
        self._layers = layers
        self.version = str(model.get("version", "unknown"))
        self.sha256 = parse_sha256(model)

    def predict(self, pixels: list[float]) -> Prediction:
        values = normalize_list(pixels, self.normalization)
        for layer in self._layers:
            values = [
                sum(weight * value for weight, value in zip(row, values, strict=True)) + bias
                for row, bias in zip(layer["weights"], layer["bias"], strict=True)
            ]
            if layer["activation"] == "relu":
                values = [max(0.0, value) for value in values]
        probabilities = softmax_list(values)
        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        return Prediction(
            class_name=self.classes[winner],
            confidence=probabilities[winner],
            probabilities=dict(zip(self.classes, probabilities, strict=True)),
        )
