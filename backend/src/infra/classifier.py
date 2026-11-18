import math

from domain.model import Prediction

LABELS = {
    "No-Anomaly": "Sin anomalía identificada",
    "Cell": "Anomalía en una celda",
    "Cell-Multi": "Anomalías en varias celdas",
    "Diode": "Patrón asociado con diodo de bypass",
}


class ModelUnavailable(RuntimeError):
    pass


class DenseClassifier:
    def __init__(self, model: dict):
        if model.get("input") != {
            "width": 24,
            "height": 40,
            "channels": 1,
            "normalization": "rounded_bt601_luma_255",
        }:
            raise ValueError("Unsupported model input specification")
        classes = model.get("classes")
        if not isinstance(classes, list) or len(classes) != 4 or set(classes) != set(LABELS):
            raise ValueError("Unsupported model classes")
        layers = model.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("Missing layers")
        input_size = 960
        for index, layer in enumerate(layers):
            weights = layer.get("weights")
            biases = layer.get("bias")
            if not isinstance(weights, list) or not isinstance(biases, list) or len(weights) != len(biases) or not weights:
                raise ValueError("Invalid layer shape")
            if any(not isinstance(row, list) or len(row) != input_size for row in weights):
                raise ValueError("Invalid weight shape")
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) for row in weights for value in row):
                raise ValueError("Invalid model weights")
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in biases):
                raise ValueError("Invalid model biases")
            if layer.get("activation") != ("softmax" if index == len(layers) - 1 else "relu"):
                raise ValueError("Invalid layer activation")
            input_size = len(biases)
        if input_size != len(classes):
            raise ValueError("Output size does not match classes")
        self._layers = layers
        self.classes = tuple(classes)
        self.version = str(model.get("version", "unknown"))

    def predict(self, pixels: list[float]) -> Prediction:
        values = pixels
        for layer in self._layers:
            values = [
                sum(weight * value for weight, value in zip(row, values, strict=True)) + bias
                for row, bias in zip(layer["weights"], layer["bias"], strict=True)
            ]
            if layer["activation"] == "relu":
                values = [max(0.0, value) for value in values]
        maximum = max(values)
        exponentials = [math.exp(value - maximum) for value in values]
        total = sum(exponentials)
        probabilities = [value / total for value in exponentials]
        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        return Prediction(
            class_name=self.classes[winner],
            confidence=probabilities[winner],
            probabilities=dict(zip(self.classes, probabilities, strict=True)),
        )
