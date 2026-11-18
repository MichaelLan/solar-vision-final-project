from dataclasses import dataclass
from typing import Protocol

WIDTH = 24
HEIGHT = 40
CHANNELS = 1
PIXELS = WIDTH * HEIGHT
PIXEL_ENCODING = "rounded_bt601_luma_255"

LABELS = {
    "No-Anomaly": "Sin anomalía identificada",
    "Cell": "Anomalía en una celda",
    "Cell-Multi": "Anomalías en varias celdas",
    "Diode": "Patrón asociado con diodo de bypass",
}


class ModelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    class_name: str
    confidence: float
    probabilities: dict[str, float]


class Classifier(Protocol):
    @property
    def version(self) -> str: ...

    @property
    def architecture(self) -> str: ...

    @property
    def sha256(self) -> str | None: ...

    @property
    def classes(self) -> tuple[str, ...]: ...

    @property
    def normalization(self) -> dict: ...

    def predict(self, pixels: list[float]) -> Prediction: ...
