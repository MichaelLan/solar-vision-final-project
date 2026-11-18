from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Prediction:
    class_name: str
    confidence: float
    probabilities: dict[str, float]


class Classifier(Protocol):
    @property
    def version(self) -> str: ...

    @property
    def classes(self) -> tuple[str, ...]: ...

    def predict(self, pixels: list[float]) -> Prediction: ...
