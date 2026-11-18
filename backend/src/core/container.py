from functools import lru_cache

from domain.model import Classifier, ModelUnavailable
from infra.classifier import DenseClassifier
from infra.cnn import ConvClassifier
from infra.model_payload import MODEL
from services.classification import ClassificationService


def build_classifier(model: dict) -> Classifier:
    architecture = model.get("architecture", "dense")
    if architecture == "cnn":
        return ConvClassifier(model)
    if architecture == "dense":
        layers = model.get("layers")
        typed = isinstance(layers, list) and bool(layers) and isinstance(layers[0], dict)
        # v2 artifacts use typed layers; v1 used {"weights", "bias", "activation"} rows.
        return ConvClassifier(model) if typed and "type" in layers[0] else DenseClassifier(model)
    raise ValueError(f"Unsupported model architecture: {architecture!r}")


@lru_cache(maxsize=1)
def get_classifier() -> Classifier:
    if MODEL is None:
        raise ModelUnavailable("Trained model artifact has not been packaged")
    try:
        return build_classifier(MODEL)
    except ValueError as exc:
        raise ModelUnavailable(f"Invalid model artifact: {exc}") from exc


async def get_service() -> ClassificationService:
    return ClassificationService(get_classifier())
