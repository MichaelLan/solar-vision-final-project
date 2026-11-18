from functools import lru_cache

from infra.classifier import DenseClassifier, ModelUnavailable
from infra.model_payload import MODEL
from services.classification import ClassificationService


@lru_cache(maxsize=1)
def get_classifier() -> DenseClassifier:
    if MODEL is None:
        raise ModelUnavailable("Trained model artifact has not been packaged")
    return DenseClassifier(MODEL)


async def get_service() -> ClassificationService:
    return ClassificationService(get_classifier())
