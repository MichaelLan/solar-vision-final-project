import math

from domain.model import HEIGHT, PIXELS, WIDTH, Classifier, Prediction


class InvalidImageTensor(ValueError):
    pass


def validate_pixel_types(value: object) -> object:
    if not isinstance(value, list) or any(type(pixel) not in (int, float) for pixel in value):
        raise ValueError("Pixels must be a list of JSON numbers")
    return value


def validate_dimension_type(value: object) -> object:
    if type(value) is not int:
        raise ValueError("Image dimensions must be JSON integers")
    return value


class ClassificationService:
    def __init__(self, classifier: Classifier):
        self.classifier = classifier

    def classify(self, pixels: list[float], width: int, height: int) -> Prediction:
        if width != WIDTH or height != HEIGHT or len(pixels) != PIXELS:
            raise InvalidImageTensor(
                f"Expected a {WIDTH} × {HEIGHT} grayscale tensor with {PIXELS} pixels"
            )
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in pixels):
            raise InvalidImageTensor("Pixels must be finite numbers in [0, 1]")
        return self.classifier.predict(pixels)
