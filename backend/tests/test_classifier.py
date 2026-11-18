import unittest

from infra.classifier import DenseClassifier
from services.classification import (
    ClassificationService,
    InvalidImageTensor,
    validate_dimension_type,
    validate_pixel_types,
)


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.model = {
            "version": "test",
            "classes": ["No-Anomaly", "Cell", "Cell-Multi", "Diode"],
            "input": {
                "width": 24,
                "height": 40,
                "channels": 1,
                "normalization": "rounded_bt601_luma_255",
            },
            "layers": [
                {
                    "weights": [[value] * 960 for value in [0.0, 0.5, -0.5, 0.1]],
                    "bias": [0.0, 0.0, 0.0, 0.0],
                    "activation": "softmax",
                }
            ],
        }

    def test_prediction_probabilities(self):
        service = ClassificationService(DenseClassifier(self.model))
        result = service.classify([1.0] * 960, 24, 40)
        self.assertEqual(result.class_name, "Cell")
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)
        self.assertTrue(all(0 <= value <= 1 for value in result.probabilities.values()))

    def test_invalid_tensor(self):
        service = ClassificationService(DenseClassifier(self.model))
        with self.assertRaises(InvalidImageTensor):
            service.classify([0.0] * 959, 24, 40)
        with self.assertRaises(InvalidImageTensor):
            service.classify([float("nan")] * 960, 24, 40)
        with self.assertRaises(InvalidImageTensor):
            service.classify([1.1] * 960, 24, 40)

    def test_invalid_model_shape(self):
        self.model["layers"][0]["weights"][0] = [0.0]
        with self.assertRaises(ValueError):
            DenseClassifier(self.model)

    def test_json_number_types(self):
        self.assertEqual(validate_pixel_types([0, 0.5, 1]), [0, 0.5, 1])
        self.assertEqual(validate_dimension_type(40), 40)
        for invalid in (["0.5"], [True], "pixels", {"0": 0.5}):
            with self.assertRaises(ValueError):
                validate_pixel_types(invalid)
        for invalid in (40.0, "40", True):
            with self.assertRaises(ValueError):
                validate_dimension_type(invalid)


if __name__ == "__main__":
    unittest.main()
