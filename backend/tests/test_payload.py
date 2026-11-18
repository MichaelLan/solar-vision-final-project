import json
import unittest
from pathlib import Path

from core.container import build_classifier
from infra.model_payload import MODEL

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "model.json"


@unittest.skipUnless(ARTIFACT.is_file(), "artifacts/model.json is not available")
class PackagedPayloadTests(unittest.TestCase):
    def test_payload_matches_artifact(self):
        artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        self.assertEqual(
            MODEL,
            artifact,
            "backend/src/infra/model_payload.py is stale; run scripts/package_model.py",
        )

    def test_payload_loads(self):
        classifier = build_classifier(MODEL)
        self.assertEqual(classifier.version, str(MODEL["version"]))
        self.assertEqual(classifier.architecture, MODEL.get("architecture", "dense"))


BASELINE = ARTIFACT.parent / "baseline_model.json"


@unittest.skipUnless(BASELINE.is_file(), "artifacts/baseline_model.json is not available")
class BaselineArtifactTests(unittest.TestCase):
    def test_typed_dense_baseline_loads(self):
        model = json.loads(BASELINE.read_text(encoding="utf-8"))
        classifier = build_classifier(model)
        self.assertEqual(classifier.architecture, "dense")
        self.assertEqual(classifier.sha256, model["sha256"])
        result = classifier.predict([0.5] * 960)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)
