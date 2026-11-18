import asyncio
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI is required")
class RealArtifactTests(unittest.IsolatedAsyncioTestCase):
    async def test_demo_image_matches_training_reference(self):
        from api.app import app

        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "real_image_tensor.json").read_text()
        )
        sample = ROOT / fixture["source"]
        self.assertEqual(hashlib.sha256(sample.read_bytes()).hexdigest(), fixture["source_sha256"])
        payload = {
            "pixels": fixture["pixels"],
            "width": 24,
            "height": 40,
        }
        body = json.dumps(payload).encode()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "method": "POST",
            "path": "/api/predict",
            "raw_path": b"/api/predict",
            "root_path": "",
            "scheme": "http",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("test", 1),
            "server": ("test", 80),
            "http_version": "1.1",
        }
        messages = []
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        await app(scope, receive, send)
        status = next(
            message["status"] for message in messages if message["type"] == "http.response.start"
        )
        response = json.loads(
            b"".join(
                message.get("body", b"")
                for message in messages
                if message["type"] == "http.response.body"
            )
        )
        expected = fixture["expected_probabilities"]
        self.assertEqual(status, 200)
        self.assertEqual(response["prediction"]["class"], fixture["expected_class"])
        self.assertEqual(response["prediction"]["class"], max(expected, key=expected.get))
        self.assertEqual(set(response["probabilities"]), set(expected))
        for name, probability in expected.items():
            self.assertLess(abs(response["probabilities"][name] - probability), 1e-6)

        from core.container import get_classifier
        from infra.artifact import canonical_sha256
        from infra.model_payload import MODEL

        self.assertEqual(fixture["model_sha256"], MODEL["sha256"])
        self.assertEqual(canonical_sha256(MODEL), MODEL["sha256"])
        self.assertEqual(get_classifier().sha256, MODEL["sha256"])
        self.assertEqual(response["model_version"], MODEL["version"])

    def test_fixture_matches_artifact_reference(self):
        reference = ROOT / "artifacts" / "reference_prediction.json"
        if not reference.is_file():
            self.skipTest("artifacts/reference_prediction.json is not available")
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "real_image_tensor.json").read_text()
        )
        self.assertEqual(fixture, json.loads(reference.read_text()))
