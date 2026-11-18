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

        fixture = json.loads((Path(__file__).parent / "fixtures" / "real_image_tensor.json").read_text())
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
        status = next(message["status"] for message in messages if message["type"] == "http.response.start")
        response = json.loads(b"".join(
            message.get("body", b"") for message in messages if message["type"] == "http.response.body"
        ))
        expected = fixture["expected_probabilities"]
        self.assertEqual(status, 200)
        self.assertEqual(response["prediction"]["class"], "Cell")
        for name, probability in expected.items():
            self.assertAlmostEqual(response["probabilities"][name], probability, places=6)
