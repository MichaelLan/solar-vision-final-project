import asyncio
import importlib.util
import json
import unittest


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI is not installed")
class ApiTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        from api.app import app
        from core import container

        cls.app = app
        cls.container = container
        cls.original_model = container.MODEL
        container.MODEL = {
            "version": "test",
            "classes": ["No-Anomaly", "Cell", "Cell-Multi", "Diode"],
            "input": {
                "width": 24,
                "height": 40,
                "channels": 1,
                "normalization": "rounded_bt601_luma_255",
            },
            "layers": [{
                "weights": [[value] * 960 for value in [0.0, 0.5, -0.5, 0.1]],
                "bias": [0.0, 0.0, 0.0, 0.0],
                "activation": "softmax",
            }],
        }
        container.get_classifier.cache_clear()

    @classmethod
    def tearDownClass(cls):
        cls.container.MODEL = cls.original_model
        cls.container.get_classifier.cache_clear()

    async def request(self, method: str, path: str, payload=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "scheme": "http",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("test", 1),
            "server": ("test", 80),
            "http_version": "1.1",
        }
        messages = []
        received = False

        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        await self.app(scope, receive, send)
        status = next(item["status"] for item in messages if item["type"] == "http.response.start")
        content = b"".join(item.get("body", b"") for item in messages if item["type"] == "http.response.body")
        return status, json.loads(content)

    async def test_health_and_model_info(self):
        status, body = await self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["model_ready"])
        status, body = await self.request("GET", "/api/model/info")
        self.assertEqual(status, 200)
        self.assertEqual(body["input"]["width"], 24)

    async def test_real_inference_path_and_validation(self):
        payload = {"pixels": [1] * 960, "width": 24, "height": 40}
        status, body = await self.request("POST", "/api/predict", payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["prediction"]["class"], "Cell")
        self.assertAlmostEqual(sum(body["probabilities"].values()), 1.0)
        payload["pixels"][0] = "1"
        status, _ = await self.request("POST", "/api/predict", payload)
        self.assertEqual(status, 422)
        payload["pixels"][0] = 1
        payload["width"] = True
        status, _ = await self.request("POST", "/api/predict", payload)
        self.assertEqual(status, 422)

    async def test_body_limit(self):
        status, _ = await self.request("POST", "/api/predict", {"pixels": [0] * 70000})
        self.assertEqual(status, 413)
