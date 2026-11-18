import copy
import math
import random
import unittest

from core.container import build_classifier
from infra.artifact import canonical_sha256
from infra.cnn import ConvClassifier

CLASSES = ["No-Anomaly", "Cell", "Cell-Multi", "Diode"]
H, W = 40, 24


def rand(rng: random.Random, size: int) -> list[float]:
    return [rng.uniform(-0.5, 0.5) for _ in range(size)]


def conv(cin, cout, kernel=3, stride=1, padding=1, rng=None):
    return {
        "type": "conv2d",
        "in_channels": cin,
        "out_channels": cout,
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "weight": rand(rng, cout * cin * kernel * kernel),
        "bias": rand(rng, cout),
    }


def dense(fin, fout, rng):
    return {
        "type": "dense",
        "in_features": fin,
        "out_features": fout,
        "weight": rand(rng, fout * fin),
        "bias": rand(rng, fout),
    }


def artifact(layers, normalization=None):
    return {
        "version": "2.0.0-test",
        "architecture": "cnn",
        "classes": list(CLASSES),
        "input": {
            "width": W,
            "height": H,
            "channels": 1,
            "normalization": normalization or {"type": "none"},
        },
        "layers": layers,
    }


def reference_cnn(rng):
    return [
        conv(1, 4, rng=rng),
        {"type": "relu"},
        {"type": "maxpool2d", "kernel": 2, "stride": 2},
        conv(4, 6, rng=rng),
        {"type": "relu"},
        {"type": "maxpool2d", "kernel": 2, "stride": 2},
        conv(6, 8, rng=rng),
        {"type": "relu"},
        {"type": "global_avg_pool"},
        dense(8, 4, rng),
        {"type": "softmax"},
    ]


# Independent pure-Python reference (nested loops, no numpy).
def naive_forward(model, pixels):
    norm = model["input"]["normalization"]
    values = list(pixels)
    if norm["type"] == "global":
        mean, std = (
            item[0] if isinstance(item, list) else item for item in (norm["mean"], norm["std"])
        )
        values = [(v - mean) / std for v in values]
    elif norm["type"] == "per_image":
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        values = [(v - mean) / (std + norm.get("eps", 1e-6)) for v in values]
    x = [[[values[r * W + c] for c in range(W)] for r in range(H)]]
    for layer in model["layers"]:
        kind = layer["type"]
        if kind == "conv2d":
            k, s, p = layer["kernel"], layer["stride"], layer["padding"]
            cin, cout = layer["in_channels"], layer["out_channels"]
            h, w = len(x[0]), len(x[0][0])
            oh, ow = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
            wt = layer["weight"]
            out = []
            for o in range(cout):
                plane = []
                for i in range(oh):
                    row = []
                    for j in range(ow):
                        acc = layer["bias"][o]
                        for ci in range(cin):
                            for a in range(k):
                                for b in range(k):
                                    r, c = i * s + a - p, j * s + b - p
                                    if 0 <= r < h and 0 <= c < w:
                                        weight = wt[((o * cin + ci) * k + a) * k + b]
                                        acc += weight * x[ci][r][c]
                        row.append(acc)
                    plane.append(row)
                out.append(plane)
            x = out
        elif kind == "relu":
            if isinstance(x[0], list):
                x = [[[max(0.0, v) for v in row] for row in plane] for plane in x]
            else:
                x = [max(0.0, v) for v in x]
        elif kind == "maxpool2d":
            k, s = layer["kernel"], layer["stride"]
            h, w = len(x[0]), len(x[0][0])
            oh, ow = (h - k) // s + 1, (w - k) // s + 1
            x = [
                [
                    [
                        max(plane[i * s + a][j * s + b] for a in range(k) for b in range(k))
                        for j in range(ow)
                    ]
                    for i in range(oh)
                ]
                for plane in x
            ]
        elif kind == "global_avg_pool":
            x = [sum(sum(row) for row in plane) / (len(plane) * len(plane[0])) for plane in x]
        elif kind in {"flatten", "dense"}:
            if isinstance(x[0], list):
                x = [v for plane in x for row in plane for v in row]
            if kind == "dense":
                fin = layer["in_features"]
                x = [
                    sum(layer["weight"][o * fin + i] * x[i] for i in range(fin)) + layer["bias"][o]
                    for o in range(layer["out_features"])
                ]
        elif kind == "softmax":
            m = max(x)
            e = [math.exp(v - m) for v in x]
            x = [v / sum(e) for v in e]
    return x


class ConvClassifierTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(7)
        self.pixels = [self.rng.random() for _ in range(H * W)]

    def assert_matches_reference(self, model):
        classifier = build_classifier(model)
        self.assertIsInstance(classifier, ConvClassifier)
        result = classifier.predict(self.pixels)
        expected = naive_forward(model, self.pixels)
        for name, value in zip(CLASSES, expected, strict=True):
            self.assertAlmostEqual(result.probabilities[name], value, places=10)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)
        self.assertEqual(result.class_name, CLASSES[expected.index(max(expected))])

    def test_reference_architecture(self):
        self.assert_matches_reference(artifact(reference_cnn(self.rng)))

    def test_normalizations(self):
        layers = reference_cnn(self.rng)
        self.assert_matches_reference(
            artifact(layers, {"type": "global", "mean": [0.55], "std": 0.12})
        )
        self.assert_matches_reference(artifact(layers, {"type": "per_image", "eps": 1e-5}))

    def test_stride_no_padding_and_flatten_dense(self):
        layers = [
            conv(1, 3, kernel=5, stride=2, padding=0, rng=self.rng),  # 18 x 10
            {"type": "relu"},
            {"type": "maxpool2d", "kernel": 3, "stride": 2},  # 8 x 4
            dense(3 * 8 * 4, 4, self.rng),
            {"type": "softmax"},
        ]
        self.assert_matches_reference(artifact(layers))
        layers.insert(3, {"type": "flatten"})
        self.assert_matches_reference(artifact(layers))

    def test_model_metadata(self):
        model = artifact(reference_cnn(self.rng))
        self.assertIsNone(build_classifier(model).sha256)
        model["sha256"] = canonical_sha256(model)
        classifier = build_classifier(model)
        self.assertEqual(classifier.architecture, "cnn")
        self.assertEqual(classifier.sha256, model["sha256"])
        self.assertEqual(classifier.version, "2.0.0-test")

    def test_shape_validation(self):
        base = artifact(reference_cnn(self.rng))
        mutations = [
            lambda m: m["layers"][0]["weight"].pop(),
            lambda m: m["layers"][0]["bias"].append(0.0),
            lambda m: m["layers"][3].update(in_channels=5),
            lambda m: m["layers"][9].update(in_features=7),
            lambda m: m["layers"][9].update(out_features=3),
            lambda m: m["layers"].pop(),
            lambda m: m["layers"].insert(2, {"type": "softmax"}),
            lambda m: m["layers"].insert(0, {"type": "dropout"}),
            lambda m: m["layers"][0]["weight"].__setitem__(0, float("nan")),
            lambda m: m["layers"][0]["weight"].__setitem__(0, True),
            lambda m: m["layers"][0].update(kernel=50),
            lambda m: m["layers"].insert(9, {"type": "maxpool2d", "kernel": 2}),
            lambda m: m["input"].update(normalization={"type": "global", "mean": 0.5, "std": 0}),
            lambda m: m["input"].update(normalization={"type": "zscore"}),
            lambda m: m["input"].update(width=32),
            lambda m: m.update(sha256="xyz"),
            lambda m: m.update(sha256="ab" * 32),
            lambda m: m.update(classes=CLASSES[:3]),
        ]
        for index, mutate in enumerate(mutations):
            model = copy.deepcopy(base)
            mutate(model)
            with self.subTest(index=index), self.assertRaises(ValueError):
                build_classifier(model)

    def test_typed_dense_artifact(self):
        model = artifact([dense(H * W, 4, self.rng), {"type": "softmax"}])
        model["architecture"] = "dense"
        self.assertEqual(build_classifier(model).architecture, "dense")
        self.assert_matches_reference(model)

    def test_unknown_architecture(self):
        model = artifact(reference_cnn(self.rng))
        model["architecture"] = "transformer"
        with self.assertRaises(ValueError):
            build_classifier(model)


if __name__ == "__main__":
    unittest.main()
