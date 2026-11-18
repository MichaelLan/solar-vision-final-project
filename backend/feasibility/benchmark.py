import json
import math
import statistics
import time
import tracemalloc
from pathlib import Path

weights_path = Path(__file__).parent / "src" / "weights.json"
weights = json.loads(weights_path.read_text())
pixels = [0.5] * 960


def infer():
    logits = [
        sum(weight * value for weight, value in zip(row, pixels, strict=True)) + bias
        for row, bias in zip(weights["weights"], weights["biases"], strict=True)
    ]
    shifted = [math.exp(value - max(logits)) for value in logits]
    return [value / sum(shifted) for value in shifted]


tracemalloc.start()
samples = []
for _ in range(100):
    start = time.perf_counter()
    infer()
    samples.append((time.perf_counter() - start) * 1000)
_, peak = tracemalloc.get_traced_memory()
print(
    json.dumps(
        {
            "artifact_bytes": weights_path.stat().st_size,
            "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[94],
            "python_peak_allocated_bytes": peak,
            "scope": "CPython only; not Cloudflare Workers",
        }
    )
)
