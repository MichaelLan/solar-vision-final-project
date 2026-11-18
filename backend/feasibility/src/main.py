import json
import math
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI
from pydantic import BaseModel, Field
from workers import asgi


class Input(BaseModel):
    pixels: list[float] = Field(min_length=960, max_length=960)


app = FastAPI()
Default = asgi.entrypoint(app)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
def predict(data: Input):
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in data.pixels):
        return {"error": "Invalid pixel range"}
    weights = json.loads(Path(__file__).with_name("weights.json").read_text())
    start = perf_counter()
    output = []
    for row, bias in zip(weights["weights"], weights["biases"], strict=True):
        output.append(sum(w * x for w, x in zip(row, data.pixels, strict=True)) + bias)
    shift = max(output)
    unscaled = [math.exp(x - shift) for x in output]
    total = sum(unscaled)
    return {"probabilities": [x / total for x in unscaled], "inference_ms": (perf_counter() - start) * 1000}
