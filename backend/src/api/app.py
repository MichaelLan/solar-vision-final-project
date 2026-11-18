from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from core.container import get_classifier, get_service
from infra.classifier import LABELS, ModelUnavailable
from services.classification import (
    ClassificationService,
    InvalidImageTensor,
    validate_dimension_type,
    validate_pixel_types,
)


class PredictRequest(BaseModel):
    pixels: list[float] = Field(min_length=960, max_length=960)
    width: int
    height: int

    @field_validator("pixels", mode="before")
    @classmethod
    def strict_pixels(cls, value: object) -> object:
        return validate_pixel_types(value)

    @field_validator("width", "height", mode="before")
    @classmethod
    def strict_dimensions(cls, value: object) -> object:
        return validate_dimension_type(value)


app = FastAPI(title="Solar Vision API")


class BodyLimitMiddleware:
    def __init__(self, inner_app, max_bytes: int = 65536):
        self.inner_app = inner_app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.inner_app(scope, receive, send)
        chunks = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_bytes:
                response = JSONResponse(status_code=413, content={"detail": "Request body too large"})
                return await response(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        return await self.inner_app(scope, replay, send)


app.add_middleware(BodyLimitMiddleware)


@app.exception_handler(ModelUnavailable)
async def model_unavailable(_request: Request, exc: ModelUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/api/health")
async def health():
    try:
        get_classifier()
        return {"status": "ok", "model_ready": True}
    except (ModelUnavailable, ValueError):
        return {"status": "degraded", "model_ready": False}


@app.get("/api/model/info")
async def model_info():
    try:
        classifier = get_classifier()
    except (ModelUnavailable, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "version": classifier.version,
        "classes": [{"class": name, "label": LABELS[name]} for name in classifier.classes],
        "input": {"width": 24, "height": 40, "channels": 1, "normalization": "rounded_bt601_luma_255"},
    }


@app.post("/api/predict")
async def predict(request: PredictRequest, service: Annotated[ClassificationService, Depends(get_service)]):
    try:
        result = service.classify(request.pixels, request.width, request.height)
    except InvalidImageTensor as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "prediction": {
            "class": result.class_name,
            "label": LABELS[result.class_name],
            "confidence": result.confidence,
        },
        "probabilities": result.probabilities,
        "model_version": service.classifier.version,
    }
