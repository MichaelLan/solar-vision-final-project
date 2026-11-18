from fastapi import HTTPException, Request
from fastapi.responses import Response
from workers import asgi

from api.app import app


@app.get("/{path:path}")
async def static_assets(path: str, request: Request):
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
    response = await request.scope["env"].ASSETS.fetch(f"https://assets.local/{path}")
    return Response(
        content=await response.bytes(), status_code=response.status, headers=response.headers
    )


Default = asgi.entrypoint(app)
