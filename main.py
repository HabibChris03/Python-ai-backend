import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.api.api import api_router, limiter
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background cleanup task: delete face images older than 1 hour
# (relevant for any files created before the base64 migration)
# ---------------------------------------------------------------------------

def _purge_old_faces() -> None:
    faces_dir = "data/extracted_faces"
    if not os.path.exists(faces_dir):
        return
    now = time.time()
    removed = 0
    for fname in os.listdir(faces_dir):
        fpath = os.path.join(faces_dir, fname)
        try:
            if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > 3600:
                os.remove(fpath)
                removed += 1
        except Exception as exc:
            logger.warning("Could not delete face file %s: %s", fpath, exc)
    if removed:
        logger.info("Cleaned up %d old face image(s)", removed)


async def _face_cleanup_loop() -> None:
    """Run once per hour to delete stale face images from disk."""
    while True:
        await asyncio.sleep(3600)
        _purge_old_faces()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_face_cleanup_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def get_application() -> FastAPI:
    application = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        debug=False,
        lifespan=lifespan,
    )

    # Attach the rate-limiter state and error handler
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS — only allow explicitly configured origins
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security response headers on every response
    @application.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        return response

    # Reject oversized requests before the body is read
    @application.middleware("http")
    async def limit_request_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large"},
            )
        return await call_next(request)

    # Request logging
    @application.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info("Incoming request: %s %s", request.method, request.url.path)
        response = await call_next(request)
        if response.status_code == 404:
            logger.warning("404 Not Found: %s %s", request.method, request.url.path)
        return response

    # Include API routes
    application.include_router(api_router, prefix="/api")

    # Catch-all route for better 404 messaging
    @application.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def catch_all(request: Request, path_name: str):
        if path_name == "favicon.ico":
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        return JSONResponse(
            status_code=404,
            content={
                "message": f"Endpoint /{path_name} not found on this server.",
                "method": request.method,
                "path": path_name,
                "suggestions": [
                    "Check if you missed the /api prefix",
                    "Check for typos in the URL",
                ],
            },
        )

    return application


app = get_application()


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to DocFinder AI Backend",
        "docs": "/api/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
