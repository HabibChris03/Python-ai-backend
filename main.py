from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
from app.api.api import api_router
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def get_application() -> FastAPI:
    application = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware for request logging
    @application.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info(f"Incoming request: {request.method} {request.url.path}")
        response = await call_next(request)
        if response.status_code == 404:
            logger.warning(f"404 Not Found: {request.method} {request.url.path}")
        return response

    # Include API routes
    application.include_router(api_router, prefix="/api")

    # Catch-all route for better 404 messaging
    @application.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def catch_all(request: Request, path_name: str):
        # Ignore common browser requests
        if path_name == "favicon.ico":
            return JSONResponse(status_code=404, content={"detail": "Not found"})
            
        return JSONResponse(
            status_code=404,
            content={
                "message": f"Endpoint /{path_name} not found on this server.",
                "method": request.method,
                "path": path_name,
                "suggestions": ["Check if you missed the /api prefix", "Check for typos in the URL"]
            }
        )

    return application

app = get_application()

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to DocFinder AI Backend",
        "docs": "/api/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )