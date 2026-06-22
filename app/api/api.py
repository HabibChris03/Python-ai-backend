from fastapi import APIRouter, Depends, Header, HTTPException

from app.api.endpoints import analysis, chat, health, ocr, recognition
from app.core.config import settings
from app.core.limiter import limiter  # re-exported so main.py can register it


async def verify_api_key(x_api_key: str = Header(default="")):
    """Reject requests that do not carry the internal service API key.
    Auth is disabled when API_KEY is not configured (development mode)."""
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


api_router = APIRouter()

# Health check is intentionally public (no auth, no rate limit)
api_router.include_router(health.router, tags=["Health"])

# All data endpoints require a valid API key
api_router.include_router(
    ocr.router,
    tags=["OCR"],
    dependencies=[Depends(verify_api_key)],
)
api_router.include_router(
    recognition.router,
    prefix="/recognition",
    tags=["Recognition"],
    dependencies=[Depends(verify_api_key)],
)
api_router.include_router(
    chat.router,
    tags=["Chat"],
    dependencies=[Depends(verify_api_key)],
)
api_router.include_router(
    analysis.router,
    tags=["Analysis"],
    dependencies=[Depends(verify_api_key)],
)
