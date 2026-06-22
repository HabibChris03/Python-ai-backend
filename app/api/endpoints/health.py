import logging

from app.models.schemas import HealthCheckResponse
from app.services.ai_models import ai_models
from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(status="healthy")


@router.get("/health/detailed")
async def health_check_detailed():
    """
    Detailed health check showing which models are loaded and whether
    the CLIP text-embedding cache is warm (fast classification = True).
    Requires API key authentication (handled at router level).
    """
    return {
        "status": "healthy",
        "models": {
            "easyocr_loaded": ai_models._easy_reader is not None,
            "clip_loaded": ai_models._clip_model is not None,
            "clip_text_cache_warm": ai_models._text_features_cache is not None,
            "face_cascade_loaded": not ai_models.face_cascade.empty(),
            "llm_configured": ai_models._llm is not None
            or bool(
                __import__(
                    "app.core.config", fromlist=["settings"]
                ).settings.GROQ_API_KEY
            ),
        },
        "device": ai_models.device,
        "document_type_count": len(ai_models.document_types),
    }
