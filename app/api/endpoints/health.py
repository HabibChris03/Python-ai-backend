from fastapi import APIRouter
from app.models.schemas import HealthCheckResponse
from app.services.ai_models import ai_models

router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        models_loaded=True,
        device=ai_models.device,
        openai_configured=ai_models.llm is not None
    )
