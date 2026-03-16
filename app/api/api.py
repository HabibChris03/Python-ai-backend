from fastapi import APIRouter
from app.api.endpoints import ocr, recognition, chat, health, analysis

api_router = APIRouter()

api_router.include_router(ocr.router, tags=["OCR"])
api_router.include_router(recognition.router, tags=["Recognition"])
api_router.include_router(chat.router, tags=["Chat"])
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(analysis.router, tags=["Analysis"])
