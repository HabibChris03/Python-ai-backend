from pydantic import BaseModel
from typing import List, Optional

class OCRResult(BaseModel):
    text: str
    confidence: float
    bounding_boxes: List[dict]
    language: str

class ImageRecognitionResult(BaseModel):
    document_type: str
    confidence: float
    description: str
    key_features: List[str]

class ChatbotResponse(BaseModel):
    response: str
    intent: str
    confidence: float

class DocumentAnalysis(BaseModel):
    ocr_result: OCRResult
    recognition_result: ImageRecognitionResult
    summary: str
    keywords: List[str]

class HealthCheckResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str
    openai_configured: bool
