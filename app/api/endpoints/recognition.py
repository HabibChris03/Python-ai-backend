from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
import torch
import logging
from app.models.schemas import ImageRecognitionResult
from app.services.ai_models import ai_models

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/image-recognition", response_model=ImageRecognitionResult)
async def recognize_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        inputs = ai_models.clip_processor(images=image, return_tensors="pt").to(ai_models.device)
        
        with torch.no_grad():
            image_features = ai_models.clip_model.get_image_features(**inputs)
            
            text_inputs = ai_models.clip_processor(
                text=ai_models.document_types, 
                return_tensors="pt", 
                padding=True
            ).to(ai_models.device)
            
            text_features = ai_models.clip_model.get_text_features(**text_inputs)
            
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarities = (image_features @ text_features.T).squeeze(0)
            
        top_idx = similarities.argmax().item()
        confidence = similarities[top_idx].item()
        doc_type = ai_models.document_types[top_idx]
        
        return ImageRecognitionResult(
            document_type=doc_type,
            confidence=confidence,
            description=f"Predicted type: {doc_type} ({confidence:.2%})",
            key_features=["Visual pattern recognition"]
        )
    except Exception as e:
        logger.error(f"Recognition Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
