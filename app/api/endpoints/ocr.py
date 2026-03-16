from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
import logging
from app.models.schemas import OCRResult
from app.services.ai_models import ai_models
from app.utils.image_processing import extract_text_with_tesseract

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ocr", response_model=OCRResult)
async def extract_text(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # EasyOCR
        easy_results = ai_models.easy_reader.readtext(image_bytes)
        text_parts = []
        total_confidence = 0
        
        for (bbox, text, confidence) in easy_results:
            text_parts.append(text)
            total_confidence += confidence
            
        avg_conf = (total_confidence / len(easy_results)) * 100 if easy_results else 0
        
        # Fallback/Merge with Tesseract if needed
        if not text_parts:
            tess_result = extract_text_with_tesseract(image)
            return OCRResult(**tess_result)
            
        return OCRResult(
            text=" ".join(text_parts),
            confidence=avg_conf,
            bounding_boxes=[], # Could map bbox here
            language="en"
        )
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
