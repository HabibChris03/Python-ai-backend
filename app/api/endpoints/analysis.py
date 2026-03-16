from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
import logging
import torch
from app.models.schemas import DocumentAnalysis, OCRResult, ImageRecognitionResult
from app.services.ai_models import ai_models

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/document-analysis", response_model=DocumentAnalysis)
async def analyze_document(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file image_bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 1. OCR Step
        logger.info("Performing OCR...")
        easy_results = ai_models.easy_reader.readtext(image_bytes)
        text_parts = [res[1] for res in easy_results]
        full_text = " ".join(text_parts)
        avg_ocr_conf = sum(res[2] for res in easy_results) / len(easy_results) if easy_results else 0
        
        # 2. Image Recognition (CLIP)
        logger.info("Performing Image Recognition...")
        inputs = ai_models.clip_processor(
            text=ai_models.document_types, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(ai_models.device)
        
        with torch.no_grad():
            outputs = ai_models.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
        best_match_idx = probs.argmax().item()
        doc_type = ai_models.document_types[best_match_idx]
        recognition_conf = probs[0][best_match_idx].item()
        
        # 3. Formulate Summary
        summary = f"Detected {doc_type} with {recognition_conf:.1%} confidence."
        if full_text:
            summary += f" Extracted {len(text_parts)} text segments."
            
        # 4. Keywords
        words = full_text.lower().split()
        # Basic keyword extraction (filter out short words)
        keywords = list(set([w for w in words if len(w) > 4]))[:10]
        
        return DocumentAnalysis(
            ocr_result=OCRResult(
                text=full_text, 
                confidence=avg_ocr_conf * 100, 
                bounding_boxes=[], 
                language="en"
            ),
            recognition_result=ImageRecognitionResult(
                document_type=doc_type, 
                confidence=recognition_conf, 
                description=f"This document appears to be a {doc_type}.",
                key_features=keywords[:3]
            ),
            summary=summary,
            keywords=keywords
        )
        
    except Exception as e:
        logger.error(f"Analysis Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
