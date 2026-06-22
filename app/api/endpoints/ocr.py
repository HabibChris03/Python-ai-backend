import io
import logging

import magic
from app.core.config import settings
from app.core.limiter import limiter
from app.core.timer import timed
from app.models.schemas import OCRResult, OCRWordResult
from app.services.ai_models import ai_models
from app.utils.image_processing import extract_text_with_tesseract
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
    "image/bmp",
}


@router.post("/ocr", response_model=OCRResult)
@limiter.limit("30/minute")
async def extract_text(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()

    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    detected_mime = magic.from_buffer(image_bytes[:2048], mime=True)
    if detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        logger.info("OCR extraction started, size=%d bytes", len(image_bytes))
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()

        with timed() as ocr_timer:
            easy_results = ai_models.easy_reader.readtext(image_bytes)

        if not easy_results:
            logger.warning("No text found with EasyOCR, trying Tesseract fallback")
            with timed() as tess_timer:
                tess_result = extract_text_with_tesseract(image)
            tess_result["time_ms"] = tess_timer.elapsed_ms
            tess_result["engine"] = "tesseract"
            tess_result["word_count"] = len(tess_result.get("text", "").split())
            tess_result["words"] = []
            return OCRResult(**tess_result)

        # Build per-word results
        words = [
            OCRWordResult(text=text, confidence=round(conf * 100, 2))
            for (_bbox, text, conf) in easy_results
        ]
        avg_conf = sum(w.confidence for w in words) / len(words) if words else 0.0

        logger.info(
            "OCR done in %.1f ms: %d segments, avg confidence %.1f%%",
            ocr_timer.elapsed_ms,
            len(words),
            avg_conf,
        )

        return OCRResult(
            text=" ".join(w.text for w in words),
            confidence=round(avg_conf, 2),
            bounding_boxes=[],
            language="en",
            word_count=len(words),
            words=words,
            time_ms=round(ocr_timer.elapsed_ms, 2),
            engine="easyocr",
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("OCR error")
        raise HTTPException(status_code=500, detail="OCR processing failed")
