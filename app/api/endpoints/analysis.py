import asyncio
import io
import logging

import magic
from app.core.config import settings
from app.core.limiter import limiter
from app.core.timer import timed
from app.models.schemas import (
    DocumentAnalysis,
    DocumentTypeScore,
    FaceRecognitionResult,
    ImageRecognitionResult,
    OCRResult,
    OCRWordResult,
    TimingBreakdown,
)
from app.services.ai_models import ai_models
from app.services.extractor import extractor
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


@router.post("/document-analysis", response_model=DocumentAnalysis)
@limiter.limit("20/minute")
async def analyze_document(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()

    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    detected_mime = magic.from_buffer(image_bytes[:2048], mime=True)
    if detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        with timed() as total_timer:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.load()

            # ── Run OCR, CLIP, and face detection concurrently ──────────────
            # All three are CPU/IO bound and release the GIL during C-extension
            # calls, giving real parallelism even without CUDA.
            loop = asyncio.get_event_loop()

            def _run_ocr():
                with timed() as t:
                    result = ai_models.run_ocr(image_bytes)
                return result, t.elapsed_ms

            def _run_classify():
                with timed() as t:
                    result = ai_models.classify_document_detailed(image)
                return result, t.elapsed_ms

            def _run_face():
                with timed() as t:
                    result = ai_models.detect_face(image)
                return result, t.elapsed_ms

            logger.info("Launching OCR + CLIP + face detection concurrently…")
            (
                (ocr_raw, ocr_ms),
                (clip_raw, clip_ms),
                (face_raw, face_ms),
            ) = await asyncio.gather(
                loop.run_in_executor(None, _run_ocr),
                loop.run_in_executor(None, _run_classify),
                loop.run_in_executor(None, _run_face),
            )

            full_text, avg_ocr_conf, easy_results = ocr_raw
            doc_type, recognition_conf, all_scores = clip_raw
            face_result = face_raw

            logger.info(
                "Concurrent tasks done — OCR: %.1f ms | CLIP: %.1f ms | Face: %.1f ms",
                ocr_ms,
                clip_ms,
                face_ms,
            )

            # ── Build OCR result ─────────────────────────────────────────────
            words = [
                OCRWordResult(text=txt, confidence=round(conf * 100, 2))
                for (_bbox, txt, conf) in easy_results
            ]
            ocr_result = OCRResult(
                text=full_text,
                confidence=round(avg_ocr_conf, 2),
                bounding_boxes=[],
                language="en",
                word_count=len(words),
                words=words,
                time_ms=round(ocr_ms, 2),
                engine="easyocr",
            )

            # ── Build face result ─────────────────────────────────────────────
            if face_result.get("face_detected"):
                face_b64 = None
                try:
                    face_b64 = ai_models.get_face_base64(
                        image, face_result.get("bounding_box")
                    )
                except Exception:
                    logger.warning("Failed to extract face crop")
                face_info = FaceRecognitionResult(
                    face_detected=True,
                    confidence=face_result.get("confidence", 0.0),
                    bounding_box=face_result.get("bounding_box"),
                    face_image=face_b64,
                    time_ms=round(face_ms, 2),
                )
            else:
                face_info = FaceRecognitionResult(
                    face_detected=False,
                    confidence=0.0,
                    time_ms=round(face_ms, 2),
                )

            # ── Build recognition result ──────────────────────────────────────
            label_scores = [
                DocumentTypeScore(document_type=label, confidence=round(score, 4))
                for label, score in all_scores
            ]
            recognition_result = ImageRecognitionResult(
                document_type=doc_type,
                confidence=round(recognition_conf, 4),
                description=f"This document appears to be a {doc_type}.",
                key_features=[w.text for w in words[:3]] if words else [],
                face_info=face_info,
                all_scores=label_scores,
                clip_time_ms=round(clip_ms, 2),
                face_time_ms=round(face_ms, 2),
            )

            # ── Field extraction (LLM — sequential, depends on OCR text) ────
            logger.info("Extracting structured fields via LLM…")
            with timed() as extract_timer:
                extraction_data = await extractor.extract_fields(full_text, doc_type)

            logger.info("Extraction done in %.1f ms", extract_timer.elapsed_ms)

            # ── Summary & keywords ───────────────────────────────────────────
            summary = (
                f"Detected {doc_type} with {recognition_conf:.1%} confidence. "
                f"OCR extracted {len(words)} segments "
                f"(avg confidence {avg_ocr_conf:.1f}%)."
            )
            if face_info.face_detected:
                summary += f" Face detected (confidence {face_info.confidence:.0%})."

            words_text = full_text.lower().split()
            keywords = list(dict.fromkeys(w for w in words_text if len(w) > 4))[:10]

        timing = TimingBreakdown(
            ocr_ms=round(ocr_ms, 2),
            classification_ms=round(clip_ms, 2),
            face_detection_ms=round(face_ms, 2),
            extraction_ms=round(extract_timer.elapsed_ms, 2),
            total_ms=round(total_timer.elapsed_ms, 2),
        )

        logger.info(
            "Analysis complete in %.1f ms total "
            "(OCR %.1f ms | CLIP %.1f ms | face %.1f ms | extract %.1f ms)",
            total_timer.elapsed_ms,
            ocr_ms,
            clip_ms,
            face_ms,
            extract_timer.elapsed_ms,
        )

        return DocumentAnalysis(
            ocr_result=ocr_result,
            recognition_result=recognition_result,
            extraction=extraction_data,
            summary=summary,
            keywords=keywords,
            timing=timing,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Analysis error")
        raise HTTPException(status_code=500, detail="Analysis processing failed")
