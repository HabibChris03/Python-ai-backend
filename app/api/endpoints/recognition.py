import io
import logging
import uuid
from typing import List, Optional

import magic
from app.core.config import settings
from app.core.limiter import limiter
from app.core.timer import timed
from app.models.schemas import (
    DocumentTypeScore,
    FaceRecognitionResult,
    ImageRecognitionResult,
)
from app.services.ai_models import ai_models
from app.services.document_scanner import document_scanner
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
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


@router.post("/image-recognition", response_model=ImageRecognitionResult)
@limiter.limit("30/minute")
async def recognize_image(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()

    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    detected_mime = magic.from_buffer(image_bytes[:2048], mime=True)
    if detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        logger.info("Starting image recognition…")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()

        # 1. CLIP classification (uses cached text embeddings)
        with timed() as clip_timer:
            doc_type, confidence, all_scores = ai_models.classify_document_detailed(
                image
            )

        logger.info(
            "CLIP classification done in %.1f ms: %s (%.2f%%)",
            clip_timer.elapsed_ms,
            doc_type,
            confidence * 100,
        )

        # 2. Face detection (resizes internally for speed)
        with timed() as face_timer:
            face_result = ai_models.detect_face(image)

        logger.info(
            "Face detection done in %.1f ms: detected=%s confidence=%.2f",
            face_timer.elapsed_ms,
            face_result["face_detected"],
            face_result["confidence"],
        )

        face_schema: Optional[FaceRecognitionResult] = None
        if face_result["face_detected"]:
            face_b64 = None
            try:
                face_b64 = ai_models.get_face_base64(image, face_result["bounding_box"])
            except Exception:
                logger.warning("Failed to extract face crop")
            face_schema = FaceRecognitionResult(
                face_detected=True,
                confidence=face_result["confidence"],
                bounding_box=face_result["bounding_box"],
                face_image=face_b64,
                time_ms=round(face_timer.elapsed_ms, 2),
            )
        else:
            face_schema = FaceRecognitionResult(
                face_detected=False,
                confidence=0.0,
                time_ms=round(face_timer.elapsed_ms, 2),
            )

        # Build ranked label scores list
        label_scores = [
            DocumentTypeScore(document_type=label, confidence=round(score, 4))
            for label, score in all_scores
        ]

        rf_used = ai_models.rf_model is not None
        classification_method = (
            "Random Forest classification (CLIP features)"
            if rf_used
            else "Visual pattern recognition (CLIP cosine similarity)"
        )

        return ImageRecognitionResult(
            document_type=doc_type,
            confidence=round(confidence, 4),
            description=f"Predicted type: {doc_type} ({confidence:.2%})",
            key_features=[
                classification_method,
                "Face detection"
                if face_result["face_detected"]
                else "No face detected",
            ],
            face_info=face_schema,
            all_scores=label_scores,
            clip_time_ms=round(clip_timer.elapsed_ms, 2),
            face_time_ms=round(face_timer.elapsed_ms, 2),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Recognition error")
        raise HTTPException(status_code=500, detail="Recognition processing failed")


async def _process_scan(
    file: UploadFile,
    use_google: bool = False,
) -> tuple[Image.Image, bool, str]:
    image_bytes = await file.read()

    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    detected_mime = magic.from_buffer(image_bytes[:2048], mime=True)
    if detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.load()
    scanned_image, success, method = document_scanner.scan(image, use_google=use_google)
    return scanned_image, success, method


@router.post("/detect-document")
@limiter.limit("30/minute")
async def detect_document(
    request: Request,
    file: UploadFile = File(...),
    use_google: bool = Query(
        False, description="Use Google Cloud Vision when configured"
    ),
):
    try:
        with timed() as scan_timer:
            logger.info("Starting document detection (use_google=%s)…", use_google)
            scanned_image, success, method = await _process_scan(
                file, use_google=use_google
            )
            if not success:
                logger.info("No document edges found, returning enhanced scan")

        img_io = io.BytesIO()
        scanned_image.save(img_io, "PNG")
        img_io.seek(0)

        logger.info(
            "Document detection complete in %.1f ms using method: %s",
            scan_timer.elapsed_ms,
            method,
        )
        return StreamingResponse(
            img_io,
            media_type="image/png",
            headers={
                "X-Document-Detected": str(success).lower(),
                "X-Scan-Filter": "grayscale",
                "X-Scan-Method": method,
                "X-Scan-Time-Ms": str(round(scan_timer.elapsed_ms, 2)),
            },
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Auto-crop error")
        raise HTTPException(status_code=500, detail="Document detection failed")


# scan-document is an alias kept for backward compatibility
@router.post("/scan-document")
@limiter.limit("30/minute")
async def scan_document(
    request: Request,
    file: UploadFile = File(...),
    use_google: bool = Query(
        False, description="Use Google Cloud Vision when configured"
    ),
):
    return await detect_document(request, file=file, use_google=use_google)


async def _load_upload_image(upload: UploadFile) -> Optional[Image.Image]:
    """Load an uploaded file as PIL image, validating size and mime type."""
    image_bytes = await upload.read()
    if not image_bytes:
        return None

    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    detected_mime = magic.from_buffer(image_bytes[:2048], mime=True)
    if detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
        return image
    except Exception as exc:
        logger.warning("Skipping upload (%s): %s", upload.content_type, exc)
        return None


@router.post("/export-pdf")
@limiter.limit("30/minute")
async def export_pdf(
    request: Request,
    files: List[UploadFile] = File(...),
    use_google: bool = Query(
        False, description="Scan each page with Google Vision first"
    ),
    rescan: bool = Query(
        True, description="Re-frame and B&W enhance images before PDF export"
    ),
):
    """Convert one or more document photos into a downloadable black-and-white PDF."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")

    try:
        with timed() as pdf_timer:
            logger.info(
                "PDF export started: %d files, use_google=%s, rescan=%s",
                len(files),
                use_google,
                rescan,
            )
            scanned_pages: List[Image.Image] = []
            for i, upload in enumerate(files):
                logger.info("Processing page %d/%d", i + 1, len(files))
                image = await _load_upload_image(upload)
                if image is None:
                    logger.warning("Skipped invalid image at index %d", i)
                    continue

                if rescan:
                    scanned, _, _ = document_scanner.scan(image, use_google=use_google)
                    scanned_pages.append(scanned)
                else:
                    enhanced, _ = document_scanner.enhance_grayscale(image)
                    scanned_pages.append(enhanced)

            if not scanned_pages:
                raise HTTPException(
                    status_code=400,
                    detail="No valid images provided. Ensure uploads are JPG/PNG image files.",
                )

            pdf_bytes = document_scanner.images_to_pdf(scanned_pages)

        filename = f"scan_{uuid.uuid4().hex}.pdf"
        logger.info(
            "PDF created: %d pages in %.1f ms", len(scanned_pages), pdf_timer.elapsed_ms
        )
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Scan-Filter": "grayscale",
                "X-Page-Count": str(len(scanned_pages)),
                "X-Processing-Time-Ms": str(round(pdf_timer.elapsed_ms, 2)),
            },
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("PDF export error")
        raise HTTPException(status_code=500, detail="PDF export failed")


from pydantic import BaseModel
import httpx

class CompareFacesRequest(BaseModel):
    img1_url: str
    img2_url: str

async def _download_pil_image(url: str) -> Image.Image:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image from {url}")
            return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")

@router.post("/compare-faces")
@limiter.limit("30/minute")
async def compare_faces(request: Request, payload: CompareFacesRequest):
    try:
        # 1. Download both images
        img1 = await _download_pil_image(payload.img1_url)
        img2 = await _download_pil_image(payload.img2_url)

        # 2. Run face detection on both
        face_res1 = ai_models.detect_face(img1)
        face_res2 = ai_models.detect_face(img2)

        face1_detected = face_res1["face_detected"]
        face2_detected = face_res2["face_detected"]

        if not face1_detected or not face2_detected:
            return {
                "success": False,
                "message": "Face not detected in one or both images",
                "face_detected_img1": face1_detected,
                "face_detected_img2": face2_detected,
                "similarity": 0.0
            }

        # 3. Crop faces from both images
        x1, y1, w1, h1 = face_res1["bounding_box"]
        margin = 0.2
        img_w1, img_h1 = img1.size
        crop1 = img1.crop((
            max(0, x1 - int(w1 * margin)),
            max(0, y1 - int(h1 * margin)),
            min(img_w1, x1 + w1 + int(w1 * margin)),
            min(img_h1, y1 + h1 + int(h1 * margin))
        ))

        x2, y2, w2, h2 = face_res2["bounding_box"]
        img_w2, img_h2 = img2.size
        crop2 = img2.crop((
            max(0, x2 - int(w2 * margin)),
            max(0, y2 - int(h2 * margin)),
            min(img_w2, x2 + w2 + int(w2 * margin)),
            min(img_h2, y2 + h2 + int(h2 * margin))
        ))

        # 4. Get CLIP embeddings & calculate cosine similarity
        feat1 = ai_models.get_image_embedding(crop1)
        feat2 = ai_models.get_image_embedding(crop2)
        
        # Calculate cosine similarity using torch
        import torch
        similarity = torch.nn.functional.cosine_similarity(feat1, feat2).item()

        logger.info("Face comparison complete: similarity = %.2f", similarity)

        return {
            "success": True,
            "face_detected_img1": True,
            "face_detected_img2": True,
            "similarity": round(similarity, 4)
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Face comparison error")
        raise HTTPException(status_code=500, detail=f"Face comparison failed: {str(exc)}")
