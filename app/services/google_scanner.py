"""
Optional Google Cloud Vision integration for document boundary detection.
Set GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_VISION_ENABLED=true in .env
"""

import io
import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_client = None


def is_google_vision_enabled() -> bool:
    return os.getenv("GOOGLE_VISION_ENABLED", "false").lower() == "true" and bool(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not is_google_vision_enabled():
        return None
    try:
        from google.cloud import vision

        _client = vision.ImageAnnotatorClient()
        logger.info("Google Cloud Vision client initialized")
        return _client
    except Exception as exc:
        logger.warning("Google Vision unavailable: %s", exc)
        return None


def _pil_to_vision_bytes(pil_image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def detect_document_quad(pil_image: Image.Image) -> Optional[np.ndarray]:
    """
    Use Google Vision full-page bounds to estimate a document quadrilateral.
    Returns 4x2 float32 points or None.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        from google.cloud import vision

        image = vision.Image(content=_pil_to_vision_bytes(pil_image))
        response = client.document_text_detection(image=image)
        if response.error.message:
            logger.warning("Vision API error: %s", response.error.message)
            return None

        if not response.full_text_annotation or not response.full_text_annotation.pages:
            return None

        page = response.full_text_annotation.pages[0]
        if not page.width or not page.height:
            return None

        vertices = []
        if page.blocks:
            xs, ys = [], []
            for block in page.blocks:
                if not block.bounding_box or not block.bounding_box.vertices:
                    continue
                for v in block.bounding_box.vertices:
                    xs.append(v.x or 0)
                    ys.append(v.y or 0)
            if xs and ys:
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        if not vertices:
            w, h = pil_image.size
            pad_x = int(w * 0.04)
            pad_y = int(h * 0.04)
            vertices = [
                (pad_x, pad_y),
                (w - pad_x, pad_y),
                (w - pad_x, h - pad_y),
                (pad_x, h - pad_y),
            ]

        return np.array(vertices, dtype="float32")
    except Exception as exc:
        logger.warning("Google document detection failed: %s", exc)
        return None


def scan_with_google(pil_image: Image.Image) -> Tuple[Optional[Image.Image], bool]:
    """
    Attempt Google-assisted crop. Enhancement still done locally for speed.
    """
    # Use the existing singleton instead of re-instantiating
    from app.services.document_scanner import document_scanner

    quad = detect_document_quad(pil_image)
    if quad is None:
        return None, False

    warped = document_scanner.warp_quad(pil_image, quad)
    enhanced, _ = document_scanner.enhance_black_white(warped)
    return enhanced, True
