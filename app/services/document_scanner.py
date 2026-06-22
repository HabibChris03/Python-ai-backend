"""
Fast CamScanner-style document scanner (OpenCV).
Optimized for mobile uploads: resize before processing, minimal copies.
"""

import io
import logging
import os
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

MAX_SCAN_EDGE = int(os.getenv("SCAN_MAX_EDGE", "1600"))


class DocumentScannerService:
    def normalize_pil_image(self, pil_image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(pil_image)
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.split()[-1])
            return background
        return image.convert("RGB")

    def _resize_for_processing(
        self, pil_image: Image.Image
    ) -> Tuple[Image.Image, float]:
        w, h = pil_image.size
        longest = max(w, h)
        if longest <= MAX_SCAN_EDGE:
            return pil_image, 1.0
        scale = MAX_SCAN_EDGE / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return pil_image.resize(new_size, Image.Resampling.LANCZOS), scale

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def warp_quad(self, pil_image: Image.Image, pts: np.ndarray) -> Image.Image:
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        warped = self._four_point_transform(image, pts)
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self.order_points(pts.astype("float32"))
        tl, tr, br, bl = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(1, int(max(width_a, width_b)))
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(1, int(max(height_a, height_b)))
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height))

    def detect_quad_opencv(self, pil_image: Image.Image) -> Tuple[Image.Image, bool]:
        pil_image = self.normalize_pil_image(pil_image)
        resized, scale = self._resize_for_processing(pil_image)
        image = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)
        image_area = image.shape[0] * image.shape[1]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged = cv2.dilate(edged, kernel, iterations=1)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

        screen_cnt = None
        for contour in contours:
            if cv2.contourArea(contour) < image_area * 0.12:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                screen_cnt = approx
                break

        if screen_cnt is None:
            return pil_image, False

        pts = screen_cnt.reshape(4, 2).astype("float32")
        if scale != 1.0:
            pts = pts / scale

        warped = self._four_point_transform(
            cv2.cvtColor(
                np.array(self.normalize_pil_image(pil_image)), cv2.COLOR_RGB2BGR
            ),
            pts,
        )
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)), True

    def _preprocess_to_gray(self, pil_image: Image.Image) -> np.ndarray:
        """Shared preprocessing: normalize -> RGB->gray -> denoise -> CLAHE."""
        pil_image = self.normalize_pil_image(pil_image)
        gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 15)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def enhance_grayscale(self, pil_image: Image.Image) -> Tuple[Image.Image, bool]:
        """Convert image to grayscale while preserving grey tones (not pure black/white)."""
        gray = self._preprocess_to_gray(pil_image)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(gray_rgb), True

    def enhance_black_white(self, pil_image: Image.Image) -> Tuple[Image.Image, bool]:
        gray = self._preprocess_to_gray(pil_image)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        bw = cv2.medianBlur(bw, 3)

        black_ratio = 1.0 - (np.count_nonzero(bw) / bw.size)
        if black_ratio < 0.01 or black_ratio > 0.70:
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return Image.fromarray(bw).convert("RGB"), True

    def scan(
        self,
        pil_image: Image.Image,
        use_google: bool = False,
    ) -> Tuple[Image.Image, bool, str]:
        """
        Returns (scanned_image, document_detected, method_used).
        method_used: opencv | google | fallback
        """
        if use_google:
            try:
                from app.services.google_scanner import (
                    is_google_vision_enabled,
                    scan_with_google,
                )

                if is_google_vision_enabled():
                    scanned, ok = scan_with_google(pil_image)
                    if ok and scanned is not None:
                        return scanned, True, "google"
            except Exception as exc:
                logger.warning("Google scan fallback to OpenCV: %s", exc)

        cropped, detected = self.detect_quad_opencv(pil_image)
        enhanced, _ = self.enhance_grayscale(cropped)
        return enhanced, detected, "opencv" if detected else "fallback"

    def images_to_pdf(self, images: List[Image.Image]) -> bytes:
        if not images:
            raise ValueError("No images to convert")

        normalized = [
            self.normalize_pil_image(image).convert("RGB") for image in images
        ]

        # 1) img2pdf — best quality when installed
        try:
            import img2pdf

            buffers = []
            for rgb in normalized:
                buf = io.BytesIO()
                rgb.save(buf, format="JPEG", quality=95, optimize=True)
                buffers.append(buf.getvalue())
            return img2pdf.convert(buffers)
        except ImportError:
            logger.info("img2pdf not installed, trying next PDF backend")
        except Exception as exc:
            logger.warning("img2pdf failed, trying next PDF backend: %s", exc)

        # 2) ReportLab fallback
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.utils import ImageReader
            from reportlab.pdfgen import canvas

            pdf_buffer = io.BytesIO()
            pdf = canvas.Canvas(pdf_buffer, pagesize=A4)
            page_w, page_h = A4

            for rgb in normalized:
                img_buf = io.BytesIO()
                rgb.save(img_buf, format="JPEG", quality=95)
                img_buf.seek(0)
                reader = ImageReader(img_buf)
                iw, ih = rgb.size
                scale = min(page_w / iw, page_h / ih) * 0.95
                draw_w, draw_h = iw * scale, ih * scale
                x = (page_w - draw_w) / 2
                y = (page_h - draw_h) / 2
                pdf.drawImage(
                    reader, x, y, draw_w, draw_h, preserveAspectRatio=True, mask="auto"
                )
                pdf.showPage()

            pdf.save()
            return pdf_buffer.getvalue()
        except ImportError:
            logger.info("reportlab not installed, using Pillow PDF fallback")
        except Exception as exc:
            logger.warning("reportlab failed, using Pillow PDF fallback: %s", exc)

        # 3) Pillow — always available in this project
        pdf_buffer = io.BytesIO()
        first, *rest = normalized
        save_kwargs: dict = {"format": "PDF", "save_all": True, "resolution": 150.0}
        if rest:
            save_kwargs["append_images"] = rest
        first.save(pdf_buffer, **save_kwargs)
        return pdf_buffer.getvalue()


document_scanner = DocumentScannerService()
