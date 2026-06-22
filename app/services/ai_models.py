"""
AI model service — lazy-loaded singletons for CLIP, EasyOCR, and face detection.

Performance optimisations applied:
  • CLIP text embeddings are cached after the first call (static label set).
  • Face detection resizes to ≤640 px before running the Haar cascade.
  • run_ocr returns per-segment confidence data alongside the aggregate.
"""

import base64
import io
import logging
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
from app.core.config import settings
from app.services.document_scanner import document_scanner
from PIL import Image

_CONFIDENCE_TEMPERATURE = 0.08

logger = logging.getLogger(__name__)

_FACE_DETECT_MAX_EDGE = 640  # resize before Haar for speed


class AIModelsService:
    """Heavy models are lazy-loaded so the server starts fast."""

    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available() and settings.DEVICE == "cuda" else "cpu"
        )
        logger.info("AI models service ready (lazy init on device: %s)", self.device)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self._easy_reader = None
        self._clip_model = None
        self._clip_processor = None
        self._sentence_model = None
        self._llm = None
        self._rf_model = None
        self._rf_classes = None


        # Cached CLIP text embeddings — computed once, reused on every call
        self._text_features_cache: "torch.Tensor | None" = None

        self.document_types: List[str] = [
            "cameroon national id card",
            "cameroon passport",
            "cameroon driving license",
            "cameroon birth certificate",
            "cameroon certificate",
            "cameroon marriage certificate",
            "cameroon first school leaving certificate education",
            "cameroon government common entrance certificate education",
            "cameroon hospital medical health record",
            "cameroon property title land certificate",
            "cameroon official document",
            "cameroon legal document",
            "credit card",
            "receipt",
            "invoice",
            "contract",
            "resume",
            "business card",
        ]

    # ── Lazy model properties ────────────────────────────────────────────────

    @property
    def easy_reader(self):
        if self._easy_reader is None:
            import easyocr

            logger.info("Loading EasyOCR reader (first request)…")
            self._easy_reader = easyocr.Reader(["en"], gpu=(self.device == "cuda"))
        return self._easy_reader

    @property
    def clip_model(self):
        if self._clip_model is None:
            from transformers import CLIPModel

            finetuned_path = "models/finetuned_clip_v2"
            if os.path.exists(finetuned_path) and os.path.exists(
                os.path.join(finetuned_path, "model.safetensors")
            ):
                logger.info("Loading fine-tuned CLIP model")
                self._clip_model = CLIPModel.from_pretrained(finetuned_path).to(
                    self.device
                )
            else:
                logger.info("Loading base CLIP model")
                self._clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                ).to(self.device)
        return self._clip_model

    @property
    def clip_processor(self):
        if self._clip_processor is None:
            from transformers import CLIPProcessor

            finetuned_path = "models/finetuned_clip_v2"
            if os.path.exists(finetuned_path):
                self._clip_processor = CLIPProcessor.from_pretrained(finetuned_path)
            else:
                self._clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
        return self._clip_processor

    @property
    def sentence_model(self):
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence transformer…")
            self._sentence_model = SentenceTransformer(
                "all-MiniLM-L6-v2", device=self.device
            )
        return self._sentence_model

    @property
    def llm(self):
        if self._llm is None and settings.GROQ_API_KEY:
            from langchain_groq import ChatGroq

            logger.info("Initializing Groq LLM")
            self._llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.6,
                api_key=settings.GROQ_API_KEY,
            )
        return self._llm

    @property
    def rf_model(self):
        if self._rf_model is None:
            finetuned_path = "models/finetuned_clip_v2"
            rf_path = os.path.join(finetuned_path, "random_forest_model.pkl")
            if os.path.exists(rf_path):
                import joblib
                logger.info("Loading Random Forest model from %s...", rf_path)
                try:
                    model_data = joblib.load(rf_path)
                    self._rf_model = model_data["classifier"]
                    self._rf_classes = model_data["classes"]
                    logger.info("Random Forest model loaded successfully with %d classes", len(self._rf_classes))
                except Exception as e:
                    logger.error("Failed to load Random Forest model: %s", e)
        return self._rf_model


    # ── CLIP text embedding cache ────────────────────────────────────────────

    def _get_text_features(self) -> "torch.Tensor":
        """
        Return normalised CLIP text embeddings for all document_types.
        Computed once on first call, then cached for the lifetime of the process.
        Saves ~50–150 ms per classification request on CPU.
        """
        if self._text_features_cache is None:
            logger.info(
                "Pre-computing CLIP text embeddings for %d labels…",
                len(self.document_types),
            )
            t0 = time.perf_counter()
            text_inputs = self.clip_processor(
                text=self.document_types, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                text_output = self.clip_model.get_text_features(**text_inputs)
                if hasattr(text_output, "text_embeds"):
                    feats = text_output.text_embeds
                elif hasattr(text_output, "pooler_output"):
                    feats = text_output.pooler_output
                else:
                    feats = text_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
            self._text_features_cache = feats
            logger.info(
                "CLIP text embeddings cached in %.1f ms",
                (time.perf_counter() - t0) * 1000,
            )
        return self._text_features_cache

    # ── Classification ───────────────────────────────────────────────────────

    def get_image_embedding(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Extract normalized CLIP features for a PIL image.
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.load()
        inputs = self.clip_processor(images=pil_image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            image_output = self.clip_model.get_image_features(**inputs)
            if hasattr(image_output, "image_embeds"):
                image_feats = image_output.image_embeds
            elif hasattr(image_output, "pooler_output"):
                image_feats = image_output.pooler_output
            else:
                image_feats = image_output
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        return image_feats

    def classify_document(self, pil_image: Image.Image) -> Tuple[str, float]:
        """
        Fast classification — uses cached text embeddings.
        Returns (document_type, top_confidence).
        """
        doc_type, confidence, _ = self.classify_document_detailed(pil_image)
        return doc_type, confidence

    def classify_document_detailed(
        self, pil_image: Image.Image
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Full classification with per-label scores.
        Uses Random Forest if available; otherwise falls back to CLIP cosine similarity.
        Returns (top_doc_type, top_confidence, [(label, score), ...] sorted descending).
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.load()
        inputs = self.clip_processor(images=pil_image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            image_output = self.clip_model.get_image_features(**inputs)
            if hasattr(image_output, "image_embeds"):
                image_feats = image_output.image_embeds
            elif hasattr(image_output, "pooler_output"):
                image_feats = image_output.pooler_output
            else:
                image_feats = image_output
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

            # 1. Try Random Forest prediction
            rf = self.rf_model
            if rf is not None and self._rf_classes is not None:
                image_feats_np = image_feats.squeeze(0).cpu().numpy().reshape(1, -1)
                rf_probs = rf.predict_proba(image_feats_np)[0]
                
                rf_classes_probs = {
                    self._rf_classes[i]: float(rf_probs[i])
                    for i in range(len(self._rf_classes))
                }
                
                # Fill missing document types with 0.0 to ensure response completeness
                all_scores_dict = {doc_type: 0.0 for doc_type in self.document_types}
                all_scores_dict.update(rf_classes_probs)
                
                scores = list(all_scores_dict.items())
                scores.sort(key=lambda x: x[1], reverse=True)
                top_type, top_conf = scores[0]
                return top_type, top_conf, scores

            # 2. Fallback to CLIP cosine similarity
            text_feats = self._get_text_features()
            sims = (image_feats @ text_feats.T).squeeze(0)
            probs = torch.nn.functional.softmax(sims / _CONFIDENCE_TEMPERATURE, dim=-1)

        scores = [
            (self.document_types[i], float(sims[i]))
            for i in range(len(self.document_types))
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_type, _ = scores[0]
        top_idx = int(torch.argmax(sims))
        top_conf = float(probs[top_idx])
        return top_type, top_conf, scores


    # ── OCR ─────────────────────────────────────────────────────────────────

    def run_ocr(self, image_bytes: bytes) -> Tuple[str, float, list]:
        """
        Run EasyOCR.
        Returns (full_text, avg_confidence_pct_0_100, raw_results).
        raw_results: list of (bbox, text, confidence_0_1)
        """
        results = self.easy_reader.readtext(image_bytes)
        texts = [r[1] for r in results]
        avg_conf = (sum(r[2] for r in results) / len(results) * 100) if results else 0.0
        return " ".join(texts), avg_conf, results

    # ── Face detection ───────────────────────────────────────────────────────

    def detect_face(self, pil_image: Image.Image) -> dict:
        """
        Detect face using Haar cascade.
        Resizes to ≤640 px first for speed.
        Confidence is an area-ratio heuristic (larger face = higher confidence).
        """
        # Resize for speed — Haar cascade doesn't need high resolution
        w, h = pil_image.size
        max_edge = max(w, h)
        if max_edge > _FACE_DETECT_MAX_EDGE:
            scale = _FACE_DETECT_MAX_EDGE / max_edge
            detect_img = pil_image.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.BILINEAR,
            )
            inv_scale = 1.0 / scale
        else:
            detect_img = pil_image
            inv_scale = 1.0

        cv_image = cv2.cvtColor(np.array(detect_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4
        )

        if len(faces) == 0:
            return {"face_detected": False, "confidence": 0.0, "bounding_box": None}

        # Pick the largest face
        areas = [fw * fh for (fx, fy, fw, fh) in faces]
        largest_idx = int(np.argmax(areas))
        fx, fy, fw, fh = faces[largest_idx]

        # Scale bounding box back to original image coordinates
        bx = int(fx * inv_scale)
        by = int(fy * inv_scale)
        bw = int(fw * inv_scale)
        bh = int(fh * inv_scale)

        # Area-ratio confidence heuristic: 0.5 baseline, up to 0.99 for large faces
        face_area_ratio = (fw * fh) / (gray.shape[0] * gray.shape[1])
        if face_area_ratio >= 0.15:
            confidence = 0.95
        elif face_area_ratio >= 0.07:
            confidence = 0.85
        elif face_area_ratio >= 0.03:
            confidence = 0.70
        else:
            confidence = 0.55

        return {
            "face_detected": True,
            "confidence": confidence,
            "bounding_box": [bx, by, bw, bh],
        }

    def get_face_base64(self, pil_image: Image.Image, face_bbox: list) -> "str | None":
        """Crop the detected face and return as base64-encoded PNG."""
        if not face_bbox:
            return None
        x, y, w, h = face_bbox
        margin = 0.2
        img_w, img_h = pil_image.size
        x1 = max(0, x - int(w * margin))
        y1 = max(0, y - int(h * margin))
        x2 = min(img_w, x + w + int(w * margin))
        y2 = min(img_h, y + h + int(h * margin))
        face_crop = pil_image.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        face_crop.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Delegation helpers ───────────────────────────────────────────────────

    def normalize_pil_image(self, pil_image: Image.Image) -> Image.Image:
        return document_scanner.normalize_pil_image(pil_image)

    def detect_and_crop_document(self, pil_image: Image.Image):
        return document_scanner.detect_quad_opencv(pil_image)

    def enhance_document_scan(self, pil_image: Image.Image, use_google: bool = False):
        scanned, detected, _method = document_scanner.scan(
            pil_image, use_google=use_google
        )
        return scanned, detected


ai_models = AIModelsService()
