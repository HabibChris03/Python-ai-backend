import json
import logging
from typing import Optional

from app.services.ai_models import ai_models
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ExtractedDocument(BaseModel):
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    id_number: Optional[str] = None
    expiry_date: Optional[str] = None
    nationality: Optional[str] = None
    address: Optional[str] = None


class ExtractionService:
    async def extract_fields(self, ocr_text: str, document_type: str) -> dict:
        """
        Extract specific fields from OCR text based on document type.
        Uses Groq to parse unstructured OCR text into structured data.
        """
        if not ai_models.llm:
            logger.warning("Extraction skipped: Groq LLM not configured.")
            return {}

        prompt = (
            f"Extract the following fields from the OCR text of a {document_type}.\n"
            "Return a JSON object with these exact keys (use null if a field is not found):\n"
            "{\n"
            '  "full_name": "...",\n'
            '  "date_of_birth": "...",\n'
            '  "id_number": "...",\n'
            '  "expiry_date": "...",\n'
            '  "nationality": "...",\n'
            '  "address": "..."\n'
            "}\n\n"
            f"OCR Text:\n{ocr_text}\n\n"
            "Return ONLY the JSON object, no explanation or markdown."
        )

        try:
            response = await ai_models.llm.ainvoke(prompt)
            content = response.content.strip()
            # Strip markdown code fences if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            try:
                parsed = json.loads(content)
                return ExtractedDocument.model_validate(parsed).model_dump(
                    exclude_none=False
                )
            except Exception:
                return {}
        except Exception:
            logger.exception("Field extraction error")
            return {}


extractor = ExtractionService()
