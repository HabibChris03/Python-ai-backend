import torch
import logging
import os
import easyocr
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class AIModelsService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() and settings.DEVICE == "cuda" else "cpu"
        logger.info(f"Initializing AI models on device: {self.device}")
        
        # OCR Reader
        self.easy_reader = easyocr.Reader(['en'], gpu=(self.device == "cuda"))
        
        # CLIP Model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Sentence Transformer
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # LLM
        if settings.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo", 
                temperature=0.7, 
                api_key=settings.OPENAI_API_KEY
            )
        else:
            logger.warning("OPENAI_API_KEY not set. Chatbot functionality will be limited.")
            self.llm = None
            
        self.document_types = [
            "passport", "driver's license", "credit card", "receipt", "invoice", 
            "contract", "resume", "business card", "medical record", "bank statement",
            "utility bill", "insurance card", "ID card", "letter", "note"
        ]
        
        logger.info("AI models service initialized successfully")

# Singleton instance
ai_models = AIModelsService()
