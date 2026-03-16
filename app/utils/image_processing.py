import cv2
import numpy as np
from PIL import Image
import pytesseract
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for better OCR results"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def extract_text_with_tesseract(image: Image.Image) -> dict:
    """Extract text using Tesseract OCR"""
    try:
        processed_img = preprocess_image(image)
        config = '--oem 3 --psm 6'
        
        text = pytesseract.image_to_string(processed_img, config=config)
        # image_to_conf returns a string in newer versions, handle appropriately
        # for now, let's just use a placeholder or check version
        confidence = 0.0 # pytesseract confidence handling is tricky across platforms
        
        return {
            "text": text.strip(),
            "confidence": confidence,
            "bounding_boxes": [], # Simplified for now
            "language": "en"
        }
    except Exception as e:
        logger.error(f"Tesseract OCR error: {e}")
        return {"text": "", "confidence": 0, "bounding_boxes": [], "language": "en"}
