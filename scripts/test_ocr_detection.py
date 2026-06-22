import os
import glob
import easyocr
from PIL import Image

def test_ocr():
    reader = easyocr.Reader(['en', 'fr'], verbose=False)
    dataset_dir = "data/dataset"
    
    categories = ["CERTIFICATES", "DRIVING LICENSE", "ID CARDS", "passport"]
    
    for cat in categories:
        cat_path = os.path.join(dataset_dir, cat)
        if not os.path.exists(cat_path):
            print(f"Path does not exist: {cat_path}")
            continue
            
        images = glob.glob(os.path.join(cat_path, "*.png")) + glob.glob(os.path.join(cat_path, "*.jpg")) + glob.glob(os.path.join(cat_path, "*.jfif")) + glob.glob(os.path.join(cat_path, "*.jpeg"))
        if not images:
            print(f"No images in {cat}")
            continue
            
        img_path = images[0]
        print(f"\n--- Testing OCR on {cat}: {os.path.basename(img_path)} ---")
        try:
            results = reader.readtext(img_path)
            for i, res in enumerate(results[:15]):  # Show first 15 detections
                bbox, text, conf = res
                print(f"[{i}] {text} (Conf: {conf:.2f}) at BBox: {bbox}")
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

if __name__ == "__main__":
    test_ocr()
