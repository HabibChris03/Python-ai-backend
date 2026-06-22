import requests
import os
import glob
import random

def test_recognition():
    url = "http://localhost:8000/api/recognition/image-recognition"
    
    # Find a sample image
    sample_images = glob.glob("ai_backend/data/training_docs/*.png")
    if not sample_images:
        print("No sample images found. Run advanced_doc_gen.py first.")
        return
        
    img_path = random.choice(sample_images)
    print(f"Testing with: {img_path}")
    
    with open(img_path, 'rb') as f:
        files = {'file': (os.path.basename(img_path), f, 'image/png')}
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                print("Success!")
                import json
                print(json.dumps(response.json(), indent=4))
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Could not connect to server: {e}")

if __name__ == "__main__":
    test_recognition()
