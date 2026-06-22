import random
import os
from PIL import Image, ImageDraw, ImageFont
import json

# Sample lists for Cameroonian data
FIRST_NAMES = ["Abena", "Eto'o", "Manga", "Sali", "Hadidja", "Foumban", "Ngassa", "Tchiroma", "Amadou", "Biya", "Paul", "Samuel", "Vincent"]
LAST_NAMES = ["Moussa", "Ekotto", "Aboubakar", "N'Koulou", "Zambo", "Anguissa", "Ngannou", "Song", "Ayuk", "Ndjoumou"]
REGIONS = ["Centre", "Littoral", "Adamaoua", "Extreme-Nord", "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest", "Est"]
CITIES = ["Yaounde", "Douala", "Garoua", "Maroua", "Bafoussam", "Ngaoundere", "Bamenda", "Kribi", "Limbe", "Buea"]

def generate_cameroon_id_data():
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    dob = f"{random.randint(1, 28):02}/{random.randint(1, 12):02}/{random.randint(1960, 2005)}"
    id_num = f"{random.randint(100000000, 999999999):09}"
    place_of_issue = random.choice(CITIES)
    issue_date = f"{random.randint(1, 28):02}/{random.randint(1, 12):02}/{random.randint(2010, 2024)}"
    expiry_date = f"{random.randint(1, 28):02}/{random.randint(1, 12):02}/{random.randint(2025, 2035)}"
    
    return {
        "full_name": f"{first_name} {last_name}",
        "date_of_birth": dob,
        "id_number": id_num,
        "place_of_issue": place_of_issue,
        "issue_date": issue_date,
        "expiry_date": expiry_date,
        "region": random.choice(REGIONS)
    }

def create_synthetic_id(data, output_path):
    # Create a blank "ID card" placeholder (800x500)
    img = Image.new('RGB', (800, 500), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        title_font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Draw header
    draw.rectangle([0, 0, 800, 80], fill=(0, 102, 204))  # Blue header
    draw.text((300, 25), "REPUBLIQUE DU CAMEROUN", fill=(255, 255, 255), font=title_font)
    
    # Draw content
    y = 120
    for key, value in data.items():
        draw.text((50, y), f"{key.replace('_', ' ').title()}: {value}", fill=(0, 0, 0), font=font)
        y += 50
        
    img.save(output_path)

if __name__ == "__main__":
    os.makedirs("data/synthetic_ids", exist_ok=True)
    for i in range(10):
        data = generate_cameroon_id_data()
        create_synthetic_id(data, f"data/synthetic_ids/nic_{i}.png")
        # Save labels
        with open(f"data/synthetic_ids/nic_{i}.json", 'w') as f:
            json.dump(data, f)
    print(f"Generated 10 synthetic Cameroon IDs in data/synthetic_ids/")
