import random
import os
import json
import glob
from PIL import Image, ImageDraw, ImageFont

# Sample lists for Cameroonian data
FIRST_NAMES = ["Abena", "Eto'o", "Manga", "Sali", "Hadidja", "Foumban", "Ngassa", "Tchiroma", "Amadou", "Biya", "Paul", "Samuel", "Vincent", "Brenda", "Daphne"]
LAST_NAMES = ["Moussa", "Ekotto", "Aboubakar", "N'Koulou", "Zambo", "Anguissa", "Ngannou", "Song", "Ayuk", "Ndjoumou", "Mbappe", "Eto'o", "Onana"]
REGIONS = ["Centre", "Littoral", "Adamaoua", "Extreme-Nord", "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest", "Est"]
CITIES = ["Yaounde", "Douala", "Garoua", "Maroua", "Bafoussam", "Ngaoundere", "Bamenda", "Kribi", "Limbe", "Buea"]

FACES_DIR = "data/assets/faces"
OUTPUT_DIR = "data/training_docs"

def generate_person_data():
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    dob = f"{random.randint(1, 28):02}/{random.randint(1, 12):02}/{random.randint(1960, 2005)}"
    sex = random.choice(["M", "F"])
    place_of_birth = random.choice(CITIES)
    return {
        "first_name": first_name,
        "last_name": last_name,
        "full_name": f"{first_name} {last_name}",
        "date_of_birth": dob,
        "sex": sex,
        "place_of_birth": place_of_birth,
        "id_number": f"{random.randint(100000000, 999999999):09}",
        "passport_number": f"P{random.randint(1000000, 9999999):07}",
        "license_number": f"L{random.randint(1000000, 9999999):07}",
        "student_id": f"STU{random.randint(10000, 99999)}",
        "health_id": f"MED{random.randint(100000, 999999)}",
        "property_id": f"LOT-{random.randint(100, 999)}/{random.randint(1000, 9999)}",
    }

def create_document(doc_type, person_data, face_path, output_path):
    # Base dimensions
    width, height = 800, 500
    
    if doc_type == "passport":
        bg_color = (255, 245, 230) # Pale peach/beige
        header_color = (153, 0, 0) # Dark red
        title = "REPUBLIC OF CAMEROON - PASSPORT"
    elif doc_type == "national_id":
        bg_color = (240, 240, 255) # Pale blue
        header_color = (0, 102, 204) # Blue
        title = "REPUBLIQUE DU CAMEROUN - CARTE D'IDENTITE"
    elif doc_type == "fslc":
        bg_color = (255, 255, 240) # Cream
        header_color = (0, 102, 0) # Green
        title = "FIRST SCHOOL LEAVING CERTIFICATE"
    elif doc_type == "common_entrance":
        bg_color = (255, 250, 250) # Snow
        header_color = (128, 0, 128) # Purple
        title = "GOVERNMENT COMMON ENTRANCE"
    elif doc_type == "health_record":
        bg_color = (240, 255, 255) # Azure
        header_color = (204, 0, 0) # Medical Red
        title = "HOSPITAL MEDICAL RECORD"
    elif doc_type == "property_title":
        bg_color = (255, 248, 220) # Cornsilk
        header_color = (101, 67, 33) # Dark Brown
        title = "PROPERTY TITLE / LAND CERTIFICATE"
    else: # driving_license
        bg_color = (240, 255, 240) # Pale green
        header_color = (0, 102, 51) # Green
        title = "CAMEROON DRIVING LICENSE"

    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        title_font = ImageFont.truetype("arial.ttf", 25)
        label_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    # Draw header
    draw.rectangle([0, 0, width, 70], fill=header_color)
    draw.text((width//2 - 200, 20), title, fill=(255, 255, 255), font=title_font)
    
    # Paste face
    try:
        face_img = Image.open(face_path).convert("RGB")
        face_img = face_img.resize((180, 220))
        img.paste(face_img, (50, 100))
        # Draw border around face
        draw.rectangle([48, 98, 50+182, 100+222], outline=(100, 100, 100), width=2)
    except Exception as e:
        print(f"Error loading face {face_path}: {e}")
        draw.rectangle([50, 100, 230, 320], fill=(200, 200, 200))
        draw.text((80, 200), "PHOTO", fill=(50, 50, 50), font=font)

    # Draw person details
    x_offset = 260
    y_start = 100
    
    details = []
    if doc_type == "passport":
        details = [
            ("Surname", person_data["last_name"]),
            ("Given Names", person_data["first_name"]),
            ("Passport No", person_data["passport_number"]),
            ("Nationality", "CAMEROONIAN"),
            ("Date of Birth", person_data["date_of_birth"]),
            ("Sex", person_data["sex"]),
            ("Place of Birth", person_data["place_of_birth"]),
        ]
    elif doc_type == "national_id":
        details = [
            ("Nom", person_data["last_name"]),
            ("Prenom", person_data["first_name"]),
            ("N° Identifiant", person_data["id_number"]),
            ("Né(e) le", person_data["date_of_birth"]),
            ("Sexe", person_data["sex"]),
            ("Lieu de délivrance", random.choice(CITIES)),
        ]
    elif doc_type == "fslc":
        details = [
            ("Candidate Name", person_data["full_name"]),
            ("School Name", f"{random.choice(CITIES)} Government School"),
            ("Year of Award", "2018"),
            ("Center Number", f"CM-{random.randint(100, 999)}"),
            ("Session", "JUNE 2018"),
        ]
    elif doc_type == "common_entrance":
        details = [
            ("Student Name", person_data["full_name"]),
            ("Examination", "Government Common Entrance"),
            ("Candidate No", person_data["student_id"]),
            ("Result", "PASSED IN LIST A"),
            ("Exam Center", random.choice(CITIES)),
        ]
    elif doc_type == "health_record":
        details = [
            ("Patient Name", person_data["full_name"]),
            ("Medical ID", person_data["health_id"]),
            ("Blood Type", random.choice(["A+", "O+", "B+", "AB-"])),
            ("Consultation Date", "14/02/2024"),
            ("Hospital", f"General Hospital {random.choice(CITIES)}"),
        ]
    elif doc_type == "property_title":
        details = [
            ("Owner Name", person_data["full_name"]),
            ("Title Number", person_data["property_id"]),
            ("Location", f"{random.choice(REGIONS)} Region"),
            ("Surface Area", f"{random.randint(200, 2000)} sqm"),
            ("Registration Date", "20/11/2020"),
        ]
    else: # driving_license
        details = [
            ("Name", person_data["full_name"]),
            ("License No", person_data["license_number"]),
            ("Categories", "B, C, D"),
            ("Date of Issue", "12/05/2022"),
            ("Expiry Date", "12/05/2032"),
            ("Blood Group", random.choice(["A+", "O+", "B+", "AB-"])),
        ]

    y = y_start
    for label, value in details:
        draw.text((x_offset, y), label, fill=(100, 100, 100), font=label_font)
        draw.text((x_offset + 150, y), str(value), fill=(0, 0, 0), font=font)
        y += 45

    # Add a watermark-like logo (simple circle)
    draw.ellipse([650, 350, 750, 450], outline=(200, 200, 200), width=3)
    draw.text((675, 390), "SDRS", fill=(200, 200, 200), font=title_font)

    img.save(output_path)
    return {
        "document_type": doc_type,
        "person_info": person_data,
        "face_bbox": [50, 100, 230, 320] # [x_min, y_min, x_max, y_max]
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    face_files = glob.glob(os.path.join(FACES_DIR, "*.png"))
    
    if not face_files:
        print(f"No face images found in {FACES_DIR}. Please add some faces first.")
        return

    doc_types = ["passport", "national_id", "driving_license", "fslc", "common_entrance", "health_record", "property_title"]
    
    count = 0
    for i in range(100): # Generate 100 samples for better training
        doc_type = random.choice(doc_types)
        person_data = generate_person_data()
        face_path = random.choice(face_files)
        
        img_name = f"doc_{i}_{doc_type}.png"
        img_path = os.path.join(OUTPUT_DIR, img_name)
        json_path = os.path.join(OUTPUT_DIR, f"doc_{i}_{doc_type}.json")
        
        meta = create_document(doc_type, person_data, face_path, img_path)
        
        with open(json_path, 'w') as f:
            json.dump(meta, f, indent=4)
        
        count += 1
        
    print(f"Generated {count} documents in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
