import os
import glob
import random
import json
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import easyocr

# Cameroon synthetic identity data pool
FIRST_NAMES = [
    "Abena", "Eto'o", "Manga", "Sali", "Hadidja", "Foumban", "Ngassa", "Tchiroma", 
    "Amadou", "Biya", "Paul", "Samuel", "Vincent", "Emmanuel", "Roger", "Albert", 
    "Rigobert", "Nicolas", "Eric", "Alex", "Marc", "Joel", "Andre", "Jean", "Pierre", 
    "Marie", "Florence", "Pauline", "Chantal", "Jacqueline", "Georgette", "Beatrice", 
    "Sidonie", "Nadine", "Christelle", "Grace", "Blessing", "Faith", "Joy", "Patience",
    "Joseph", "Therese", "Henriette", "Ahmadou", "Bello", "Boubakary", "Ali", "Ousmanou"
]

LAST_NAMES = [
    "Moussa", "Ekotto", "Aboubakar", "N'Koulou", "Zambo", "Anguissa", "Ngannou", "Song", 
    "Ayuk", "Ndjoumou", "Mbappe", "Milla", "Emana", "Webo", "Kameni", "N'Guemo", 
    "Choupo", "Moting", "Toko", "Ekambi", "Oyongo", "Fai", "Ngadeu", "Castelletto",
    "Njie", "Nsame", "Ondoa", "Nguemo", "Tchoutou", "Nlatte", "Atangana", "Ebolo",
    "Mebenga", "Assou", "Ekotto", "Biloa", "Belinga", "Tchakoute", "Ndam", "Sop"
]

REGIONS = [
    "Centre", "Littoral", "Adamaoua", "Extreme-Nord", "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest", "Est"
]

CITIES = [
    "Yaounde", "Douala", "Garoua", "Maroua", "Bafoussam", "Ngaoundere", "Bamenda", "Kribi", "Limbe", "Buea",
    "Dschang", "Foumban", "Ebolowa", "Bertoua", "Edéa", "Kumba", "Loum", "Nkongsamba", "Maroua", "Guider"
]

def generate_random_cameroon_data():
    fn = random.choice(FIRST_NAMES)
    ln = random.choice(LAST_NAMES)
    dob = f"{random.randint(1, 28):02}/{random.randint(1, 12):02}/{random.randint(1960, 2005)}"
    id_num = f"{random.randint(100000000, 999999999):09}"
    lic_num = f"{random.randint(100000, 999999):06}/{random.randint(1, 10)}"
    pass_num = f"09{random.randint(1000000, 9999999)}"
    issue_date = f"{random.randint(1, 28):02}/{random.randint(1, 12):02}/{random.randint(2015, 2024)}"
    expiry_date = f"{random.randint(1, 28):02}/{random.randint(1, 12):02}/{random.randint(2026, 2035)}"
    
    return {
        "first_name": fn,
        "last_name": ln,
        "full_name": f"{ln} {fn}".upper(),
        "date_of_birth": dob,
        "id_number": id_num,
        "license_number": lic_num,
        "passport_number": pass_num,
        "issue_date": issue_date,
        "expiry_date": expiry_date,
        "place_of_issue": random.choice(CITIES),
        "region": random.choice(REGIONS)
    }

def try_load_font(font_size=16):
    paths = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\times.ttf",
        "C:\\Windows\\Fonts\\cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except:
                pass
    return ImageFont.load_default()

def get_surrounding_color(img, bbox):
    # Sample the color near the bounding box to erase the text cleanly
    x1, y1 = int(bbox[0][0]), int(bbox[0][1])
    x2, y2 = int(bbox[2][0]), int(bbox[2][1])
    
    w, h = img.size
    sample_points = []
    if x1 > 5:
        sample_points.append(img.getpixel((x1 - 3, (y1 + y2) // 2)))
    if x2 < w - 5:
        sample_points.append(img.getpixel((x2 + 3, (y1 + y2) // 2)))
    if y1 > 5:
        sample_points.append(img.getpixel(((x1 + x2) // 2, y1 - 3)))
        
    if sample_points:
        colors = np.array(sample_points)
        if len(colors.shape) == 1:
            return int(np.median(colors))
        else:
            return tuple(np.median(colors, axis=0).astype(int))
    return (240, 240, 240) if img.mode == 'RGB' else 240

def apply_photocopy_filters(img):
    # Convert to grayscale
    img_gray = img.convert('L')
    
    # 1. High contrast enhancement
    enhancer = ImageEnhance.Contrast(img_gray)
    img_contrast = enhancer.enhance(random.uniform(1.8, 3.5))
    
    # 2. Photocopy threshold / Binarization (mimics dark Xerox printer toner)
    threshold = random.randint(110, 160)
    img_bin = img_contrast.point(lambda p: 255 if p > threshold else random.randint(0, 40))
    
    # 3. Add paper noise / speckles
    np_img = np.array(img_bin)
    h, w = np_img.shape
    noise_count = int(w * h * random.uniform(0.001, 0.008))
    for _ in range(noise_count):
        ny = random.randint(0, h - 1)
        nx = random.randint(0, w - 1)
        np_img[ny, nx] = random.choice([0, 100, 150])
        
    img_noisy = Image.fromarray(np_img)
    
    # 4. Blur slightly (mimics lens/glass scan blur)
    img_blur = img_noisy.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.6)))
    
    # 5. Add border scanner shadow
    draw = ImageDraw.Draw(img_blur)
    shadow_width = random.randint(5, 20)
    draw.rectangle([0, 0, w, shadow_width], fill=0)
    draw.rectangle([0, h - shadow_width, w, h], fill=0)
    if random.random() > 0.5:
        draw.rectangle([0, 0, shadow_width, h], fill=0)
        
    # 6. Rotate slightly (skew)
    angle = random.uniform(-3.5, 3.5)
    img_rot = img_blur.rotate(angle, expand=True, fillcolor=255)
    
    return img_rot

def process_and_generate(template_path, category, num_variations, output_dir, reader):
    print(f"Processing template: {os.path.basename(template_path)} for category: {category}")
    
    try:
        base_img = Image.open(template_path)
    except Exception as e:
        print(f"Error opening image {template_path}: {e}")
        return
        
    ocr_results = []
    if base_img.size[0] > 150 and base_img.size[1] > 150:
        try:
            ocr_results = reader.readtext(np.array(base_img))
        except Exception as e:
            print(f"OCR failed for {template_path}: {e}")
            
    targets = []
    for res in ocr_results:
        bbox, text, conf = res
        text_upper = text.upper().strip()
        
        is_date = re.search(r'\d{2}[/.-]\d{2}[/.-]\d{4}', text) is not None
        is_number = re.search(r'\b\d{6,10}\b', text) is not None
        is_name = False
        
        if len(text_upper) > 3 and text_upper.isalpha() and conf > 0.4:
            if not any(keyword in text_upper for keyword in ["CAMEROUN", "REPUBLIC", "REPUBLIQUE", "PASSPORT", "PASSEPORT", "LICENCE", "CONDUIRE", "DELEGUE", "GENERAL", "SURETE", "NATIONALE"]):
                is_name = True
                
        if is_date or is_number or is_name:
            targets.append({
                "bbox": bbox,
                "text": text,
                "type": "date" if is_date else ("number" if is_number else "name")
            })
            
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)
    font = try_load_font(18)
    
    for i in range(num_variations):
        img_copy = base_img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        cam_data = generate_random_cameroon_data()
        
        for tgt in targets:
            bbox = tgt["bbox"]
            tgt_type = tgt["type"]
            
            if tgt_type == "date":
                replacement = cam_data["date_of_birth"] if random.random() > 0.5 else cam_data["issue_date"]
            elif tgt_type == "number":
                if category.lower() == "passport":
                    replacement = cam_data["passport_number"]
                elif "license" in category.lower() or "driving" in category.lower():
                    replacement = cam_data["license_number"]
                else:
                    replacement = cam_data["id_number"]
            else:
                replacement = random.choice([cam_data["first_name"], cam_data["last_name"]])
                if tgt["text"].isupper():
                    replacement = replacement.upper()
            
            bg_color = get_surrounding_color(img_copy, bbox)
            x1, y1 = int(bbox[0][0]), int(bbox[0][1])
            x2, y2 = int(bbox[2][0]), int(bbox[2][1])
            draw.rectangle([x1, y1, x2, y2], fill=bg_color)
            draw.text((x1, y1 - 2), replacement, fill=(0, 0, 0) if img_copy.mode == 'RGB' else 0, font=font)
            
        photocopy_img = apply_photocopy_filters(img_copy)
        
        out_name = f"photocopy_{os.path.splitext(os.path.basename(template_path))[0]}_var_{i}.png"
        out_path = os.path.join(output_dir, category, out_name)
        photocopy_img.save(out_path)
        
        label_path = os.path.splitext(out_path)[0] + ".json"
        with open(label_path, 'w', encoding='utf-8') as lf:
            json.dump({
                "original_template": os.path.basename(template_path),
                "category": category,
                "document_type": f"cameroon {category.lower()}",
                "extracted_fields": cam_data
            }, lf, indent=2)

def generate_all_dataset():
    reader = easyocr.Reader(['en', 'fr'], verbose=False)
    
    dataset_dir = "data/dataset"
    output_dir = "data/training_docs"
    
    categories = {
        "CERTIFICATES": "certificates",
        "DRIVING LICENSE": "driving_license",
        "ID CARDS": "id_cards",
        "passport": "passport"
    }
    
    target_count_per_category = 300
    
    for folder_name, cat_label in categories.items():
        cat_path = os.path.join(dataset_dir, folder_name)
        if not os.path.exists(cat_path):
            print(f"Category folder not found: {cat_path}")
            continue
            
        templates = glob.glob(os.path.join(cat_path, "*.png")) + glob.glob(os.path.join(cat_path, "*.jpg")) + glob.glob(os.path.join(cat_path, "*.jfif")) + glob.glob(os.path.join(cat_path, "*.jpeg"))
        unique_templates = sorted(list(set([t for t in templates if " - Copy" not in t])))
        
        if not unique_templates:
            print(f"No templates found in {cat_path}")
            continue
            
        print(f"\n===== Found {len(unique_templates)} templates in {folder_name} =====")
        
        vars_per_template = target_count_per_category // len(unique_templates)
        remainder = target_count_per_category % len(unique_templates)
        
        for idx, template in enumerate(unique_templates):
            count = vars_per_template
            if idx < remainder:
                count += 1
            process_and_generate(template, cat_label, count, output_dir, reader)

if __name__ == "__main__":
    generate_all_dataset()
    print("\nDataset generation successful! Created 1200 Cameroon document photocopy images in data/training_docs/")
