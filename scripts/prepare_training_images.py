"""
Render training PNGs from JSON labels in data/training_docs when images are missing.
Run before train_classifier.py:

    python scripts/prepare_training_images.py
    python scripts/train_classifier.py --data-root data/training_docs
"""
import argparse
import glob
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int = 18):
    for path in [
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def render_document(label: dict, width: int = 900, height: int = 560) -> Image.Image:
    category = (label.get("category") or "document").replace("_", " ").title()
    doc_type = label.get("document_type", f"Cameroon {category}")
    identity = label.get("identity") or label.get("extracted_fields") or {}

    img = Image.new("RGB", (width, height), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    title_font = load_font(28)
    body_font = load_font(18)

    draw.rectangle([24, 24, width - 24, height - 24], outline=(30, 30, 30), width=3)
    draw.text((48, 48), "REPUBLIQUE DU CAMEROUN", fill=(20, 20, 20), font=body_font)
    draw.text((48, 88), doc_type.upper(), fill=(10, 10, 10), font=title_font)

    y = 150
    fields = [
        ("Surname", identity.get("surname") or identity.get("last_name")),
        ("Given names", identity.get("given_names") or identity.get("first_name")),
        ("Date of birth", identity.get("date_of_birth")),
        ("Place of birth", identity.get("place_of_birth")),
        ("ID number", identity.get("id_number")),
        ("Passport", identity.get("passport_number")),
        ("License", identity.get("license_number")),
        ("Issue date", identity.get("date_of_issue") or identity.get("issue_date")),
        ("Expiry", identity.get("date_of_expiry") or identity.get("expiry_date")),
        ("Region", identity.get("region")),
    ]

    for label_text, value in fields:
        if not value:
            continue
        draw.text((48, y), f"{label_text}: {value}", fill=(25, 25, 25), font=body_font)
        y += 34
        if y > height - 80:
            break

    # subtle noise for visual variety
    for _ in range(120):
        x = random.randint(30, width - 30)
        yp = random.randint(30, height - 30)
        shade = random.randint(200, 235)
        draw.point((x, yp), fill=(shade, shade, shade))

    return img


def prepare(data_root: str, force: bool = False) -> int:
    json_files = glob.glob(os.path.join(data_root, "**", "*.json"), recursive=True)
    created = 0

    for json_path in sorted(json_files):
        png_path = os.path.splitext(json_path)[0] + ".png"
        if os.path.exists(png_path) and not force:
            continue

        with open(json_path, "r", encoding="utf-8") as handle:
            label = json.load(handle)

        image = render_document(label)
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        image.save(png_path, "PNG")
        created += 1

    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PNG training images from JSON labels")
    parser.add_argument("--data-root", default="data/training_docs")
    parser.add_argument("--force", action="store_true", help="Regenerate even if PNG exists")
    args = parser.parse_args()

    count = prepare(args.data_root, force=args.force)
    print(f"Created {count} training images in {args.data_root}")
