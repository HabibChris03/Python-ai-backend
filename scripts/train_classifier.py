import argparse
import glob
import json
import os

import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

LABEL_MAP = {
    "passport": "cameroon passport",
    "driving license": "cameroon driving license",
    "driving_license": "cameroon driving license",
    "id cards": "cameroon national id card",
    "id_cards": "cameroon national id card",
    "id card": "cameroon national id card",
    "id_card": "cameroon national id card",
    "birth certificate": "cameroon birth certificate",
    "birth_certificate": "cameroon birth certificate",
    "birth_certificates": "cameroon birth certificate",
    "marriage certificate": "cameroon marriage certificate",
    "marriage_certificate": "cameroon marriage certificate",
    "certificates": "cameroon certificate",
    "certificate": "cameroon certificate",
    "legal document": "cameroon legal document",
    "legal_document": "cameroon legal document",
    "official document": "cameroon official document",
    "official_document": "cameroon official document",
    "government document": "cameroon official document",
    "government_document": "cameroon official document",
}


def infer_label_from_filename(filename: str) -> str | None:
    filename = filename.lower()
    if "passport" in filename:
        return "cameroon passport"
    if "driving" in filename or "license" in filename or "licence" in filename:
        return "cameroon driving license"
    if "national_id" in filename or "national id" in filename or "id card" in filename or "id_card" in filename:
        return "cameroon national id card"
    if "birth" in filename and "certificate" in filename:
        return "cameroon birth certificate"
    if "marriag" in filename and "certificate" in filename:
        return "cameroon marriage certificate"
    if "property" in filename and "title" in filename:
        return "cameroon property title land certificate"
    if "health" in filename or "hospital" in filename or "medical" in filename:
        return "cameroon hospital medical health record"
    if "fslc" in filename or "school leaving" in filename:
        return "cameroon first school leaving certificate education"
    if "common_entrance" in filename or "common entrance" in filename:
        return "cameroon government common entrance certificate education"
    if "certificate" in filename:
        return "cameroon certificate"
    if "legal" in filename:
        return "cameroon legal document"
    if "document" in filename and not any(skip in filename for skip in ["bank card", "credit card", "receipt", "invoice", "contract", "resume", "business card"]):
        return "cameroon official document"
    return None


class CameroonDocDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.jfif"]
        image_paths = []
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(self.root_dir, pattern), recursive=True))

        for img_path in sorted(set(image_paths)):
            rel_path = os.path.relpath(img_path, self.root_dir)
            parts = rel_path.replace("\\", "/").split("/")
            label_text = None

            if len(parts) > 1:
                folder_name = parts[0].lower()
                label_text = LABEL_MAP.get(folder_name, f"cameroon {folder_name.replace('_', ' ')}")

            if not label_text:
                json_path = os.path.splitext(img_path)[0] + ".json"
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as handle:
                        label_data = json.load(handle)
                    doc_type = label_data.get("document_type", "document")
                    label_text = LABEL_MAP.get(doc_type, doc_type)

            if not label_text:
                filename_label = infer_label_from_filename(os.path.basename(img_path))
                label_text = LABEL_MAP.get(filename_label, filename_label) if filename_label else None

            if label_text:
                self.samples.append((img_path, label_text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(
            text=[text_label],
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)
        return inputs


def train_random_forest(model, processor, dataset, output_dir, device):
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    print("\n--- Training Random Forest Classifier ---")
    print("Extracting CLIP image features for Random Forest training...")
    model.eval()
    
    X = []
    y = []
    
    # We will use the dataset's samples
    unique_labels = sorted(list(set(label for _, label in dataset.samples)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    with torch.no_grad():
        for img_path, label_text in tqdm(dataset.samples, desc="Extracting visual features"):
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                image_output = model.get_image_features(**inputs)
                
                # Extract and normalize
                if hasattr(image_output, "image_embeds"):
                    feats = image_output.image_embeds
                elif hasattr(image_output, "pooler_output"):
                    feats = image_output.pooler_output
                else:
                    feats = image_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
                
                X.append(feats.squeeze(0).cpu().numpy())
                y.append(label_to_idx[label_text])
            except Exception as e:
                print(f"Error extracting features for {img_path}: {e}")
                
    if not X:
        print("No features extracted. Random Forest training skipped.")
        return
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Features shape: {X.shape}, labels shape: {y.shape}")
    print(f"Training Random Forest on {len(X)} samples across {len(unique_labels)} classes...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Save the model
    rf_model_path = os.path.join(output_dir, "random_forest_model.pkl")
    model_data = {
        "classifier": rf,
        "classes": unique_labels
    }
    joblib.dump(model_data, rf_model_path)
    print(f"Random Forest model successfully saved to {rf_model_path}")


def train(
    data_root,
    output_dir,
    batch_size=8,
    epochs=10,
    lr=1e-6,
    num_workers=0,
    use_amp=True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = use_amp and device == "cuda"
    print(f"Training on device: {device} (AMP: {use_amp})")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dataset = CameroonDocDataset(data_root, processor)
    if len(dataset) == 0:
        raise ValueError(
            f"No training samples in {data_root}. Run: python scripts/prepare_training_images.py"
        )

    print(f"Loaded {len(dataset)} labeled samples")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )
    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                labels = torch.arange(len(outputs.logits_per_image), device=device)
                loss_i = torch.nn.functional.cross_entropy(outputs.logits_per_image, labels)
                loss_t = torch.nn.functional.cross_entropy(outputs.logits_per_text, labels)
                loss = (loss_i + loss_t) * 0.5

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Fine-tuning complete. Model saved to {output_dir}")

    # Train Random Forest classifier using the fine-tuned CLIP features
    train_random_forest(model, processor, dataset, output_dir, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on labeled Cameroon documents")
    parser.add_argument("--data-root", default="data/training_docs")
    parser.add_argument("--output-dir", default="models/finetuned_clip_v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--only-rf", action="store_true", help="Only train/regenerate the Random Forest model using existing CLIP weights")
    args = parser.parse_args()

    if args.only_rf:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = args.output_dir if os.path.exists(os.path.join(args.output_dir, "model.safetensors")) else "openai/clip-vit-base-patch32"
        print(f"Bypassing fine-tuning. Loading CLIP model from: {model_path} on device: {device}")
        
        model = CLIPModel.from_pretrained(model_path).to(device)
        processor = CLIPProcessor.from_pretrained(model_path)
        
        dataset = CameroonDocDataset(args.data_root, processor)
        if len(dataset) == 0:
            raise ValueError(f"No training samples in {args.data_root}. Run prepare_training_images first.")
            
        print(f"Loaded {len(dataset)} labeled samples for feature extraction")
        train_random_forest(model, processor, dataset, args.output_dir, device)
    else:
        train(
            data_root=args.data_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            use_amp=not args.no_amp,
        )

