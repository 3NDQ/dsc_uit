# extract_features.py
import numpy as np
import json
import os
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, AutoModelForImageClassification
import cv2
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data, image_folder, ocr_cache_path, mode="train"):
        self.data = data
        self.image_folder = image_folder
        self.mode = mode

        # Load OCR cache
        if os.path.exists(ocr_cache_path):
            with open(ocr_cache_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            self.ocr_data = {os.path.basename(k): v for k, v in json_data.items()}
        else:
            raise FileNotFoundError(f"OCR cache file not found at {ocr_cache_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_id, item = list(self.data.items())[idx]
        image_name = item["image"]
        text = item["caption"]
        image_path = os.path.join(self.image_folder, image_name)
        ocr_text = self.ocr_data.get(image_name, "")

        if self.mode == "train":
            label = item["label"]
            return image_path, text, ocr_text, label, item_id
        else:
            return image_path, text, ocr_text, item_id

def collate_fn(batch):
    image_paths, texts, ocr_texts, *labels = zip(*batch)
    return list(image_paths), list(texts), list(ocr_texts), labels[0] if labels else None

def extract_and_save_features(data_path, image_folder, ocr_cache_path, output_dir, mode="train", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)  
    image_encoder = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device).to(torch.float32)
    text_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True, use_flash_attn=False)
    text_encoder = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True, use_flash_attn=False).to(device).to(torch.float32)

    # Load data from JSON
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create dataset and dataloader
    dataset = CustomDataset(data, image_folder, ocr_cache_path, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_image_features = []
    all_text_features = []
    all_labels = []

    for batch in tqdm(dataloader, desc=f"Extracting features for {mode} data"):
        image_paths, texts, ocr_texts, labels = batch
        image_features, text_features = preprocess_batch(
            image_paths, texts, ocr_texts, image_processor, image_encoder, text_tokenizer, text_encoder, device
        )

        all_image_features.extend(image_features)
        all_text_features.extend(text_features)

        if mode == "train":
            label_to_id = {
                'multi-sarcasm': 0, 
                'text-sarcasm': 1, 
                'image-sarcasm': 2, 
                'not-sarcasm': 3,
            }
            for item_id, label in zip(labels[1], labels[0]):
                all_labels.append({
                    "item_id": item_id,
                    "label_id": label_to_id.get(label, 3)
                })
        elif mode == "test":
            for item_id in labels[0]:
                all_labels.append({
                    "item_id": item_id
                })

    # Save features and labels (or identifiers for test mode)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "image_features.npy"), np.array(all_image_features))
    np.save(os.path.join(output_dir, "text_features.npy"), np.array(all_text_features))

    with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2)

    print(f"Features saved to {output_dir}")

def preprocess_batch(image_paths, texts, ocr_texts, image_processor, image_encoder, text_tokenizer, text_encoder, device):
    image_features = []
    text_features = []

    # Process images in batch
    images = [cv2.imread(image_path) for image_path in image_paths]
    inputs = image_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_outputs = image_encoder(**inputs)
    image_features_batch = image_outputs.logits.cpu().numpy()

    # Process OCR texts in batch
    ocr_inputs = text_tokenizer(
        ocr_texts, 
        return_tensors="pt", 
        padding="longest",
        truncation=True, 
        max_length=512
    ).to(device)
    with torch.no_grad():
        ocr_outputs = text_encoder(**ocr_inputs)
    ocr_features_batch = ocr_outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    # Combine image and OCR features
    for img_feat, ocr_feat in zip(image_features_batch, ocr_features_batch):
        combined_features = np.concatenate([img_feat, ocr_feat])
        image_features.append(combined_features)

    # Process texts in batch
    text_inputs = text_tokenizer(
        texts, 
        return_tensors="pt", 
        padding="longest",
        truncation=True, 
        max_length=512
    ).to(device)
    with torch.no_grad():
        text_outputs = text_encoder(**text_inputs)
    text_features_batch = text_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    text_features.extend(text_features_batch)

    return image_features, text_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from image and text data.")
    parser.add_argument("--data_path", required=True, help="Path to the JSON data file.")
    parser.add_argument("--image_folder", required=True, help="Path to the folder containing images.")
    parser.add_argument("--ocr_cache_path", required=True, help="Path to the OCR cache file.")
    parser.add_argument("--output_dir", required=True, help="Path to the directory where features will be saved.")
    parser.add_argument("--mode", default="train", choices=["train", "test"], help="Mode: 'train' or 'test'.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")

    args = parser.parse_args()

    extract_and_save_features(
        data_path=args.data_path,
        image_folder=args.image_folder,
        ocr_cache_path=args.ocr_cache_path,
        output_dir=args.output_dir,
        mode=args.mode,
        batch_size=args.batch_size
    )