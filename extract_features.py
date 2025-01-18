# extract_features.py
import numpy as np
import json
import os
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import cv2
import pandas as pd
import torch
from tqdm import tqdm
import argparse

def extract_and_save_features(data_path, image_folder, ocr_cache_path, output_dir, mode="train"):
    # Initialize models and processors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224-in21k")  
    vit_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
    text_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
    text_encoder = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True).to(device)
    
    # Load data from JSON
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_image_features = []
    all_text_features = []
    all_labels = []

    for item_id, item in tqdm(data.items(), desc=f"Extracting features for {mode} data"):
        image_features, text_features = preprocess_data(
            [item["image"]], 
            [item["caption"]],
            vit_processor,
            vit_model,
            text_tokenizer,
            text_encoder,
            image_folder,
            ocr_cache_path,
            mode
        )
        
        all_image_features.append(image_features)
        all_text_features.append(text_features)
        if mode == "train":
            label_to_id = {
                'multi-sarcasm': 0, 
                'text-sarcasm': 1, 
                'image-sarcasm': 2, 
                'not-sarcasm': 3,
            }
            all_labels.append({
                "item_id": item_id,
                "label_id": label_to_id.get(item["label"], 3) # Correctly access "label" and map to ID
            })

        elif mode == "test":
            # For test mode, you might just want to store item IDs or other identifiers
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

def preprocess_data(images, texts, vit_processor, vit_model, text_tokenizer, text_encoder, image_folder, ocr_cache_path, mode='train'):
    image_features = []
    ocr_features = []
    total_images = len(images)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_json_file_path = ocr_cache_path

    if os.path.exists(input_json_file_path):
        with open(input_json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        data = []
        for image_path, text in json_data.items():
            image_name = os.path.basename(image_path)
            data.append({"image_path": image_name, "ocr_text": text})
        df = pd.DataFrame(data)
        existing_images = df["image_path"].tolist()
        df["ocr_text"] = df["ocr_text"].fillna("").astype(str)
    else:
        raise FileNotFoundError(f"JSON file not found at {input_json_file_path}")

    for i, image_name in enumerate(images, 1):
        try:
            image_path = os.path.join(image_folder, image_name)
            img = cv2.imread(image_path)

            # Process the image using ViT model
            inputs = vit_processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                vit_outputs = vit_model(**inputs)
            vit_features = vit_outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

            if image_name in existing_images:
                combined_text = df[df["image_path"] == image_name]["ocr_text"].values[0]
            else:
                combined_text = ""

            if combined_text.strip():
                # Use Jina tokenizer and model for text processing
                text_inputs = text_tokenizer(
                    combined_text,
                    return_tensors="pt", 
                    padding="longest",
                    truncation=True, 
                    max_length=512
                ).to(device)

                with torch.no_grad():
                    jina_outputs = text_encoder(**text_inputs)

                # Extract Jina features
                jina_features = jina_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                combined_features = np.concatenate([vit_features, jina_features])
            else:
                combined_features = np.concatenate([vit_features, np.zeros(text_encoder.config.hidden_size)])

            image_features.append(combined_features)

        except Exception as e:
            print(f"\nError processing image {image_name}: {str(e)}")
            image_features.append(np.zeros(vit_model.config.hidden_size + text_encoder.config.hidden_size))

    text_features = []
    total_texts = len(texts)
    for i, text in enumerate(texts, 1):
        try:
            # Use Jina tokenizer and model for text processing
            inputs = text_tokenizer(
                text, 
                return_tensors="pt", 
                padding="longest",
                truncation=True, 
                max_length=512
            ).to(device)

            with torch.no_grad():
                jina_outputs = text_encoder(**inputs)

            jina_feature = jina_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            text_features.append(jina_feature)

        except Exception as e:
            print(f"\nError processing text: {str(e)}")
            text_features.append(np.zeros(text_encoder.config.hidden_size))

    return np.array(image_features), np.array(text_features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from image and text data.")
    parser.add_argument("--data_path", required=True, help="Path to the JSON data file.")
    parser.add_argument("--image_folder", required=True, help="Path to the folder containing images.")
    parser.add_argument("--ocr_cache_path", required=True, help="Path to the OCR cache file.")
    parser.add_argument("--output_dir", required=True, help="Path to the directory where features will be saved.")
    parser.add_argument("--mode", default="train", choices=["train", "test"], help="Mode: 'train' or 'test'.")

    args = parser.parse_args()

    extract_and_save_features(
        data_path=args.data_path,
        image_folder=args.image_folder,
        ocr_cache_path=args.ocr_cache_path,
        output_dir=args.output_dir,
        mode=args.mode
    )