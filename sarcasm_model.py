import torch
import torch.nn as nn
import logging
from utils import CrossAttention, SelfAttention
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
import os
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import json

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)  # Probabilities of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class VietnameseSarcasmClassifier(nn.Module):
    def __init__(self,
                 mode,
                 text_encoder,
                 text_tokenizer,
                 image_encoder,
                 train_image_folder,
                 test_image_folder,
                 train_ocr_cache_path,
                 test_ocr_cache_path,
                 image_processor=None,
                 fusion_method='concat',
                 num_labels=4):
        super(VietnameseSarcasmClassifier, self).__init__()
        self.num_labels = num_labels
        self.mode = mode
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.train_path = train_image_folder
        self.test_path = test_image_folder
        self.fusion_method = fusion_method
        self.train_ocr_cache_path = train_ocr_cache_path
        self.test_ocr_cache_path = test_ocr_cache_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize ViT model and processor
        self.vit_processor = image_processor
        self.vit_model = image_encoder

        # Initialize Jina model and tokenizer
        self.jina_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en", model_max_length=512)
        self.text_encoder = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en").to(self.device)

        combined_dim = self.image_encoder.config.hidden_size + self.text_encoder.config.hidden_size
        logging.info(f"Combined dimension: {combined_dim}")
        
        if self.fusion_method == 'cross_attention':
            hidden_size = self.image_encoder.config.hidden_size
            self.text_to_image_attention = CrossAttention(d_in=hidden_size, d_out_kq=hidden_size, d_out_v=hidden_size)
            self.image_to_text_attention = CrossAttention(d_in=hidden_size, d_out_kq=hidden_size, d_out_v=hidden_size)
            logging.info("Cross-Attention layers initialized for both text-to-image and image-to-text.")
        
        elif self.fusion_method == 'attention':
            self.self_attention = SelfAttention(d_in=combined_dim, d_out_kq=combined_dim, d_out_v=combined_dim)
            logging.info("Self-Attention layer initialized for feature fusion.")
        
        self.projector = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        logging.info("Projector layers initialized.")
        
        # Classification heads
        self.text_classifier = nn.Linear(512, 2)
        self.image_classifier = nn.Linear(512, 2)
        self.multi_classifier = nn.Linear(512, 2)
        logging.info("Classification heads initialized.")        

    def preprocess_data(self, images, texts, mode='train'):
        train_path = "/kaggle/input/vimmsd/train-images"
        test_path = "/kaggle/input/vimmsd/test-images"
        image_features = []
        ocr_features = []
        total_images = len(images)

        input_json_file_path = self.test_ocr_cache_path if mode == 'test' else self.test_ocr_cache_path

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

        print("\nProcessing images:")
        for i, image_name in enumerate(images, 1):
            try:
                print(f"Processing image {i}/{total_images}", end='\r')
                image_path = os.path.join(train_path if not is_test else test_path, image_name)
                img = cv2.imread(image_path)

                # Process the image using ViT model
                inputs = self.vit_processor(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    vit_outputs = self.vit_model(**inputs)
                vit_features = vit_outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

                if image_name in existing_images:
                    combined_text = df[df["image_name"] == image_name]["combined_text"].values[0]
                else:
                    combined_text = ""

                if combined_text.strip():
                    # Use Jina tokenizer and model for text processing
                    text_inputs = self.jina_tokenizer(
                        combined_text,
                        return_tensors="pt", 
                        padding="longest",
                        truncation=True, 
                        max_length=512
                    ).to(self.device)

                    with torch.no_grad():
                        jina_outputs = self.jina_model(**text_inputs)

                    # Extract Jina features - it returns 1024-dimensional embeddings
                    jina_features = jina_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    combined_features = np.concatenate([vit_features, jina_features])
                else:
                    combined_features = np.concatenate([vit_features, np.zeros(self.jina_model.config.hidden_size)])

                image_features.append(combined_features)

            except Exception as e:
                print(f"\nError processing image {image_name}: {str(e)}")
                image_features.append(np.zeros(self.vit_model.config.hidden_size + self.jina_model.config.hidden_size))

        print("\nProcessing texts:")
        text_features = []
        total_texts = len(texts)
        for i, text in enumerate(texts, 1):
            try:
                print(f"Processing text {i}/{total_texts}", end='\r')

                # Use Jina tokenizer and model for text processing
                inputs = self.jina_tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding="longest",
                    truncation=True, 
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    jina_outputs = self.jina_model(**inputs)

                # Extract Jina features (1024-dimensional)
                jina_feature = jina_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                text_features.append(jina_feature)

            except Exception as e:
                print(f"\nError processing text: {str(e)}")
                text_features.append(np.zeros(self.jina_model.config.hidden_size))

        print("\nPreprocessing completed!")
        return np.array(image_features), np.array(text_features)

    def forward(self, image, caption, labels=None, is_test=False):
        logging.debug("Forward pass started.")

        # Preprocess data
        image_features, text_features = self.preprocess_data(image, caption, is_test=is_test)
        image_features = torch.tensor(image_features, dtype=torch.float).to(self.device)
        text_features = torch.tensor(text_features, dtype=torch.float).to(self.device)

        # Combine features based on fusion method
        if self.fusion_method == 'cross_attention':
            attended_text = self.text_to_image_attention(text_features, image_features)
            attended_image = self.image_to_text_attention(image_features, text_features)
            combined_features = torch.cat((attended_text, attended_image), dim=1)
        elif self.fusion_method == 'attention':
            combined_features = torch.cat((image_features, text_features), dim=1)
            attended_features = self.self_attention(combined_features)
            combined_features = attended_features
        else:
            combined_features = torch.cat((image_features, text_features), dim=1)

        # Project combined features
        shared_features = self.projector(combined_features)

        # Classification heads
        text_logits = self.text_classifier(shared_features)
        image_logits = self.image_classifier(shared_features)
        multi_logits = self.multi_classifier(shared_features)

        # Final logits
        final_logits = torch.zeros((shared_features.size(0), 4), device=shared_features.device)
        final_logits[:, 0] = multi_logits[:, 1]
        final_logits[:, 1] = text_logits[:, 1]
        final_logits[:, 2] = image_logits[:, 1]
        final_logits[:, 3] = 1 - (multi_logits[:, 1] + text_logits[:, 1] + image_logits[:, 1]).clamp(0, 1)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            criterion = FocalLoss()
            loss = criterion(final_logits, labels)

        return {'loss': loss, 'logits': final_logits} if loss is not None else {'logits': final_logits}