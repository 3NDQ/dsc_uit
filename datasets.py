# datasets.py

import os
import json
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import easyocr
import torch
import logging

class SarcasmDataset(Dataset):
    def __init__(self, data, image_folder, text_tokenizer, max_length=256, 
                 use_ocr_cache=False, ocr_cache_path='ocr_cache.json'):
        logging.info("Initializing SarcasmDataset...")
        self.data = list(data.values()) if isinstance(data, dict) else data
        self.image_folder = image_folder
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.use_ocr_cache = use_ocr_cache
        self.ocr_cache_path = ocr_cache_path

        # Label mapping
        self.label_to_id = {
            'multi-sarcasm': 0, 
            'text-sarcasm': 1, 
            'image-sarcasm': 2, 
            'not-sarcasm': 3,
        }

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize EasyOCR reader once with GPU disabled
        self.reader = easyocr.Reader(['vi', 'en'], gpu=False)
        
        # Initialize OCR cache if enabled
        if self.use_ocr_cache:
            if os.path.exists(self.ocr_cache_path):
                try:
                    with open(self.ocr_cache_path, 'r', encoding='utf-8') as f:
                        self.ocr_cache = json.load(f)
                    logging.info(f"OCR cache loaded from {self.ocr_cache_path}")
                except Exception as e:
                    logging.error(f"Failed to load OCR cache from {self.ocr_cache_path}: {e}")
                    self.ocr_cache = {}
            else:
                self.ocr_cache = {}
                logging.info("OCR caching enabled but cache file not found. A new cache will be created.")
        else:
            self.ocr_cache = {}
        
        logging.info("SarcasmDataset initialized.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'])
        
        # OCR processing using the pre-initialized reader or cache
        raw_ocr = ""
        if self.use_ocr_cache:
            if image_path in self.ocr_cache:
                raw_ocr = self.ocr_cache[image_path]
                logging.debug(f"OCR cache hit for {image_path}")
            else:
                try:
                    ocr_results = self.reader.readtext(image_path, detail=0)
                    raw_ocr = ' '.join(ocr_results).lower()
                    self.ocr_cache[image_path] = raw_ocr  # Update cache
                    logging.debug(f"OCR processed and cached for {image_path}")
                except Exception as e:
                    logging.error(f"OCR failed for image {image_path}: {e}")
                    raw_ocr = ""
        else:
            try:
                ocr_results = self.reader.readtext(image_path, detail=0)
                raw_ocr = ' '.join(ocr_results).lower()
                logging.debug(f"OCR processed for {image_path}")
            except Exception as e:
                logging.error(f"OCR failed for image {image_path}: {e}")
                raw_ocr = ""
        
        # Image processing
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
            logging.debug(f"Image loaded and transformed for {image_path}")
        except Exception as e:
            logging.error(f"Image loading failed for {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Text processing
        caption = item['caption'].lower()
        combined_text = f"[CAPTION] {caption} [OCR] {raw_ocr}"
        encoded_text = self.text_tokenizer(
            combined_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )

        label_id = self.label_to_id.get(item['label'], 3)  # Default to 'not-sarcasm' if label missing

        return {
            'image': image,
            'input_ids': encoded_text['input_ids'].squeeze(),
            'attention_mask': encoded_text['attention_mask'].squeeze(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }
    
    def save_ocr_cache(self):
        if self.use_ocr_cache:
            try:
                with open(self.ocr_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.ocr_cache, f, ensure_ascii=False, indent=2)
                logging.info(f"OCR cache saved to {self.ocr_cache_path}")
            except Exception as e:
                logging.error(f"Failed to save OCR cache to {self.ocr_cache_path}: {e}")


class TestSarcasmDataset(SarcasmDataset):
    def __init__(self, json_data_path, image_folder, text_tokenizer, max_length=256, 
                 use_ocr_cache=False, ocr_cache_path='test_ocr_cache.json'):
        # Load test JSON data
        try:
            with open(json_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Test data loaded from {json_data_path}")
        except Exception as e:
            logging.error(f"Failed to load test data from {json_data_path}: {e}")
            data = []
        
        super().__init__(
            data=data, 
            image_folder=image_folder, 
            text_tokenizer=text_tokenizer, 
            max_length=max_length, 
            use_ocr_cache=use_ocr_cache, 
            ocr_cache_path=ocr_cache_path
        )
        # Override label_to_id since test dataset might not have labels
        self.label_to_id = {}

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'])
        
        # OCR processing using the pre-initialized reader or cache
        raw_ocr = ""
        if self.use_ocr_cache:
            if image_path in self.ocr_cache:
                raw_ocr = self.ocr_cache[image_path]
                logging.debug(f"OCR cache hit for {image_path}")
            else:
                try:
                    ocr_results = self.reader.readtext(image_path, detail=0)
                    raw_ocr = ' '.join(ocr_results).lower()
                    self.ocr_cache[image_path] = raw_ocr  # Update cache
                    logging.debug(f"OCR processed and cached for {image_path}")
                except Exception as e:
                    logging.error(f"OCR failed for image {image_path}: {e}")
                    raw_ocr = ""
        else:
            try:
                ocr_results = self.reader.readtext(image_path, detail=0)
                raw_ocr = ' '.join(ocr_results).lower()
                logging.debug(f"OCR processed for {image_path}")
            except Exception as e:
                logging.error(f"OCR failed for image {image_path}: {e}")
                raw_ocr = ""
        
        # Image processing
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
            logging.debug(f"Image loaded and transformed for {image_path}")
        except Exception as e:
            logging.error(f"Image loading failed for {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Text processing
        text = item['caption'].lower()
        combined_text = f"[CAPTION] {text} [OCR] {raw_ocr}"
        encoded_text = self.text_tokenizer(
            combined_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoded_text['input_ids'].squeeze(),
            'attention_mask': encoded_text['attention_mask'].squeeze(),
        }
