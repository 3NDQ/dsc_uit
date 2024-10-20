import os
import json
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import easyocr

class BaseSarcasmDataset(Dataset):
    def __init__(self, image_folder, text_tokenizer, max_length=256, use_ocr_cache=False, ocr_cache_path='ocr_cache.json'):
        self.image_folder = image_folder
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.use_ocr_cache = use_ocr_cache
        self.ocr_cache_path = ocr_cache_path

        # Label mapping (this can be overridden in subclasses)
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

        # Initialize OCR reader once with GPU disabled
        self.reader = easyocr.Reader(['vi', 'en'], gpu=False)

        # Initialize or load OCR cache
        self.ocr_cache = self._load_ocr_cache() if use_ocr_cache else {}

    def _load_ocr_cache(self):
        """Load OCR cache from the file if it exists."""
        if os.path.exists(self.ocr_cache_path):
            try:
                with open(self.ocr_cache_path, 'r', encoding='utf-8') as f:
                    logging.info(f"OCR cache loaded from {self.ocr_cache_path}")
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load OCR cache: {e}")
        logging.info("OCR cache file not found or failed to load. Initializing empty cache.")
        return {}

    def _perform_ocr(self, image_path):
        """Perform OCR using EasyOCR or return an empty string if it fails."""
        if image_path in self.ocr_cache:
            return self.ocr_cache[image_path]
        
        try:
            logging.debug(f"Performing OCR for {image_path}")
            ocr_results = self.reader.readtext(image_path, detail=0)
            raw_ocr = ' '.join(ocr_results).lower()
            self.ocr_cache[image_path] = raw_ocr
            return raw_ocr
        except Exception as e:
            logging.error(f"OCR failed for image {image_path}: {e}")
            return ""

    def _load_image(self, image_path):
        """Load an image and apply the transformations. Return a zero tensor on failure."""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            logging.error(f"Failed to load image {image_path}: {e}")
            return torch.zeros(3, 224, 224)

    def save_ocr_cache(self):
        """Save the OCR cache to a file."""
        if self.use_ocr_cache:
            try:
                with open(self.ocr_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.ocr_cache, f, ensure_ascii=False, indent=2)
                logging.info(f"OCR cache saved to {self.ocr_cache_path}")
            except Exception as e:
                logging.error(f"Failed to save OCR cache: {e}")


class SarcasmDataset(BaseSarcasmDataset):
    def __init__(self, data, image_folder, text_tokenizer, max_length=256, 
                 use_ocr_cache=False, ocr_cache_path='ocr_cache.json'):
        logging.info("Initializing SarcasmDataset...")
        super().__init__(image_folder, text_tokenizer, max_length, use_ocr_cache, ocr_cache_path)
        self.data = list(data.values()) if isinstance(data, dict) else data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'])

        # Perform OCR
        raw_ocr = self._perform_ocr(image_path)
        
        # Load and process the image
        image = self._load_image(image_path)
        
        # Process the text
        caption = item['caption'].lower()
        combined_text = f"[CAPTION] {caption} [OCR] {raw_ocr}"
        encoded_text = self.text_tokenizer(
            combined_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get label, defaulting to 'not-sarcasm' if not present
        label_id = self.label_to_id.get(item['label'], 3)
        
        return {
            'image': image,
            'input_ids': encoded_text['input_ids'].squeeze(),
            'attention_mask': encoded_text['attention_mask'].squeeze(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }


class TestSarcasmDataset(BaseSarcasmDataset):
    def __init__(self, json_data_path, image_folder, text_tokenizer, max_length=256, 
                 use_ocr_cache=False, ocr_cache_path='test_ocr_cache.json'):
        logging.info("Initializing TestSarcasmDataset...")
        super().__init__(image_folder, text_tokenizer, max_length, use_ocr_cache, ocr_cache_path)
        
        # Load test JSON data
        try:
            with open(json_data_path, 'r', encoding='utf-8') as f:
                self.data = list(json.load(f).values())
            logging.info(f"Test data loaded from {json_data_path}")
        except Exception as e:
            logging.error(f"Failed to load test data from {json_data_path}: {e}")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'])

        # Perform OCR
        raw_ocr = self._perform_ocr(image_path)

        # Load and process the image
        image = self._load_image(image_path)

        # Process the text
        caption = item['caption'].lower()
        combined_text = f"[CAPTION] {caption} [OCR] {raw_ocr}"
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
