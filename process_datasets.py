import os
import json
import logging
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import easyocr
import torch

class BaseSarcasmDataset(Dataset):
    def __init__(self, data_path, image_folder, text_tokenizer, ocr_cache_path=None, max_length=256):
        self.image_folder = image_folder
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.ocr_cache_path = ocr_cache_path
        self.ocr_cache = self._load_ocr_cache()
        self.data = self._load_data(data_path)

        # Image transformation
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logging.info(f"{self.__class__.__name__} initialized.")

    def _load_ocr_cache(self):
        if self.ocr_cache_path and os.path.exists(self.ocr_cache_path):
            try:
                with open(self.ocr_cache_path, 'r', encoding='utf-8') as f:
                    logging.info(f"OCR cache loaded from \"{self.ocr_cache_path}\"")
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load OCR cache from \"{self.ocr_cache_path}: {e}\"")
        return {}

    def _load_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            logging.error(f"Image loading failed for {image_path}: {e}")
            return torch.zeros(3, 224, 224)

    def _get_combined_text(self, caption, ocr_text=""):
        if self.ocr_cache_path:
            return f"[CAPTION] {caption.lower()} [OCR] {ocr_text}"
        return caption.lower()

    def _load_data(self, data_path):
        if isinstance(data_path, str) and os.path.isfile(data_path):
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = list(json.load(f).values())
                logging.info(f"Data loaded from \"{data_path}\"")
            except Exception as e:
                logging.error(f"Failed to load data from \"{data_path}\": {e}")
                data = []
        else:
            logging.error("Provided data_path is not a valid path to a JSON file.")
            data = []
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'])

        raw_ocr = self.ocr_cache.get(image_path) if self.ocr_cache_path else ""
        if self.ocr_cache_path:
            self.ocr_cache[image_path] = raw_ocr

        # Process Image and Text
        image = self._load_image(image_path)
        combined_text = self._get_combined_text(item['caption'], raw_ocr)
        encoded_text = self.text_tokenizer(
            combined_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        output = {
            'image': image,
            'input_ids': encoded_text['input_ids'].squeeze(),
            'attention_mask': encoded_text['attention_mask'].squeeze()
        }

        if 'label' in item and item['label'] is not None and hasattr(self, 'label_to_id'):
            output['labels'] = torch.tensor(self.label_to_id.get(item['label'], 3), dtype=torch.long)
        elif 'label_id' in item and item['label_id'] is not None:
            output['labels'] = torch.tensor(item['label_id'], dtype=torch.long)

        return output

class TrainSarcasmDataset(BaseSarcasmDataset):
    def __init__(self, data_path, image_folder, text_tokenizer, ocr_cache_path=None, max_length=256):
        self.label_to_id = {
            'multi-sarcasm': 0,
            'text-sarcasm': 1,
            'image-sarcasm': 2,
            'not-sarcasm': 3,
        }
        super().__init__(data_path, image_folder, text_tokenizer, ocr_cache_path, max_length)
        self._process_labels()

    def _process_labels(self):
        for item in self.data:
            if isinstance(item, dict) and 'label' in item:
                item['label_id'] = self.label_to_id.get(item['label'], 3)
            else:
                logging.warning("Skipping an item due to unexpected structure.")

class TestSarcasmDataset(BaseSarcasmDataset):
    def __init__(self, data_path, image_folder, text_tokenizer, ocr_cache_path=None, max_length=256):
        super().__init__(data_path, image_folder, text_tokenizer, ocr_cache_path, max_length)