# process_data.py
import os
import json
import logging
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import easyocr
import torch
        
class BaseSarcasmDataset(Dataset):
    def __init__(self, data_path, image_folder, text_tokenizer, 
                 use_ocr_cache=False, active_ocr=True, ocr_cache_path=None, max_length=256, is_test=False):
        self.image_folder = image_folder
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.use_ocr_cache = use_ocr_cache
        self.ocr_cache_path = ocr_cache_path
        self.active_ocr = active_ocr
        self.is_test = is_test  # Thêm tham số is_test để chỉ định tập kiểm thử
        self.ocr_cache = self._load_ocr_cache()
        self.ocr_reader = easyocr.Reader(['vi', 'en'], gpu=True)
        self.data = self._load_data(data_path)

        # Image transformation
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logging.info(f"{self.__class__.__name__} initialized.")

    def _load_data(self, data_path):
        if isinstance(data_path, str) and os.path.isfile(data_path):
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = list(json.load(f).values())
                logging.info(f"Data loaded from {data_path}")
            except Exception as e:
                logging.error(f"Failed to load data from {data_path}: {e}")
                data = []
        else:
            logging.error("Provided data_path is not a valid path to a JSON file.")
            data = []
        return data

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'])

        # Perform OCR
        raw_ocr = ""
        image = self._load_image(image_path)
        combined_text = self._get_combined_text(item['caption'], raw_ocr)
        encoded_text = self.text_tokenizer(
            combined_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )

        # Nếu là tập kiểm thử, không cần trả về 'labels'
        if self.is_test:
            return {
                'image': image,
                'input_ids': encoded_text['input_ids'].squeeze(),
                'attention_mask': encoded_text['attention_mask'].squeeze()
            }
        else:
            # Trả về 'labels' nếu là tập huấn luyện
            return {
                'image': image,
                'input_ids': encoded_text['input_ids'].squeeze(),
                'attention_mask': encoded_text['attention_mask'].squeeze(),
                'labels': torch.tensor(item.get('label_id', -1), dtype=torch.long) if 'label_id' in item else None
            }

class TrainSarcasmDataset(BaseSarcasmDataset):
    def _load_data(self, data_path):
        data = super()._load_data(data_path)
        label_to_id = {
            'multi-sarcasm': 0, 
            'text-sarcasm': 1, 
            'image-sarcasm': 2, 
            'not-sarcasm': 3,
        }
        for item in data:
            if isinstance(item, dict) and 'label' in item:
                item['label_id'] = label_to_id.get(item['label'], 3)
            else:
                logging.warning("Skipping an item due to unexpected structure.")
        return data

class TestSarcasmDataset(BaseSarcasmDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item['image'])

        # Perform OCR (nếu cần)
        raw_ocr = self.ocr_cache.get(image_path, self._perform_ocr(image_path) if not self.use_ocr_cache else "")
        if self.use_ocr_cache:
            self.ocr_cache[image_path] = raw_ocr
        # raw_ocr = ""
        image = self._load_image(image_path)
        combined_text = self._get_combined_text(item['caption'], raw_ocr)
        encoded_text = self.text_tokenizer(
            combined_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )

        # Trả về 'image', 'input_ids', và 'attention_mask' mà không có 'labels'
        return {
            'image': image,
            'input_ids': encoded_text['input_ids'].squeeze(),
            'attention_mask': encoded_text['attention_mask'].squeeze()
        }
