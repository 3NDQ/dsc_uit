import os
import json
import logging
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class BaseSarcasmDataset(Dataset):
    def __init__(self, data_path, image_folder, text_tokenizer, 
                 use_ocr_cache=False, active_ocr=True, ocr_cache_path=None, max_length=1024):
        self.image_folder = image_folder
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.use_ocr_cache = use_ocr_cache
        self.ocr_cache_path = ocr_cache_path
        self.active_ocr = active_ocr
        self.ocr_cache = self._load_ocr_cache()
        self.data = self._load_data(data_path)

        logging.info(f"{self.__class__.__name__} initialized.")

    def _load_ocr_cache(self):
        if self.use_ocr_cache and self.ocr_cache_path and os.path.exists(self.ocr_cache_path):
            try:
                with open(self.ocr_cache_path, 'r', encoding='utf-8') as f:
                    logging.info(f"OCR cache loaded from {self.ocr_cache_path}")
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load OCR cache from {self.ocr_cache_path}: {e}")
        return {}

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_name = item['image']

        if 'label' in item and item['label'] is not None:
            return {
                'image': image_name,
                'caption': item['caption'],
                'label': torch.tensor(item['label_id'], dtype=torch.long) if 'label_id' in item else None
            }
        else:
            return {
                'image': image_name,
                'caption': item['caption'],
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
        image_name = item['image']

        return {
            'image': image_name,
            'caption': item['caption'],
        }