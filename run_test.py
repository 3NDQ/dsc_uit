# # run_test.py
import logging
import torch
import os
import json
import shutil
from process_datasets import TestSarcasmDataset
from torch.utils.data import DataLoader
from sarcasm_model import VietnameseSarcasmClassifier
from tqdm import tqdm

def test_model(model, device, dataloader):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    return predictions


def run_test(test_json, test_image_folder, tokenizer, 
             device, batch_size, num_workers, 
             test_ocr_cache_path, model_paths, 
             text_encoder, image_encoder, fusion_method):
    logging.info("Starting TESTING...")

    # Create test dataset with OCR caching parameters
    test_dataset = TestSarcasmDataset(
        data_path=test_json, 
        image_folder=test_image_folder, 
        text_tokenizer=tokenizer, 
        ocr_cache_path=test_ocr_cache_path,
    )
    
    # Create DataLoader
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    logging.info('Finished loading Test DataLoader')

    # Initialize model with passed encoders
    try:
        model = VietnameseSarcasmClassifier(text_encoder, image_encoder, fusion_method).to(device)
        logging.info('Model initialized and moved to device')
    except Exception as e:
        logging.error(f"Failed to initialize the model: {e}")
        return
    
    # Iterate over each model path
    for idx, model_path in enumerate(model_paths):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info(f"Model loaded from \"{model_path}\"")
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            return
        
        model.eval()
        logging.info(f'Model set to evaluation mode - {model_path}')
        
        # Generate predictions
        try:
            predictions = test_model(model, device, test_dataloader)
            logging.info(f"Predictions generated successfully for model {idx+1}")
        except Exception as e:
            logging.error(f"Failed to generate predictions for model {idx+1}: {e}")
            return
        
        # Map prediction IDs to labels
        id_to_label = {0: 'multi-sarcasm', 1: 'text-sarcasm', 2: 'image-sarcasm', 3: 'not-sarcasm'}
        predicted_labels = [id_to_label.get(pred, 'not-sarcasm') for pred in predictions]
        
        # Load test data keys
        try:
            with open(test_json, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            logging.info(f"Test data loaded from {test_json}")
        except Exception as e:
            logging.error(f"Failed to load test data from {test_json}: {e}")
            return
        
        # Prepare results
        try:
            results = {key: label for key, label in zip(test_data.keys(), predicted_labels)}
        except Exception as e:
            logging.error(f"Failed to map predictions to test data keys for model {idx+1}: {e}")
            return
        
        output = {
            "results": results,
            "phase": "test"
        }
        
        try:
            model_name = os.path.basename(model_path)
            epoch = model_name.split('_')[-1].split('.')[0]
            directory_name = f'model_epoch_{epoch}'
            
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(directory_name, exist_ok=True)
            
            json_filename = os.path.join(directory_name, 'results.json')
        except:
            directory_name = f'model_{idx + 1}'
            os.makedirs(directory_name, exist_ok=True)
            json_filename = os.path.join(directory_name, 'results.json')
            
        try:
            # Lưu kết quả vào tệp results.json
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            logging.info(f"Predictions saved to {json_filename} for model {idx+1}")
            
            # Tạo tệp nén results.zip trong cùng thư mục
            zip_filename = os.path.join(directory_name, 'results')
            shutil.make_archive(zip_filename, 'zip', directory_name, 'results.json')
            
            # Xóa tệp results.json sau khi nén
            os.remove(json_filename)
            
            logging.info(f"Results zipped into {zip_filename}.zip for model {idx+1}")
        except Exception as e:
            logging.error(f"Failed to save or zip predictions for model {idx + 1}: {e}")