# # run_test.py
import logging
import torch
import os
import json
from process_datasets import TestSarcasmDataset
from torch.utils.data import DataLoader
from sarcasm_models import VietnameseSarcasmClassifier
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


def run_test(test_json, test_image_folder, tokenizer, device, 
                            batch_size, num_workers, 
                            test_ocr_cache_path, model_paths, 
                            text_encoder, image_encoder, fusion_method, use_test_ocr_cache=False, active_ocr=True):
    if model_paths is None:
        logging.error("No model paths were provided, using default.")
    else:
        logging.info(f"Model paths received: {model_paths}")
    logging.info("Starting testing with multiple models...")

    # Create test dataset with OCR caching parameters
    test_dataset = TestSarcasmDataset(
        data=test_json, 
        image_folder=test_image_folder, 
        text_tokenizer=tokenizer, 
        use_ocr_cache=use_test_ocr_cache, 
        ocr_cache_path=test_ocr_cache_path,
        active_ocr=active_ocr
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
    except Exception as e:
        logging.error(f"Failed to initialize the model: {e}")
        return
    
    # Iterate over each model path
    for idx, model_path in enumerate(model_paths):
        # Load trained model weights
        if not os.path.isfile(model_path):
            logging.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info(f"Model loaded from {model_path}")
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
            "phase": "dev"
        }
        
        # Save predictions to JSON for each model
        output_filename = f'results_model_{idx + 1}.json'
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            logging.info(f"Predictions saved to {output_filename} for model {idx+1}")
        except Exception as e:
            logging.error(f"Failed to save predictions for model {idx + 1}: {e}")
        
        # Save OCR cache explicitly
        try:
            test_dataset.save_ocr_cache()
        except Exception as e:
            logging.error(f"Failed to save OCR cache for model {idx + 1}: {e}")