import logging
import torch
import os
import json
from torch.utils.data import DataLoader, TensorDataset
from sarcasm_model import VietnameseSarcasmClassifier
from tqdm import tqdm
import numpy as np

def test_model(model, device, dataloader):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            combined_image_features, text_features, text_features2, ocr_features, image_features = batch
            combined_image_features = combined_image_features.to(device)
            text_features = text_features.to(device)
            text_features2 = text_features2.to(device)
            ocr_features = ocr_features.to(device)
            image_features = image_features.to(device)
            
            outputs = model(
                combined_image_features=combined_image_features,
                text_features=text_features,
                text_features2=text_features2,
                ocr_features=ocr_features,
                image_features=image_features,
            )
            logits = outputs
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions

def run_test(test_features_dir, test_features_dir2, device, batch_size, num_workers, model_paths, fusion_method):
    logging.info("Starting testing with multiple models...")

    # Load pre-extracted features
    test_combined_image_features = np.load(os.path.join(test_features_dir, "combined_image_features.npy"))
    test_text_features = np.load(os.path.join(test_features_dir, "text_features.npy"))
    test_text_features2 = np.load(os.path.join(test_features_dir2, "text_features.npy"))
    test_ocr_features = np.load(os.path.join(test_features_dir, "ocr_features.npy"))
    test_image_features = np.load(os.path.join(test_features_dir, "image_features.npy"))

    # Convert to a single NumPy array
    test_combined_image_features = np.squeeze(np.array(test_combined_image_features))
    test_text_features = np.squeeze(np.array(test_text_features))
    test_text_features2 = np.squeeze(np.array(test_text_features2))
    test_ocr_features = np.squeeze(np.array(test_ocr_features))
    test_image_features = np.squeeze(np.array(test_image_features))

    # Create a TensorDataset
    test_dataset = TensorDataset(
        torch.tensor(test_combined_image_features, dtype=torch.float),
        torch.tensor(test_text_features, dtype=torch.float),
        torch.tensor(test_text_features2, dtype=torch.float),
        torch.tensor(test_ocr_features, dtype=torch.float),
        torch.tensor(test_image_features, dtype=torch.float),
    )

    # Create DataLoader
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    logging.info('Finished loading Test DataLoader')

    # Initialize model
    model = VietnameseSarcasmClassifier(
        mode="test",
        fusion_method=fusion_method,
    ).to(device)

    # Load and test each model
    for idx, model_path in enumerate(model_paths):
        if not os.path.isfile(model_path):
            logging.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Model loaded from {model_path}")
        
        model.eval()
        logging.info(f'Model set to evaluation mode - {model_path}')
        
        predictions = test_model(model, device, test_dataloader)
        logging.info(f"Predictions generated successfully for model {idx+1}")
        
        id_to_label = {0: 'multi-sarcasm', 1: 'text-sarcasm', 2: 'image-sarcasm', 3: 'not-sarcasm'}
        predicted_labels = [id_to_label.get(pred, 'not-sarcasm') for pred in predictions]
        
        test_data_keys = list(range(len(predicted_labels)))
        
        results = {key: label for key, label in zip(test_data_keys, predicted_labels)}
        
        output = {
            "results": results,
            "phase": "test"
        }
        
        output_filename = f'results_model_{idx + 1}.json'
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logging.info(f"Predictions saved to {output_filename} for model {idx+1}")