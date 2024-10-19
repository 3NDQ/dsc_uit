# main.py (Updated Sections)

import os
import json
import torch
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
from datasets import SarcasmDataset, TestSarcasmDataset
from sarcasm_models import VietnameseSarcasmClassifier
from train import train_model
from test import test_model
from utils import evaluate_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import sys
import logging
from model_factory import get_text_encoder, get_image_encoder, get_tokenizer  # Import factory functions

# ... (Rest of the imports and existing code)

def main():
    # Configure logging
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s [%(levelname)s] %(message)s',
    #     handlers=[
    #         logging.StreamHandler(),
    #         logging.FileHandler("sarcasm_classifier.log")  # Optional: Log to a file
    #     ]
    # )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Explicitly use sys.stdout
            logging.FileHandler("sarcasm_classifier.log")  # Optional: Log to a file
        ],
        force=True
    )
    parser = argparse.ArgumentParser(description="Vietnamese Sarcasm Classifier")
    
    # Mode: train or test
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
    
    # Encoder and tokenizer arguments
    parser.add_argument('--text_encoder', type=str, default="vinai/phobert-base-v2", help='Name/path of the text encoder model')
    parser.add_argument('--image_encoder', type=str, default="google/vit-base-patch16-224", help='Name/path of the image encoder model')
    parser.add_argument('--tokenizer', type=str, default="vinai/phobert-base-v2", help='Name/path of the tokenizer')
    
    # Training arguments
    parser.add_argument('--train_json', type=str, default='/kaggle/input/vimmsd-training-dataset/vimmsd-train.json', help='Path to the training JSON file')
    parser.add_argument('--train_image_folder', type=str, default='/kaggle/input/vimmsd-training-dataset/training-images/train-images', help='Path to the training images folder')
    
    # Testing arguments
    parser.add_argument('--test_json', type=str, default='/kaggle/input/vimmsd-public-test/vimmsd-public-test.json', help='Path to the testing JSON file')
    parser.add_argument('--test_image_folder', type=str, default='/kaggle/input/vimmsd-public-test/public-test-images/dev-images', help='Path to the testing images folder')
    parser.add_argument('--model_path', type=str, default='sarcasm_classifier_model.pth', help='Path to trained model')
    
    # Common arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    
    # OCR Training Caching arguments
    parser.add_argument('--use_train_ocr_cache', action='store_true', help='Enable OCR caching for training')
    parser.add_argument('--train_ocr_cache_path', type=str, default='train_ocr_cache.json', help='Path to store or load train OCR cache')
    
    # OCR Testing Caching arguments
    parser.add_argument('--use_test_ocr_cache', action='store_true', help='Enable OCR caching for testing')
    parser.add_argument('--test_ocr_cache_path', type=str, default='test_ocr_cache.json', help='Path to store or load test OCR cache')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    args = parser.parse_args()
    
    # Validate paths
    if args.mode == 'train':
        if not os.path.isfile(args.train_json):
            parser.error(f"Training JSON file not found at {args.train_json}")
        if not os.path.isdir(args.train_image_folder):
            parser.error(f"Training image folder not found at {args.train_image_folder}")
    elif args.mode == 'test':
        if not os.path.isfile(args.test_json):
            parser.error(f"Testing JSON file not found at {args.test_json}")
        if not os.path.isdir(args.test_image_folder):
            parser.error(f"Testing image folder not found at {args.test_image_folder}")
        if not os.path.isfile(args.model_path):
            parser.error(f"Model file not found at {args.model_path}")
    
    # Initialize tokenizer using factory function
    try:
        tokenizer = get_tokenizer(args.tokenizer)
    except Exception:
        logging.error("Tokenizer initialization failed.")
        return
    
    # Initialize text and image encoders using factory functions
    try:
        text_encoder = get_text_encoder(args.text_encoder)
        image_encoder = get_image_encoder(args.image_encoder)
    except Exception:
        logging.error("Encoder initialization failed.")
        return
    
    # Add special tokens to tokenizer
    special_tokens = {"additional_special_tokens": ["[OCR]", "[CAPTION]"]}
    tokenizer.add_special_tokens(special_tokens)
    logging.info("Tokenizer special tokens added.")
    
    # Resize token embeddings to accommodate new tokens
    try:
        text_encoder.resize_token_embeddings(len(tokenizer))
        logging.info("Token embeddings resized to accommodate new tokens.")
    except Exception as e:
        logging.error(f"Failed to resize token embeddings: {e}")
        return
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    if args.mode == 'train':
        train_and_evaluate(
            train_json=args.train_json,
            train_image_folder=args.train_image_folder,
            tokenizer=tokenizer,
            device=device,
            num_epochs=args.num_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_train_ocr_cache=args.use_train_ocr_cache,
            train_ocr_cache_path=args.train_ocr_cache_path,
            text_encoder=text_encoder,
            image_encoder=image_encoder  # Pass encoders as arguments
        )
    elif args.mode == 'test':
        run_test(
            test_json=args.test_json,
            test_image_folder=args.test_image_folder,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_test_ocr_cache=args.use_test_ocr_cache,
            test_ocr_cache_path=args.test_ocr_cache_path,
            model_path=args.model_path,
            text_encoder=text_encoder,
            image_encoder=image_encoder  # Pass encoders as arguments
        )

# main.py (Within the same file, after main function or elsewhere)

def train_and_evaluate(train_json, train_image_folder, tokenizer, device, 
                      num_epochs=50, patience=10, batch_size=16, num_workers=4,
                      use_train_ocr_cache=False, train_ocr_cache_path='train_ocr_cache.json',
                      text_encoder=None, image_encoder=None):
    logging.info("Starting training and evaluation...")
    
    # Load JSON data
    try:
        with open(train_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Training data loaded from {train_json}")
    except Exception as e:
        logging.error(f"Failed to load training data from {train_json}: {e}")
        return
    
    # Create dataset with OCR caching parameters
    dataset = SarcasmDataset(
        data=data, 
        image_folder=train_image_folder, 
        text_tokenizer=tokenizer, 
        use_ocr_cache=use_train_ocr_cache, 
        ocr_cache_path=train_ocr_cache_path
    )
    
    # Extract labels for stratified splitting
    try:
        labels = [dataset[i]['labels'].item() for i in range(len(dataset))]
    except Exception as e:
        logging.error(f"Failed to extract labels for stratified splitting: {e}")
        return
    
    # Split data into training and validation sets
    try:
        train_idx, val_idx = train_test_split(
            range(len(labels)), 
            test_size=0.2, 
            stratify=labels, 
            random_state=42
        )
        logging.info('Finished splitting train/dev indices')
    except Exception as e:
        logging.error(f"Failed to split data into train/dev sets: {e}")
        return
    
    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    logging.info('Finished creating train/dev sets')
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    logging.info('Finished loading DataLoaders')
    
    # Initialize model with passed encoders
    try:
        model = VietnameseSarcasmClassifier(text_encoder, image_encoder, num_labels=4).to(device)
        logging.info('Model initialized and moved to device')
    except Exception as e:
        logging.error(f"Failed to initialize the model: {e}")
        return
    
    # Train the model
    model = train_model(
        model, 
        train_dataloader, 
        val_dataloader, 
        device, 
        num_epochs=num_epochs, 
        patience=patience
    )
    logging.info('Model training complete')
    
    # Save the trained model
    try:
        torch.save(model.state_dict(), 'sarcasm_classifier_model.pth')
        logging.info('Model saved as sarcasm_classifier_model.pth')
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
    
    # Evaluate the model on validation set
    try:
        metrics = evaluate_model(model, val_dataloader, device)
    except Exception as e:
        logging.error(f"Failed to evaluate model: {e}")
        return
    
    # Print evaluation metrics
    logging.info("Validation Metrics:")
    logging.info(f"Overall F1 Score: {metrics['overall_f1']:.4f}")
    logging.info(f"Overall Precision: {metrics['overall_precision']:.4f}")
    logging.info(f"Overall Recall: {metrics['overall_recall']:.4f}")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
    
    logging.info("\nClass-wise Metrics:")
    for class_name, class_metrics in metrics['class_metrics'].items():
        logging.info(f"{class_name}:")
        logging.info(f"  Precision: {class_metrics['precision']:.4f}")
        logging.info(f"  Recall: {class_metrics['recall']:.4f}")
        logging.info(f"  F1 Score: {class_metrics['f1']:.4f}")
    
    # Save OCR cache explicitly
    try:
        dataset.save_ocr_cache()
    except Exception as e:
        logging.error(f"Failed to save OCR cache: {e}")

def run_test(test_json, test_image_folder, tokenizer, device, 
             batch_size=16, num_workers=4, use_test_ocr_cache=True, test_ocr_cache_path='test_ocr_cache.json', model_path='sarcasm_classifier_model.pth',
             text_encoder=None, image_encoder=None):
    logging.info("Starting testing...")
    
    # Create test dataset with OCR caching parameters
    test_dataset = TestSarcasmDataset(
        json_data_path=test_json, 
        image_folder=test_image_folder, 
        text_tokenizer=tokenizer, 
        use_ocr_cache=use_test_ocr_cache, 
        ocr_cache_path=test_ocr_cache_path
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
        model = VietnameseSarcasmClassifier(text_encoder, image_encoder, num_labels=4).to(device)
    except Exception as e:
        logging.error(f"Failed to initialize the model: {e}")
        return
    
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
    logging.info('Model set to evaluation mode')
    
    # Generate predictions
    try:
        predictions = test_model(model, test_dataset, device, dataloader=test_dataloader)
        logging.info("Predictions generated successfully.")
    except Exception as e:
        logging.error(f"Failed to generate predictions: {e}")
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
        logging.error(f"Failed to map predictions to test data keys: {e}")
        return
    
    output = {
        "results": results,
        "phase": "dev"
    }
    
    # Save predictions to JSON
    try:
        with open('results.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logging.info("Predictions saved to results.json")
    except Exception as e:
        logging.error(f"Failed to save predictions to results.json: {e}")
    
    # Save OCR cache explicitly
    try:
        test_dataset.save_ocr_cache()
    except Exception as e:
        logging.error(f"Failed to save OCR cache: {e}")

if __name__ == "__main__":
    main()