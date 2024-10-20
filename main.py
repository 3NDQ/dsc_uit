# main.py (Updated Sections)

import os
import torch
import argparse
import sys
import logging
from model_factory import get_text_encoder, get_image_encoder, get_tokenizer  
from train_and_evaluate import train_and_evaluate  
from run_test_multiple_models import run_test_multiple_models 

# ... (Rest of the imports and existing code)

def main():
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
    parser.add_argument('--model_paths', type=str, nargs='+', default=['model_epoch_1.pth', 'model_epoch_2.pth', 'model_epoch_3.pth', 'model_epoch_4.pth', 'model_epoch_5.pth'], required=False, help='Paths to trained models')
    
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
        for model_path in args.model_paths:
            if not os.path.isfile(model_path):
                parser.error(f"Model file not found at {model_path}")

    
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
            image_encoder=image_encoder 
        )
   
    elif args.mode == 'test':
        run_test_multiple_models(
            test_json=args.test_json,
            test_image_folder=args.test_image_folder,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_test_ocr_cache=args.use_test_ocr_cache,
            test_ocr_cache_path=args.test_ocr_cache_path,
            model_paths=args.model_paths,  
            text_encoder=text_encoder,
            image_encoder=image_encoder
        )

if __name__ == "__main__":
    main()