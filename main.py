# main.py 
import os
import torch
import argparse
import sys
import logging
from model_factory import get_text_encoder, get_image_encoder, get_tokenizer  
from run_train import run_train
from run_test import run_test

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  
            logging.FileHandler("sarcasm_classifier.log")  
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
    parser.add_argument('--model_paths', type=str, nargs='+', default=['model_epoch_1.pth'], help='Paths to trained models')
    
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
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate for the optimizer')
    parser.add_argument('--val_size', type=float, default=0.2, help='Val size for train test split')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'attention', 'cross_attention'], help='Method to fuse features: concat (default) or attention')
    parser.add_argument('--active_ocr', action='store_true', help='Enable combining OCR and text')

    
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
        if not args.model_paths:
            parser.error("No model paths provided for testing.")
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
    if args.active_ocr:
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
        run_train(
            train_json=args.train_json,
            train_image_folder=args.train_image_folder,
            active_ocr=args.active_ocr,
            use_train_ocr_cache=args.use_train_ocr_cache,
            train_ocr_cache_path=args.train_ocr_cache_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fusion_method =args.fusion_method,
            num_epochs=args.num_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            val_size=args.val_size,
            random_state=args.random_state,
        )
   
    elif args.mode == 'test':
        run_test(
            test_json=args.test_json,
            test_image_folder=args.test_image_folder,
            active_ocr=args.active_ocr,
            use_test_ocr_cache=args.use_test_ocr_cache,
            test_ocr_cache_path=args.test_ocr_cache_path, 
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_paths=args.model_paths,
            fusion_method = args.fusion_method
        )

if __name__ == "__main__":
    main()