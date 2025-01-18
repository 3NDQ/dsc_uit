# main.py 
import os
import torch
import argparse
import sys
import logging
from model_factory import get_text_encoder, get_image_encoder, get_tokenizer, get_image_processor  
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
    
    # Encoder arguments (still needed for model initialization)
    parser.add_argument('--text_encoder', type=str, default="jinaai/jina-embeddings-v2-base-en", help='Name/path of the text encoder model')
    parser.add_argument('--image_encoder', type=str, default="google/vit-base-patch16-224-in21k", help='Name/path of the image encoder model')
    
    # Data paths for feature extraction (used in extract_features.py)
    parser.add_argument('--train_json', type=str, default='/kaggle/input/vimmsd-training-dataset/vimmsd-train.json', help='Path to the training JSON file')
    parser.add_argument('--train_image_folder', type=str, default='/kaggle/input/vimmsd-training-dataset/training-images/train-images', help='Path to the training images folder')
    parser.add_argument('--test_json', type=str, default='/kaggle/input/vimmsd-public-test/vimmsd-public-test.json', help='Path to the testing JSON file')
    parser.add_argument('--test_image_folder', type=str, default='/kaggle/input/vimmsd-public-test/public-test-images/dev-images', help='Path to the testing images folder')

    # Paths to pre-extracted features
    parser.add_argument('--train_features_dir', type=str, default='train_features', help='Directory containing pre-extracted training features')
    parser.add_argument('--test_features_dir', type=str, default='test_features', help='Directory containing pre-extracted testing features')
    
    # Model paths for testing
    parser.add_argument('--model_paths', type=str, nargs='+', default=['model_epoch_1.pth'], help='Paths to trained models')
    
    # Common arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    
    # OCR Caching arguments (used in extract_features.py)
    parser.add_argument('--use_train_ocr_cache', action='store_true', help='Enable OCR caching for training')
    parser.add_argument('--train_ocr_cache_path', type=str, default='train_ocr_cache.json', help='Path to store or load train OCR cache')
    parser.add_argument('--use_test_ocr_cache', action='store_true', help='Enable OCR caching for testing')
    parser.add_argument('--test_ocr_cache_path', type=str, default='test_ocr_cache.json', help='Path to store or load test OCR cache')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate for the optimizer')
    parser.add_argument('--val_size', type=float, default=0.2, help='Val size for train test split')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'attention', 'cross_attention'], help='Method to fuse features: concat (default) or attention, cross_attention')

    args = parser.parse_args()

    # Initialize text and image encoders using factory functions
    try:
        text_encoder = get_text_encoder(args.text_encoder)
        image_encoder = get_image_encoder(args.image_encoder)
    except Exception:
        logging.error("Encoder initialization failed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if args.mode == 'train':
        run_train(
            train_features_dir=args.train_features_dir,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fusion_method=args.fusion_method,
            num_epochs=args.num_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            val_size=args.val_size,
            random_state=args.random_state,
        )
    elif args.mode == 'test':
        run_test(
            test_features_dir=args.test_features_dir,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_paths=args.model_paths,
            fusion_method=args.fusion_method
        )

if __name__ == "__main__":
    main()