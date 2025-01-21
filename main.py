import os
import torch
import argparse
import sys
import logging
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
    # Paths to pre-extracted features
    parser.add_argument('--train_features_dir', type=str, default='train_features', help='Directory containing pre-extracted training features')
    parser.add_argument('--train_features_dir2', type=str, default='train_features', help='Second directory containing pre-extracted training features')
    parser.add_argument('--test_features_dir', type=str, default='test_features', help='Directory containing pre-extracted testing features')
    parser.add_argument('--test_features_dir2', type=str, default='test_features', help='Second directory containing pre-extracted testing features')

    # Model paths for testing
    parser.add_argument('--model_paths', type=str, nargs='+', default=['model_epoch_1.pth'], help='Paths to trained models')

    # Common arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate for the optimizer')
    parser.add_argument('--val_size', type=float, default=0.2, help='Val size for train test split')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'attention', 'cross_attention'], help='Method to fuse features: concat (default) or attention, cross_attention')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['focal', 'cross_entropy'], help='Loss type: focal or cross_entropy')

    # Hyperparameters for Focal Loss and Label smoothing
    parser.add_argument('--gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss')
    parser.add_argument('--label_smoothing', type=float, default=0.15, help='Label smoothing value (e.g., 0.15)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if args.mode == 'train':
        run_train(
            train_features_dir=args.train_features_dir,
            train_features_dir2=args.train_features_dir2,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fusion_method=args.fusion_method,
            num_epochs=args.num_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            val_size=args.val_size,
            random_state=args.random_state,
            gamma=args.gamma,
            loss_type=args.loss_type,
            label_smoothing=args.label_smoothing
        )
    elif args.mode == 'test':
        run_test(
            test_features_dir=args.test_features_dir,
            test_features_dir2=args.test_features_dir2,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_paths=args.model_paths,
            fusion_method=args.fusion_method
        )

if __name__ == "__main__":
    main()