# # train_and_evaluate.py
import logging
import json
import torch
from torch.utils.data import DataLoader, Subset
from utils import evaluate_model, train_model
from datasets import SarcasmDataset
from sarcasm_models import VietnameseSarcasmClassifier
from sklearn.model_selection import train_test_split

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
    
    # # Save the trained model
    # try:
    #     torch.save(model.state_dict(), 'sarcasm_classifier_model.pth')
    #     logging.info('Model saved as sarcasm_classifier_model.pth')
    # except Exception as e:
    #     logging.error(f"Failed to save model: {e}")
    
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