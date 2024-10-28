# utils.py

import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import logging

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            logging.debug(f"EarlyStopping initialized with best_score={self.best_score}")
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logging.debug(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info("Early stopping triggered.")
        else:
            self.best_score = score
            self.counter = 0
            logging.debug(f"EarlyStopping counter reset. New best_score={self.best_score}")

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            if 'loss' in outputs and batch.get('labels') is not None:
                total_loss += outputs['loss'].item()
            
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # Define class labels
    labels = ['multi-sarcasm', 'text-sarcasm', 'image-sarcasm', 'not-sarcasm']
    
    # Calculate and log metrics for each class
    logging.info("\nClass-wise Metrics:")
    for i, label in enumerate(labels):
        y_true = [1 if l == i else 0 for l in all_labels]
        y_pred = [1 if p == i else 0 for p in all_preds]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        logging.info(f"{label}:")
        logging.info(f"  Precision: {precision:.4f}")
        logging.info(f"  Recall: {recall:.4f}")
        logging.info(f"  F1 Score: {f1:.4f}")
    
    # Calculate overall metrics
    overall_acc = accuracy_score(all_labels, all_preds)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Log overall metrics
    logging.info(f"\nOverall Accuracy: {overall_acc:.4f}")
    logging.info(f"Overall Precision: {overall_precision:.4f}")
    logging.info(f"Overall Recall: {overall_recall:.4f}")
    logging.info(f"Overall F1 Score: {overall_f1:.4f}")
    
    # Calculate and log average loss
    average_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    logging.info(f"Average Loss: {average_loss:.4f}")
    return overall_f1