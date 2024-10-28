# utils.py
import torch.nn as nn
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import logging

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq=d_out_kq
        self.W_query=nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key=nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value=nn.Parameter(torch.rand(d_in, d_out_v))
        
    def forward(self, x):
        keys=x.matmul(self.W_key)
        queries=x.matmul(self.W_query)
        values=x.matmul(self.W_value)
        
        # unnormalized attention weights
        attn_scores=queries.matmul(keys.T)
        
        attn_weights=torch.softmax(
            attn_scores/self.d_out_kq**0.5, dim=-1
        )
        
        context_vex=attn_weights.matmul(values)
        return context_vex

class CrossAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq=d_out_kq
        self.W_query=nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key  = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value=nn.Parameter(torch.rand(d_in, d_out_v))
    
    def forward(self, x_1, x_2):
        queries_1=x_1.matmul(self.W_query)
        keys_2=x_2.matmul(self.W_key)
        values_2=x_2.matmul(self.W_value)
        
        attn_scores=queries_1.matmul(keys_2.T)
        attn_weights=torch.softmax(
            attn_scores/self.d_out_kq**0.5, dim=-1
        )
        
        context_vec=attn_weights.matmul(values_2)
        return context_vec

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
    logging.info("\n----Class-wise Metrics----")
    for i, label in enumerate(labels):
        y_true = [1 if l == i else 0 for l in all_labels]
        y_pred = [1 if p == i else 0 for p in all_preds]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        logging.info(f"{label}: precision: {precision:.4f}, recall: {recall:.4f}, f1 score: {f1:.4f}")
    
    # Calculate overall metrics
    overall_acc = accuracy_score(all_labels, all_preds)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Log overall metrics
    logging.info("\n ----OVERALL----")
    average_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    logging.info(f"Val Loss: {average_loss:.4f}")
    logging.info(f"Overall Accuracy: {overall_acc:.4f}")
    logging.info(f"Overall Precision: {overall_precision:.4f}")
    logging.info(f"Overall Recall: {overall_recall:.4f}")
    logging.info(f"OVERALL F1 SCORE: {overall_f1:.4f}")
    
    return overall_f1