# utils.py
import torch.nn as nn
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import logging
from typing import List

import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal loss (https://arxiv.org/pdf/1708.02002.pdf)
    Shape:
        - input: (N, C) where C = number of classes
        - target: (N) where each value is 0 <= targets[i] <= C-1
        - Output: scalar. If reduction is 'none', then (N).
    Examples:
        >>> loss = FocalLoss(gamma=2, alpha=[1.0]*7)
        >>> input = torch.randn(3, 7, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(7)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, device, gamma=0, alpha: List[float] = None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.FloatTensor(alpha).to(device)  # device là cuda:0 nếu model đang ở trên GPU
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, input, target):
        target = target.unsqueeze(-1)
        pt = F.softmax(input, dim=-1)
        logpt = F.log_softmax(input, dim=-1)
        pt = pt.gather(1, target).squeeze(-1)
        logpt = logpt.gather(1, target).squeeze(-1)

        if self.alpha is not None:
            at = self.alpha.gather(0, target.squeeze(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

    @staticmethod
    def convert_binary_pred_to_two_dimension(x, is_logits=True):
        probs = torch.sigmoid(x) if is_logits else x
        probs = probs.unsqueeze(-1)
        probs = torch.cat([1-probs, probs], dim=-1)
        logprob = torch.log(probs + 1e-4)
        return logprob

    def __str__(self):
        return f"Focal Loss gamma:{self.gamma}"

    def __repr__(self):
        return str(self)

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

def evaluate_model(model, dataloader, device, epoch):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", leave=False):
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
    logging.info(f"\n----Epoch: {epoch} - Class-wise Metrics----")
    for i, label in enumerate(labels):
        y_true = [1 if l == i else 0 for l in all_labels]
        y_pred = [1 if p == i else 0 for p in all_preds]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        logging.info(f"Epoch: {epoch} - {label}: precision: {precision:.4f}, recall: {recall:.4f}, f1 score: {f1:.4f}")

    # Calculate overall metrics
    overall_acc = accuracy_score(all_labels, all_preds)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    # Log overall metrics
    logging.info(f"\n----Epoch: {epoch} - OVERALL----")
    average_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    logging.info(f"Epoch: {epoch} - val Loss: {average_loss:.4f} | accuracy: {overall_acc:.4f} | precision: {overall_precision:.4f} | recall: {overall_recall:.4f} | f1 score: {overall_f1:.4f}")

    return overall_f1