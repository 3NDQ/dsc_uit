# utils.py
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import logging

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='none'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):

        BCE_loss = F.nll_loss(inputs, targets, reduction=self.reduction)
        targets = targets.type(torch.long)
        # at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha[targets] * (1-pt) ** self.gamma * BCE_loss
        loss_weighted_manual = F_loss.sum() / self.alpha[targets].sum()
        return loss_weighted_manual
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float)

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            alpha_t = alpha.gather(0, targets.data.view(-1))
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model  # Dimensionality of the input
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimensionality of each head's key, query, and value
        
        # Linear layers for query, key, value, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        # Linear transformations
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads
        attn_output = self.combine_heads(attn_output)
        
        # Final linear transformation
        output = self.W_o(attn_output)
        
        return output
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.randn(d_in, d_out_kq) * (1. / d_in**0.5))  # Scaled initialization
        self.W_key = nn.Parameter(torch.randn(d_in, d_out_kq) * (1. / d_in**0.5))  # Scaled initialization
        self.W_value = nn.Parameter(torch.randn(d_in, d_out_v) * (1. / d_in**0.5))  # Scaled initialization

    def forward(self, x):
        keys = x.matmul(self.W_key)
        queries = x.matmul(self.W_query)
        values = x.matmul(self.W_value)
        attn_scores = queries.matmul(keys.T)
        attn_scores = attn_scores / (self.d_out_kq ** 0.5)  # Add scaling here
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vec = attn_weights.matmul(values)
        return context_vec

class CrossAttention(nn.Module):
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v, num_heads):
        super().__init__()
        self.d_model = d_in  # Input dimension
        self.d_out_kq = d_out_kq
        self.d_out_v = d_out_v
        self.num_heads = num_heads
        self.head_dim = d_out_kq // num_heads  # Dimension per head

        assert d_out_kq % num_heads == 0, "d_out_kq must be divisible by num_heads"

        # Linear projections for Q, K, V for all heads at once
        self.W_q = nn.Linear(d_in, d_out_kq)
        self.W_k = nn.Linear(d_in, d_out_kq)
        self.W_v = nn.Linear(d_in, d_out_v)

        # Final linear layer for the combined output
        self.fc_out = nn.Linear(d_out_v, d_out_v)  

    def forward(self, x_1, x_2):
        batch_size = x_1.shape[0]

        Q = self.W_q(x_1) 
        K = self.W_k(x_2)  # (batch_size, seq_len_k, d_out_kq)
        V = self.W_v(x_2)  # (batch_size, seq_len_k, d_out_v)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_q, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_k, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_k, head_dim)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        attn = torch.softmax(energy, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        out = torch.matmul(attn, V)  # (batch_size, num_heads, seq_len_q, head_dim)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_out_v)  # (batch_size, seq_len_q, d_out_v)

        out = self.fc_out(out) 

        return out
    
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
            image_features, text_features, labels = batch  # Adjusted for pre-extracted features
            image_features = image_features.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)

            outputs = model(image_features=image_features, text_features=text_features, labels=labels) # Using pre-extracted features
            loss, logits = outputs

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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