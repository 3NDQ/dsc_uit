import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CrossAttention, SelfAttention, MultiHeadAttention  # Assuming MultiHeadAttention is implemented
import numpy as np

# Focal Loss with Label Smoothing
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float)

    def forward(self, logits, targets):
        # Apply label smoothing
        targets = (1 - self.label_smoothing) * F.one_hot(targets, num_classes=self.alpha.size(0)) + \
                  self.label_smoothing / self.alpha.size(0)
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            alpha_t = alpha.gather(0, targets.argmax(dim=1))
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Vietnamese Sarcasm Classifier with Advanced Techniques
class VietnameseSarcasmClassifier(nn.Module):
    def __init__(self,
                 mode,
                 text_encoder,
                 image_encoder,
                 fusion_method='concat',
                 num_labels=4,
                 gamma=2.0,
                 dropout_rate=0.1,
                 label_smoothing=0.1,
                 num_heads=8):  # Multi-head attention heads
        super(VietnameseSarcasmClassifier, self).__init__()
        self.num_labels = num_labels
        self.mode = mode
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_method = fusion_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        self.num_heads = num_heads

        # Define attention layers based on fusion method
        if self.fusion_method == 'cross_attention':
            self.text_to_image_attention = MultiHeadAttention(d_model=768, num_heads=num_heads)
            self.image_to_text_attention = MultiHeadAttention(d_model=768, num_heads=num_heads)
        elif self.fusion_method == 'attention':
            self.self_attention = MultiHeadAttention(d_model=768 * 2, num_heads=num_heads)
        
        # Define the output layer
        combined_size = 0
        if self.fusion_method == 'concat':
            combined_size = 768 * 2  # Image + Text features
        elif self.fusion_method == 'cross_attention':
            combined_size = 768 * 2  # Attended features
        elif self.fusion_method == 'attention':
            combined_size = 768 * 2  # Self-attended features

        # Add residual connections and layer normalization
        self.residual_layer_norm = nn.LayerNorm(combined_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer with Xavier initialization
        self.fc = nn.Linear(combined_size, num_labels)
        nn.init.xavier_uniform_(self.fc.weight)  # Xavier initialization

        # Focal Loss with label smoothing
        self.loss_fct = FocalLoss(gamma=self.gamma, alpha=[0.1, 0.4, 0.2, 0.1], label_smoothing=self.label_smoothing)

    def forward(self, image_features, text_features, labels=None):
        # Combine features based on fusion method
        if self.fusion_method == 'cross_attention':
            attended_text = self.text_to_image_attention(text_features, image_features)
            attended_image = self.image_to_text_attention(image_features, text_features)
            combined_features = torch.cat((attended_text, attended_image), dim=1)
        elif self.fusion_method == 'attention':
            combined_features = torch.cat((image_features, text_features), dim=1)
            attended_features = self.self_attention(combined_features)
            combined_features = attended_features
        else:
            combined_features = torch.cat((image_features, text_features), dim=1)

        # Add residual connection and layer normalization
        combined_features = self.residual_layer_norm(combined_features + combined_features)  # Residual connection
        combined_features = self.dropout(combined_features)  # Dropout

        # Pass through the fully connected layer
        logits = self.fc(combined_features)

        if labels is not None:
            # Calculate loss using Focal Loss with label smoothing
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits

# Example usage with advanced techniques
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, clip_value=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            image_features, text_features, labels = batch
            image_features, text_features, labels = image_features.to(device), text_features.to(device), labels.to(device)

            optimizer.zero_grad()
            loss, logits = model(image_features, text_features, labels)
            loss.backward()
            optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                image_features, text_features, labels = batch
                image_features, text_features, labels = image_features.to(device), text_features.to(device), labels.to(device)
                loss, _ = model(image_features, text_features, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss / len(val_loader)}")
