import torch
import torch.nn as nn
import logging
from utils import CrossAttention, SelfAttention
import numpy as np

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
            # Make alpha device-compatible
            alpha = self.alpha.to(targets.device)
            # Use alpha according to class
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

class VietnameseSarcasmClassifier(nn.Module):
    def __init__(self,
                 mode,
                 text_encoder,
                 image_encoder,
                 fusion_method='concat',
                 num_labels=4,
                 gamma=2.0):  # Add gamma parameter
        super(VietnameseSarcasmClassifier, self).__init__()
        self.num_labels = num_labels
        self.mode = mode
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_method = fusion_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma  # Store gamma
        # Define attention layers based on fusion method
        if self.fusion_method == 'cross_attention':
            self.text_to_image_attention = CrossAttention(d_in=768, d_out_kq=768, d_out_v=768)
            self.image_to_text_attention = CrossAttention(d_in=768+768, d_out_kq=768+768, d_out_v=768+768)
        elif self.fusion_method == 'attention':
            self.self_attention = SelfAttention(d_in=768+768+768, d_out_kq=768+768+768, d_out_v=768+768+768)
            
        # Define the output layer
        combined_size = 0
        if self.fusion_method == 'concat':
          combined_size = 768 + 768 + 768 
        elif self.fusion_method == 'cross_attention':
          combined_size = 768 + 768 + 768+768
        elif self.fusion_method == 'attention':
          combined_size = 768 + 768 + 768
        self.fc = nn.Linear(combined_size, num_labels)
        self.loss_fct = FocalLoss(gamma=self.gamma, alpha=[0.1, 0.4, 0.2, 0.1])

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

        # Pass through the fully connected layer
        logits = self.fc(combined_features)

        if labels is not None:
            # Calculate loss using Focal Loss
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits