import torch
import torch.nn as nn
import logging
from utils import CrossAttention, SelfAttention, FocalLoss
import numpy as np

class VietnameseSarcasmClassifier(nn.Module):
    def __init__(self,
                 mode,
                 text_encoder,
                 image_encoder,
                 class_weight=None,
                 fusion_method='concat',
                 num_labels=4,
                 gamma=2.0):
        super(VietnameseSarcasmClassifier, self).__init__()
        self.num_labels = num_labels
        self.mode = mode
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_method = fusion_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.class_weight = class_weight

        if self.fusion_method == 'cross_attention':
            self.text_to_image_attention = CrossAttention(d_in=1024, d_out_kq=2024, d_out_v=2024)
            self.image_to_text_attention = CrossAttention(d_in=2024, d_out_kq=1024, d_out_v=1024)
            combined_size = 1024 + 2024 
        elif self.fusion_method == 'attention':
            self.self_attention = SelfAttention(d_in=2024 + 1024, d_out_kq=2024 + 1024, d_out_v=2024 + 1024)
            combined_size = 2024 + 1024
        else:
            combined_size = 2024 + 1024

        self.fc = nn.Sequential(
            nn.Linear(combined_size, combined_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(combined_size // 2, num_labels),
        )
        logging.info(f"Using class_weight: {self.class_weight}")
        self.loss_fct = FocalLoss(gamma=self.gamma, alpha=self.class_weight) if self.class_weight is not None else FocalLoss(gamma=self.gamma)

    def forward(self, image_features, text_features, labels=None):
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

        logits = self.fc(combined_features)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits