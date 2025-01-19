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
                 class_weight_tensor,
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
            self.image_to_text_attention = CrossAttention(d_in=768 + 768, d_out_kq=768 + 768, d_out_v=768 + 768)
        elif self.fusion_method == 'attention':
            self.self_attention = SelfAttention(d_in=768 + 768 + 768, d_out_kq=768+768+768, d_out_v=768 + 768 + 768)
        # Define the output layer
        combined_size = 0
        if self.fusion_method == 'concat':
          combined_size = 768 + 768 + 768 
        elif self.fusion_method == 'cross_attention':
          combined_size = 768 + 768 + 768 + 768
        elif self.fusion_method == 'attention':
          combined_size = 768 + 768 + 768
        self.fc = nn.Linear(combined_size, num_labels)
        self.loss_fct = FocalLoss(gamma=self.gamma, alpha=class_weight_tensor)

    def forward(self, image_features, text_features, labels=None):
        print(len(image_features))
        print(len(text_features))
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
            # Calculate loss using Focal Loss
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits