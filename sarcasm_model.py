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
            self.self_attention = SelfAttention(d_in=768+768+ 768, d_out_kq=768+768+768, d_out_v=768 + 768 + 768)
            
        self.mixer = nn.Linear(768 + 768, 2) 
        self.text_refinement = nn.Linear(768, 768)
        self.image_refinement = nn.Linear(768, 768)    
        
        # Define the output layer
        combined_size = 0
        if self.fusion_method == 'concat':
          combined_size = 768 + 768 + 768 
        elif self.fusion_method == 'cross_attention':
          combined_size = 768 + 768 + 768 + 768
        elif self.fusion_method == 'attention':
          combined_size = 768 + 768 + 768
        self.fc = nn.Sequential(
            nn.Linear(combined_size, combined_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(combined_size // 2, num_labels)
        )
        self.loss_fct = FocalLoss(gamma=self.gamma, alpha=class_weight_tensor)

    def forward(self, image_features, text_features, labels=None):
        mixer_input = torch.cat((image_features, text_features), dim=1)
        attention_weights = torch.softmax(self.mixer(mixer_input), dim=1)
        alpha_image, alpha_text = attention_weights[:, 0].unsqueeze(1), attention_weights[:, 1].unsqueeze(1)
        mixed_features = alpha_image * image_features + alpha_text * text_features
        
        refined_text_features = text_features + self.text_refinement(text_features)
        refined_image_features = image_features + self.image_refinement(image_features)
        
        if self.fusion_method == 'cross_attention':
            attended_text = self.text_to_image_attention(text_features, image_features)
            attended_image = self.image_to_text_attention(image_features, text_features)
            combined_features = torch.cat((attended_text, attended_image), dim=1)
        elif self.fusion_method == 'attention':
            combined_features = torch.cat((image_features, text_features), dim=1)
            attended_features = self.self_attention(combined_features)
            combined_features = attended_features
        elif self.fusion_method == 'mean':
            combined_features = torch.cat((image_features.mean(), text_features.mean()), dim=1)
        else:
            combined_features = torch.cat((image_features, text_features), dim=1)
    
        logits = self.fc(combined_features)

        if labels is not None:
            # Calculate loss using Focal Loss
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits