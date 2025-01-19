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
                 dropout_rate=0.2,
                 gamma=2.0): 
        
        super(VietnameseSarcasmClassifier, self).__init__()
        self.num_labels = num_labels
        self.mode = mode
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_method = fusion_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma  # Store gamma
        self.class_weight = class_weight
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        
        
        self.image_dense = nn.Linear(2024, 2048)
        
        self.text_dense1 = nn.Linear(1024, 1024)
        self.text_dense2 = nn.Linear(1024, 512)
        
        self.fusion_dense = nn.Linear(3584, 2048)
        self.fusion_dense1 = nn.Linear(2048, 1024)
        self.fusion_dense2 = nn.Linear(2048, 512)

        # Define attention layers based on fusion method
        if self.fusion_method == 'cross_attention':
            self.text_to_image_attention = CrossAttention(d_in_q=1024, d_in_kv=2024, d_out_kq=2024, d_out_v=2024)  # d_in for W_query should be 1024 (text_features)
            self.image_to_text_attention = CrossAttention(d_in_q=2024, d_in_kv=1024, d_out_kq=1024, d_out_v=1024)  # d_in for W_query should be 2024 (image_features)
            combined_size = 1024 + 2024
        elif self.fusion_method == 'attention':
            self.self_attention = SelfAttention(d_in=1024 + 2024, d_out_kq=1024 + 2024, d_out_v=1024 + 2024)
            combined_size = 1024 + 2024
        else:
            combined_size = 1024 + 2024
            
        self.fc = nn.Sequential(
            nn.Linear(512, 512 // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512 // 2, num_labels),
        )
        self.class_weight = [1, 94.97, 5, 1]
        logging.info(f"Using class_weight: {self.class_weight}")
        self.loss_fct = FocalLoss(gamma=self.gamma, alpha=self.class_weight) if self.class_weight is not None else FocalLoss(gamma=self.gamma)

    def forward(self, image_features, text_features, labels=None):
        
        image_out = self.image_dense(image_features)
        image_out = nn.ReLU()(image_out)
        image_out = self.dropout(image_out)
        
        text_out1 = self.text_dense1(text_features)
        text_out1 = nn.ReLU()(text_out1)
        text_out1 = self.dropout(text_out1)
        
        text_out2 = self.text_dense2(text_features)
        text_out2 = nn.ReLU()(text_out2)
        text_out2 = self.dropout(text_out2)
        
        text_out_combined = torch.cat((text_out1, text_out2), dim=1)
        
        if self.fusion_method == 'cross_attention':
            attended_text = self.text_to_image_attention(text_features, image_features)
            attended_image = self.image_to_text_attention(image_features, text_features)
            combined_features = torch.cat((attended_text, attended_image), dim=1)
        elif self.fusion_method == 'attention':
            combined_features = torch.cat((image_features, text_features), dim=1)
            attended_features = self.self_attention(combined_features)
            combined_features = attended_features
        else:
            combined_features = torch.cat((image_out, text_out_combined), dim=1)
            
        fusion_out = self.fusion_dense1(combined_features)
        fusion_out = nn.GELU()
        fusion_out = self.dropout(fusion_out)
        
        fusion_out = self.fusion_dense2(fusion_out)
        fusion_out = nn.GELU()
        fusion_out = self.dropout(fusion_out)
        
        fusion_out = self.fusion_dense3(fusion_out)
        fusion_out = nn.GELU()
        fusion_out = self.dropout(fusion_out)
        
        logits = self.fc(combined_features)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits