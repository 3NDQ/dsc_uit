# sarcasm_model.py
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
                 text_encoder2=None,
                 image_encoder2=None,
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
        
        
        self.image_dense1 = nn.Linear(2024, 2048)
        self.image_dense2 = nn.Linear(2048, 1024)

        self.text_dense1 = nn.Linear(1024, 1024)
        self.text_dense3 = nn.Linear(1024, 512)
        
        self.text_dense2 = nn.Linear(1024, 1024)
        self.text_dense4 = nn.Linear(1024, 512)

        self.combined_dense = nn.Linear(1024, 512)
        
        # Define attention layers based on fusion method
        if self.fusion_method == 'cross_attention':
            self.text_to_image_attention = CrossAttention(d_in_q=1024, d_in_kv=1024, d_out_kq=512, d_out_v=512)
            self.image_to_text_attention = CrossAttention(d_in_q=1024, d_in_kv=1024, d_out_kq=512, d_out_v=512)
            self.fusion_dense = nn.Linear(1024, 512)
        elif self.fusion_method == 'attention':
            self.self_attention = SelfAttention(d_in=2048, d_out_kq=1024, d_out_v=512)
            self.fusion_dense = nn.Linear(512, 512)
        else: # concat
            self.fusion_dense = nn.Linear(1536, 1024)
            self.fusion_dense1 = nn.Linear(1024, 512)
            
        self.fc = nn.Sequential(
            nn.Linear(512, 512 // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512 // 2, num_labels),
        )
        logging.info(f"Using class_weight: {self.class_weight}")
        self.loss_fct = FocalLoss(gamma=self.gamma, alpha=self.class_weight) if self.class_weight is not None else FocalLoss(gamma=self.gamma)

    def forward(self, image_features, text_features, labels=None):
        
        image_out = self.image_dense1(image_features)
        image_out = nn.ReLU()(image_out)
        image_out = self.dropout(image_out)
        
        image_out = self.image_dense2(image_out)
        image_out = nn.ReLU()(image_out)
        image_out = self.dropout(image_out)
        
        
        text_out1 = self.text_dense1(text_features)
        text_out1 = nn.ReLU()(text_out1)
        text_out1 = self.dropout(text_out1)
        text_out1 = self.text_dense3(text_out1)
        text_out1 = nn.ReLU()(text_out1)
        text_out1 = self.dropout(text_out1)
        
        text_out2 = self.text_dense2(text_features)
        text_out2 = nn.ReLU()(text_out2)
        text_out2 = self.dropout(text_out2)
        text_out2 = self.text_dense4(text_out2)
        text_out2 = nn.ReLU()(text_out2)
        text_out2 = self.dropout(text_out2)
        
        combined_text = torch.cat((text_out1, text_out2), dim=1)
        combined_text = self.combined_dense(combined_text)
        combined_text = nn.ReLU()(combined_text)
        
        if self.fusion_method == 'cross_attention':
            text_out = torch.cat((text_out1, text_out2), dim=1)
            attended_text = self.text_to_image_attention(text_out, image_out)
            attended_image = self.image_to_text_attention(image_out, text_out)
            combined_features = torch.cat((attended_text, attended_image), dim=1)
        elif self.fusion_method == 'attention':
            combined_features = torch.cat((image_out, text_out1, text_out2), dim=1)
            combined_features = self.self_attention(combined_features)
        else: # concat
            combined_features = torch.cat((image_out, combined_text), dim=1)
            
        fusion_out = self.fusion_dense(combined_features)
        fusion_out = nn.ReLU()(fusion_out)
        fusion_out = self.dropout(fusion_out)
        
        if self.fusion_method != 'attention' and self.fusion_method != 'cross_attention':
            fusion_out = self.fusion_dense1(fusion_out)
            fusion_out = nn.ReLU()(fusion_out)
            fusion_out = self.dropout(fusion_out)
        
        logits = self.fc(fusion_out)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits