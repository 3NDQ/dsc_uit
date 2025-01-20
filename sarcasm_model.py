import torch
import torch.nn as nn
import logging
from utils import CrossAttention, SelfAttention, FocalLoss, WeightedCrossEntropyLoss
import numpy as np

class VietnameseSarcasmClassifier(nn.Module):
    def __init__(self,
                 mode,
                 class_weight=None,
                 fusion_method='concat',
                 num_labels=4,
                 dropout_rate=0.2,
                 gamma=5.0,
                 loss_type='focal',
                 label_smoothing=0.0): 
        super(VietnameseSarcasmClassifier, self).__init__()
        self.num_labels = num_labels
        self.mode = mode
        self.fusion_method = fusion_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.class_weight = class_weight
        self.dropout_rate = dropout_rate
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing 
        
        self.dropout = nn.Dropout(dropout_rate)
        image_feature_size = 1000  
        text_feature_size = 1024
        ocr_feature_size = 1024
        combined_feature_size = 2024
        text_feature2_size = 768
        self.image_dense1 = nn.Linear(image_feature_size, 1024)
        self.image_dense2 = nn.Linear(1024, 512)

        self.ocr_dense1 = nn.Linear(ocr_feature_size, 1024)
        self.ocr_dense2 = nn.Linear(1024, 512)

        self.image_dense3 = nn.Linear(image_feature_size + 512 + 512, 1024)
        self.image_dense4 = nn.Linear(1024, 512)

        self.text_dense1 = nn.Linear(text_feature_size, 512)
        self.text_dense3 = nn.Linear(512, 256)

        self.text_dense2 = nn.Linear(text_feature2_size, 512)
        self.text_dense4 = nn.Linear(512, 256)

        self.text_dense5 = nn.Linear(text_feature_size + 256 + 256, 512)

        if self.fusion_method == 'cross_attention':
            self.text_to_image_attention = CrossAttention(d_in_q=512, d_in_kv=512, d_out_kq=512, d_out_v=512)
            self.image_to_text_attention = CrossAttention(d_in_q=512, d_in_kv=512, d_out_kq=512, d_out_v=512)
            combined_size = 512 + 512 
        elif self.fusion_method == 'attention':
            self.self_attention = SelfAttention(d_in=512 + 512, d_out_kq=512, d_out_v=512)
            combined_size = 512
        else: # concat
            combined_size = 512 + 512

        self.fusion_dense1 = nn.Linear(combined_size, 512)
        self.fusion_dense2 = nn.Linear(512, 256)
            
        self.fc = nn.Sequential(
            nn.Linear(256, self.num_labels),
        )

        logging.info(f"Using class_weight: {self.class_weight}")
        if self.loss_type == 'focal':
            self.loss_fct = FocalLoss(gamma=self.gamma, alpha=self.class_weight, label_smoothing=self.label_smoothing)
        elif self.loss_type == 'cross_entropy':
            self.loss_fct = WeightedCrossEntropyLoss(weight=self.class_weight, label_smoothing=self.label_smoothing)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def forward(self,
                combined_image_features,
                ocr_features,
                image_features,
                text_features,
                text_features2,
                labels=None):
        image_out = self.image_dense1(image_features)
        image_out = nn.GELU()(image_out)
        image_out = self.dropout(image_out)
        image_out = self.image_dense2(image_out)
        image_out = nn.GELU()(image_out)
        image_out = self.dropout(image_out)
        
        ocr_out = self.ocr_dense1(ocr_features)
        ocr_out = nn.GELU()(ocr_out)
        ocr_out = self.dropout(ocr_out)
        ocr_out = self.ocr_dense2(ocr_out)
        ocr_out = nn.GELU()(ocr_out)
        ocr_out = self.dropout(ocr_out)
        
        image_out_combined = torch.cat((image_out, ocr_out, image_features), dim=1)
        image_out_combined = self.image_dense3(image_out_combined)
        image_out_combined = nn.GELU()(image_out_combined)
        image_out_combined = self.dropout(image_out_combined)
        image_out_combined = self.image_dense4(image_out_combined)
        image_out_combined = nn.GELU()(image_out_combined)
        image_out_combined = self.dropout(image_out_combined)
        
        text_out1 = self.text_dense1(text_features)
        text_out1 = nn.GELU()(text_out1)
        text_out1 = self.dropout(text_out1)
        text_out1 = self.text_dense3(text_out1)
        text_out1 = nn.GELU()(text_out1)
        text_out1 = self.dropout(text_out1)
        
        text_out2 = self.text_dense2(text_features2)
        text_out2 = nn.GELU()(text_out2)
        text_out2 = self.dropout(text_out2)
        text_out2 = self.text_dense4(text_out2)
        text_out2 = nn.GELU()(text_out2)
        text_out2 = self.dropout(text_out2)
        
        text_out_combined = torch.cat((text_out1, text_out2, text_features), dim=1)
        text_out_combined = self.text_dense5(text_out_combined)
        text_out_combined = nn.GELU()(text_out_combined)
        text_out_combined = self.dropout(text_out_combined)
        
        if self.fusion_method == 'cross_attention':
            attended_text = self.text_to_image_attention(text_out_combined, image_out_combined)
            attended_image = self.image_to_text_attention(image_out_combined, text_out_combined)
            combined_features = torch.cat((attended_text, attended_image), dim=1)
        elif self.fusion_method == 'attention':
            combined_features = torch.cat((image_out_combined, text_out_combined), dim=1)
            combined_features = self.self_attention(combined_features)
        else:
            combined_features = torch.cat((image_out_combined, text_out_combined), dim=1)
            
        fusion_out = self.fusion_dense1(combined_features)
        fusion_out = nn.GELU()(fusion_out)
        fusion_out = self.dropout(fusion_out)
        
        fusion_out = self.fusion_dense2(fusion_out)
        fusion_out = nn.GELU()(fusion_out)
        fusion_out = self.dropout(fusion_out)
        
        logits = self.fc(fusion_out)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits