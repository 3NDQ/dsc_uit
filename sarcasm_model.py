import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from utils import CrossAttention, SelfAttention, FocalLoss, WeightedCrossEntropyLoss
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
                 gamma=5.0,
                 loss_type='focal',
                 label_smoothing=0.0):  # Add label_smoothing parameter
        super(VietnameseSarcasmClassifier, self).__init__()
        self.num_labels = num_labels
        self.mode = mode
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_method = fusion_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.class_weight = class_weight
        self.dropout_rate = dropout_rate
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing 
        
        self.image_dense1 = nn.Linear(image_dim, 1000)
        self.image_dropout1 = nn.Dropout(dropout_rate)
        self.image_dense2 = nn.Linear(1000, 512)
        self.image_dropout2 = nn.Dropout(dropout_rate)
        self.image_dense3 = nn.Linear(512, 256)
        
        # Text branch
        self.text_dense1 = nn.Linear(text_dim, 1024)
        self.text_dropout1 = nn.Dropout(dropout_rate)
        self.text_dense2 = nn.Linear(1024, 512)
        
        # Combined branch
        self.combined_dense1 = nn.Linear(256 + 512, 1024)  # Concatenated image and text features
        self.combined_dropout1 = nn.Dropout(dropout_rate)
        self.combined_dense2 = nn.Linear(1024, 512)
        self.combined_dropout2 = nn.Dropout(dropout_rate)
        self.combined_dense3 = nn.Linear(512, 256)
        self.combined_dropout3 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.output_layer = nn.Linear(256, num_labels)
        
        logging.info(f"Using class_weight: {self.class_weight}")
        if self.loss_type == 'focal':
            self.loss_fct = FocalLoss(gamma=self.gamma, alpha=self.class_weight, label_smoothing=self.label_smoothing)
        elif self.loss_type == 'cross_entropy':
            self.loss_fct = WeightedCrossEntropyLoss(weight=self.class_weight, label_smoothing=self.label_smoothing)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def forward(self, image_features, text_features):
        # Image branch
        image_out = F.relu(self.image_dense1(image_features))
        image_out = self.image_dropout1(image_out)
        image_out = F.relu(self.image_dense2(image_out))
        image_out = self.image_dropout2(image_out)
        image_out = F.relu(self.image_dense3(image_out))
        
        # Text branch
        text_out = F.relu(self.text_dense1(text_features))
        text_out = self.text_dropout1(text_out)
        text_out = F.relu(self.text_dense2(text_out))
        
        # Combine image and text features
        combined = torch.cat((image_out, text_out), dim=1)
        
        # Combined branch
        combined_out = F.relu(self.combined_dense1(combined))
        combined_out = self.combined_dropout1(combined_out)
        combined_out = F.relu(self.combined_dense2(combined_out))
        combined_out = self.combined_dropout2(combined_out)
        combined_out = F.relu(self.combined_dense3(combined_out))
        combined_out = self.combined_dropout3(combined_out)
        
        # Output layer
        logits = self.output_layer(combined_out)
        
        return logits