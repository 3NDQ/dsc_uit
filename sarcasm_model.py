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
        self.gamma = gamma
        self.class_weight = class_weight
        self.dropout_rate = dropout_rate

        # --- Text and Image Processing (from "new" architecture) ---
        self.image_dense = nn.Linear(2024, 2048)
        self.image_dropout = nn.Dropout(dropout_rate)

        self.text_dense1 = nn.Linear(1024, 1024)
        self.text_dense3 = nn.Linear(1024, 1024)
        self.text_dropout1 = nn.Dropout(dropout_rate)
        self.text_dropout2 = nn.Dropout(dropout_rate)
        self.text_dense2 = nn.Linear(1024, 512)
        self.text_dense4 = nn.Linear(1024, 512)

        # --- Fusion Preparation (from "old" architecture) ---
        # We'll adjust the input sizes to match the outputs of the new processing
        self.text_projection = nn.Sequential(
            nn.Linear(512, 256),  # Adjusted input size
            nn.GeLU(),
            nn.Dropout(dropout_rate)
        )
        self.image_projection = nn.Sequential(
            nn.Linear(2048, 256),  # Adjusted input size
            nn.GeLU(),
            nn.Dropout(dropout_rate)
        )

        # --- Attention Mechanism (from "old" architecture) ---
        if self.fusion_method == 'cross_attention':
            self.text_to_image_attention = CrossAttention(d_in_q=256, d_in_kv=256, d_out_kq=256, d_out_v=256)
            self.image_to_text_attention = CrossAttention(d_in_q=256, d_in_kv=256, d_out_kq=256, d_out_v=256)
            self.attention_combined_size = 256 * 2
        elif self.fusion_method == 'attention':
            self.self_attention = SelfAttention(d_in=256 * 2, d_out_kq=256, d_out_v=256)
            self.attention_combined_size = 256
        else:
            self.attention_combined_size = 256 * 2 # For 'concat'

        # --- Post-Attention Fusion and Output (from "new" architecture) ---
        self.fusion_dense5 = nn.Linear(self.attention_combined_size, 512)  # Adjusted input size
        self.fusion_dropout3 = nn.Dropout(dropout_rate)
        self.fusion_dense6 = nn.Linear(512, 256)
        self.fusion_dropout4 = nn.Dropout(dropout_rate)
        self.fusion_dense7 = nn.Linear(256, 128)
        self.fusion_dropout5 = nn.Dropout(dropout_rate)

        # --- Output Layer ---
        self.fc = nn.Linear(128, num_labels) # Output layer

        # --- Loss Function ---
        logging.info(f"Using class_weight: {self.class_weight}")
        self.loss_fct = FocalLoss(gamma=self.gamma, alpha=self.class_weight) if self.class_weight is not None else FocalLoss(gamma=self.gamma)

    def forward(self, image_features, text_features, labels=None):
        # --- Initial Processing (from "new" architecture) ---
        image_out = self.image_dense(image_features)
        image_out = nn.GeLU()(image_out)
        image_out = self.image_dropout(image_out)

        text_out = self.text_dense1(text_features)
        text_out = nn.GeLU()(text_out)
        text_out = self.text_dropout1(text_out)

        text_out_2 = self.text_dense3(text_features)
        text_out_2 = nn.GeLU()(text_out_2)
        text_out_2 = self.text_dropout2(text_out_2)

        text_out = self.text_dense2(text_out)
        text_out = nn.GeLU()(text_out)

        text_out_2 = self.text_dense4(text_out_2)
        text_out_2 = nn.GeLU()(text_out_2)
        
        # --- Projection (from "old" architecture) ---
        # Note: We are now projecting the processed features
        projected_text_features = self.text_projection(text_out)
        projected_image_features = self.image_projection(image_out)

        # --- Attention (from "old" architecture) ---
        if self.fusion_method == 'cross_attention':
            attended_text = self.text_to_image_attention(projected_text_features, projected_image_features)
            attended_image = self.image_to_text_attention(projected_image_features, projected_text_features)
            attention_combined_features = torch.cat((attended_text, attended_image), dim=1)
        elif self.fusion_method == 'attention':
            combined_features = torch.cat((projected_image_features, projected_text_features), dim=1)
            attention_combined_features = self.self_attention(combined_features)
        else: # 'concat'
            attention_combined_features = torch.cat((projected_image_features, projected_text_features), dim=1)
            
        # --- Fusion and Output (from "new" architecture) ---
        fusion_out = self.fusion_dense5(attention_combined_features)
        fusion_out = nn.GeLU()(fusion_out)
        fusion_out = self.fusion_dropout3(fusion_out)
        fusion_out = self.fusion_dense6(fusion_out)
        fusion_out = nn.GeLU()(fusion_out)
        fusion_out = self.fusion_dropout4(fusion_out)
        fusion_out = self.fusion_dense7(fusion_out)
        fusion_out = nn.GeLU()(fusion_out)
        fusion_out = self.fusion_dropout5(fusion_out)

        logits = self.fc(fusion_out)

        # --- Calculate Loss ---
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits