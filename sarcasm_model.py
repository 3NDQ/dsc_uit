import torch
import torch.nn as nn
import logging
from utils import CrossAttention, SelfAttention, FocalLoss, WeightedFocalLoss
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

        # --- "Old" Code (Projection and Attention) ---
        self.text_projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.image_projection = nn.Sequential(
            nn.Linear(2024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        if self.fusion_method == 'cross_attention':
            self.text_to_image_attention = CrossAttention(d_in_q=512, d_in_kv=512, d_out_kq=512, d_out_v=512)
            self.image_to_text_attention = CrossAttention(d_in_q=512, d_in_kv=512, d_out_kq=512, d_out_v=512)
            self.attention_combined_size = 512 * 2
        elif self.fusion_method == 'attention':
            self.self_attention = SelfAttention(d_in=512 * 2, d_out_kq=512, d_out_v=512)
            self.attention_combined_size = 512
        else:
            self.attention_combined_size = 512 * 2

        # --- "New" Code (Image Architecture) ---
        self.image_dense = nn.Linear(2024, 2048)
        self.image_dropout = nn.Dropout(dropout_rate)

        self.text_dense1 = nn.Linear(1024, 1024)
        self.text_dense3 = nn.Linear(1024, 1024)
        self.text_dropout1 = nn.Dropout(dropout_rate)
        self.text_dropout2 = nn.Dropout(dropout_rate)
        self.text_dense2 = nn.Linear(1024, 512)
        self.text_dense4 = nn.Linear(1024, 512)

        self.fusion_dense5 = nn.Linear(1024, 1024)
        self.fusion_dropout3 = nn.Dropout(dropout_rate)
        self.fusion_dense6 = nn.Linear(1024, 512)
        self.fusion_dropout4 = nn.Dropout(dropout_rate)
        self.fusion_dense7 = nn.Linear(512, 256)
        self.fusion_dropout5 = nn.Dropout(dropout_rate)

        # --- Output Layers ---
        # We'll use a weighted combination of the outputs from both architectures
        self.fc = nn.Linear(self.attention_combined_size, num_labels)  # From "old" code
        self.new_output_dense = nn.Linear(256, num_labels)  # From "new" code
        self.output_weights = nn.Parameter(torch.tensor([0, 1.0]))  # Learnable weights for combining outputs

        # --- Loss Function ---
        logging.info(f"Using class_weight: {self.class_weight}")
        self.class_weight = [0.001, 0.9498, 0.05, 0.001]
        self.loss_fct = WeightedFocalLoss(gamma=self.gamma, alpha=self.class_weight) if self.class_weight is not None else FocalLoss(gamma=self.gamma)
    def forward(self, image_features, text_features, labels=None):
        # --- "Old" Code Forward Pass ---
        projected_text_features = self.text_projection(text_features)
        projected_image_features = self.image_projection(image_features)

        if self.fusion_method == 'cross_attention':
            attended_text = self.text_to_image_attention(projected_text_features, projected_image_features)
            attended_image = self.image_to_text_attention(projected_image_features, projected_text_features)
            attention_combined_features = torch.cat((attended_text, attended_image), dim=1)
        elif self.fusion_method == 'attention':
            combined_features = torch.cat((projected_image_features, projected_text_features), dim=1)
            attention_combined_features = self.self_attention(combined_features)
        else:
            attention_combined_features = torch.cat((projected_image_features, projected_text_features), dim=1)

        old_logits = self.fc(attention_combined_features)

        # --- "New" Code Forward Pass ---
        image_out = self.image_dense(image_features)
        image_out = nn.ReLU()(image_out)
        image_out = self.image_dropout(image_out)

        text_out = self.text_dense1(text_features)
        text_out = nn.ReLU()(text_out)
        text_out = self.text_dropout1(text_out)

        text_out_2 = self.text_dense3(text_features)
        text_out_2 = nn.ReLU()(text_out_2)
        text_out_2 = self.text_dropout2(text_out_2)

        text_out = self.text_dense2(text_out)
        text_out = nn.ReLU()(text_out)

        text_out_2 = self.text_dense4(text_out_2)
        text_out_2 = nn.ReLU()(text_out_2)

        concat_out = torch.cat((text_out, text_out_2), dim=1)

        fusion_out = self.fusion_dense5(concat_out)
        fusion_out = nn.ReLU()(fusion_out)
        fusion_out = self.fusion_dropout3(fusion_out)
        fusion_out = self.fusion_dense6(fusion_out)
        fusion_out = nn.ReLU()(fusion_out)
        fusion_out = self.fusion_dropout4(fusion_out)
        fusion_out = self.fusion_dense7(fusion_out)
        fusion_out = nn.ReLU()(fusion_out)
        fusion_out = self.fusion_dropout5(fusion_out)

        new_logits = self.new_output_dense(fusion_out)

        # --- Combine Outputs ---
        # Normalize weights to sum to 1
        normalized_weights = nn.functional.softmax(self.output_weights, dim=0)
        # Weighted average of logits
        logits = normalized_weights[0] * old_logits + normalized_weights[1] * new_logits

        # --- Calculate Loss ---
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits