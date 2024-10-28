# sarcasm_models.py
import torch
import torch.nn as nn
import logging
from utils import CrossAttention, SelfAttention

class VietnameseSarcasmClassifier(nn.Module):
    def __init__(self, text_encoder, image_encoder, fusion_method='concat', num_labels=4):
        super(VietnameseSarcasmClassifier, self).__init__()
        self.num_labels = num_labels
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_method = fusion_method
        combined_dim = self.image_encoder.config.hidden_size + self.text_encoder.config.hidden_size
        logging.info(f"Combined dimension: {combined_dim}")
        
        if self.fusion_method == 'cross_attention':
            hidden_size = self.image_encoder.config.hidden_size
            self.text_to_image_attention = CrossAttention(d_in=hidden_size, d_out_kq=hidden_size, d_out_v=hidden_size)
            self.image_to_text_attention = CrossAttention(d_in=hidden_size, d_out_kq=hidden_size, d_out_v=hidden_size)
            logging.info("Cross-Attention layers initialized for both text-to-image and image-to-text.")
        
        elif self.fusion_method == 'attention':
            self.self_attention = SelfAttention(d_in=combined_dim, d_out_kq=combined_dim, d_out_v=combined_dim)
            logging.info("Self-Attention layer initialized for feature fusion.")
        
        # self.projector = nn.Sequential(
        #     nn.Linear(combined_dim, 1024),
        #     nn.LayerNorm(1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(1024, 512),
        #     nn.LayerNorm(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        self.projector = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        logging.info("Projector layers initialized.")
        
        # Classification heads
        self.text_classifier = nn.Linear(512, 2)
        self.image_classifier = nn.Linear(512, 2)
        self.multi_classifier = nn.Linear(512, 2)
        logging.info("Classification heads initialized.")
        
    def forward(self, image, input_ids, attention_mask, labels=None):
        logging.debug("Forward pass started.")
        # Image encoding
        image_outputs = self.image_encoder(image)
        image_features = image_outputs.last_hidden_state[:, 0, :]  
        
        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  
        
        if self.fusion_method == 'cross_attention':
            attended_text = self.text_to_image_attention(text_features, image_features)
            attended_image = self.image_to_text_attention(image_features, text_features)
            
            combined_features = torch.cat((attended_text, attended_image), dim=1)
            logging.debug("Applied cross-attention to combine image and text features.")
            shared_features = self.projector(combined_features)
        
        elif self.fusion_method == 'attention':
            combined_features = torch.cat((image_features, text_features), dim=1)
            attended_features = self.self_attention(combined_features)
            shared_features = self.projector(attended_features)
        else:
            combined_features = torch.cat((image_features, text_features), dim=1)
            shared_features = self.projector(combined_features)
        
        text_logits = self.text_classifier(shared_features)
        image_logits = self.image_classifier(shared_features)
        multi_logits = self.multi_classifier(shared_features)
        
        final_logits = torch.zeros((shared_features.size(0), 4), device=shared_features.device)
        final_logits[:, 0] = multi_logits[:, 1]
        final_logits[:, 1] = text_logits[:, 1]
        final_logits[:, 2] = image_logits[:, 1]
        final_logits[:, 3] = 1 - (multi_logits[:, 1] + text_logits[:, 1] + image_logits[:, 1]).clamp(0, 1)
        
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(final_logits, labels)
            
        return {'loss': loss, 'logits': final_logits} if loss is not None else final_logits

