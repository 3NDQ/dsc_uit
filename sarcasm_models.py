# sarcasm_models.py (No changes needed if already accepting encoders as parameters)

import torch
import torch.nn as nn
import logging

class VietnameseSarcasmClassifier(nn.Module):
    def __init__(self, text_encoder, image_encoder, fusion_method = 'concat', num_labels=4):
        super(VietnameseSarcasmClassifier, self).__init__()
        self.num_labels = num_labels
        
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_method = fusion_method
        # Multimodal Fusion
        combined_dim = self.image_encoder.config.hidden_size + self.text_encoder.config.hidden_size
        logging.info(f"Combined dimension: {combined_dim}")
        
        if self.fusion_method == 'attention':
            self.self_attention = nn.MultiheadAttention(embed_dim=combined_dim, num_heads=8)
            logging.info("Self-Attention layer initialized.")
        self.projector = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        logging.info("Projector layers initialized.")
        
        # Classification heads
        self.text_classifier = nn.Linear(512, 2)  # For text sarcasm
        self.image_classifier = nn.Linear(512, 2)  # For image sarcasm
        self.multi_classifier = nn.Linear(512, 2)  # For multi-modal sarcasm
        logging.info("Classification heads initialized.")
        
    def forward(self, image, input_ids, attention_mask, labels=None):
        logging.debug("Forward pass started.")
        # Image encoding
        image_outputs = self.image_encoder(image)
        image_features = image_outputs.last_hidden_state[:, 0, :]  # Use CLS token
        
        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        
        if self.fusion_method == 'attention':
            # Combine features using attention
            combined_features = torch.cat((image_features, text_features), dim=1).unsqueeze(0)
            attended_features, _ = self.self_attention(combined_features, combined_features, combined_features)
            attended_features = attended_features.squeeze(0)  # Remove extra dimension
            logging.debug("Applied self-attention to combined features.")
            shared_features = self.projector(attended_features)
        else:
            # Default method: concatenate features
            combined_features = torch.cat((image_features, text_features), dim=1)
            logging.debug("Combined image and text features using concatenation.")
            shared_features = self.projector(combined_features)
        logging.debug("Projected combined features to shared space.")
        
        # Multiple classification heads
        text_logits = self.text_classifier(shared_features)
        image_logits = self.image_classifier(shared_features)
        multi_logits = self.multi_classifier(shared_features)
        logging.debug("Obtained logits from classification heads.")
        
        # Combine logits for final prediction
        final_logits = torch.zeros((shared_features.size(0), 4), device=shared_features.device)
        final_logits[:, 0] = multi_logits[:, 1]  # multi-sarcasm
        final_logits[:, 1] = text_logits[:, 1]   # text-sarcasm
        final_logits[:, 2] = image_logits[:, 1]  # image-sarcasm
        final_logits[:, 3] = 1 - (multi_logits[:, 1] + text_logits[:, 1] + image_logits[:, 1]).clamp(0, 1)  # not-sarcasm
        logging.debug("Combined logits for final prediction.")
        
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(final_logits, labels)
            logging.debug("Computed loss.")
            
        return {'loss': loss, 'logits': final_logits} if loss is not None else final_logits

logging.info("sarcasm_models.py loaded successfully.")
