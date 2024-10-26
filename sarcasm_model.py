# sarcasm_models.py
import torch
import torch.nn as nn
import logging

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
            # Cross-attention layers
            self.text_to_image_attention = nn.MultiheadAttention(embed_dim=self.image_encoder.config.hidden_size, num_heads=8)
            self.image_to_text_attention = nn.MultiheadAttention(embed_dim=self.text_encoder.config.hidden_size, num_heads=8)
            logging.info("Cross-Attention layers initialized for both text-to-image and image-to-text.")
        
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
        self.text_classifier = nn.Linear(512, 2)
        self.image_classifier = nn.Linear(512, 2)
        self.multi_classifier = nn.Linear(512, 2)
        logging.info("Classification heads initialized.")
        
    def forward(self, image, input_ids, attention_mask, labels=None):
        logging.debug("Forward pass started.")
        # Image encoding
        image_outputs = self.image_encoder(image)
        image_features = image_outputs.last_hidden_state[:, 0, :]  # Use CLS token for image
        
        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # Use CLS token for text
        
        if self.fusion_method == 'cross_attention':
            # Text-to-Image Attention
            image_features_expanded = image_features.unsqueeze(1)  # Add sequence dimension for attention
            text_features_expanded = text_features.unsqueeze(1)    # Add sequence dimension for attention
            
            attended_text, _ = self.text_to_image_attention(
                query=text_features_expanded, key=image_features_expanded, value=image_features_expanded
            )
            attended_image, _ = self.image_to_text_attention(
                query=image_features_expanded, key=text_features_expanded, value=text_features_expanded
            )
            
            attended_text = attended_text.squeeze(1)  # Remove sequence dimension
            attended_image = attended_image.squeeze(1)  # Remove sequence dimension
            
            # Combine attended features by concatenation
            combined_features = torch.cat((attended_text, attended_image), dim=1)
            logging.debug("Applied cross-attention to combine image and text features.")
            shared_features = self.projector(combined_features)
        elif self.fusion_method == 'attention':
            # Self-attention (previously defined)
            combined_features = torch.cat((image_features, text_features), dim=1).unsqueeze(0)
            attended_features, _ = self.self_attention(combined_features, combined_features, combined_features)
            attended_features = attended_features.squeeze(0)
            shared_features = self.projector(attended_features)
        else:
            # Concatenate features (default)
            combined_features = torch.cat((image_features, text_features), dim=1)
            shared_features = self.projector(combined_features)
        
        # Classification heads
        text_logits = self.text_classifier(shared_features)
        image_logits = self.image_classifier(shared_features)
        multi_logits = self.multi_classifier(shared_features)
        
        # Final logits
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

