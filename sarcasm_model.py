import torch
import torch.nn as nn
import logging
from utils import FocalLoss  # Assuming you still want to use FocalLoss

class VietnameseSarcasmClassifier(nn.Module):
    def __init__(self,
                 class_weight=None,
                 num_labels=4,
                 dropout_rate=0.2,
                 gamma=2.0):
        super(VietnameseSarcasmClassifier, self).__init__()

        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
        self.gamma = gamma
        self.class_weight = class_weight

        # Image Processing Branch
        self.image_dense = nn.Linear(2024, 2048)
        self.image_dropout = nn.Dropout(dropout_rate)

        # Text Processing Branch
        self.text_dense1 = nn.Linear(1024, 1024)
        self.text_dense3 = nn.Linear(1024, 1024)
        self.text_dropout1 = nn.Dropout(dropout_rate)
        self.text_dropout2 = nn.Dropout(dropout_rate)
        self.text_dense2 = nn.Linear(1024, 512)
        self.text_dense4 = nn.Linear(1024, 512)

        # Fusion
        self.fusion_dense5 = nn.Linear(1024, 1024)
        self.fusion_dropout3 = nn.Dropout(dropout_rate)
        self.fusion_dense6 = nn.Linear(1024, 512)
        self.fusion_dropout4 = nn.Dropout(dropout_rate)
        self.fusion_dense7 = nn.Linear(512, 256)
        self.fusion_dropout5 = nn.Dropout(dropout_rate)

        # Output
        self.output_dense = nn.Linear(256, num_labels)

        # Loss Function
        logging.info(f"Using class_weight: {self.class_weight}")
        self.loss_fct = FocalLoss(gamma=self.gamma, alpha=self.class_weight) if self.class_weight is not None else FocalLoss(gamma=self.gamma)

    def forward(self, image_features, text_features, labels=None):
        # Image Branch
        image_out = self.image_dense(image_features)
        image_out = nn.ReLU()(image_out)
        image_out = self.image_dropout(image_out)

        # Text Branch
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
        
        # Concatenate
        concat_out = torch.cat((text_out, text_out_2), dim=1)

        # Fusion
        fusion_out = self.fusion_dense5(concat_out)
        fusion_out = nn.ReLU()(fusion_out)
        fusion_out = self.fusion_dropout3(fusion_out)
        fusion_out = self.fusion_dense6(fusion_out)
        fusion_out = nn.ReLU()(fusion_out)
        fusion_out = self.fusion_dropout4(fusion_out)
        fusion_out = self.fusion_dense7(fusion_out)
        fusion_out = nn.ReLU()(fusion_out)
        fusion_out = self.fusion_dropout5(fusion_out)

        # Output
        logits = self.output_dense(fusion_out)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits