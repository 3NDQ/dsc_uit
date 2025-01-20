# main.py
import os
import torch
import argparse
import sys
import logging
from run_train import run_train
from run_test import run_test

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("sarcasm_classifier.log")
        ],
        force=True
    )
    parser = argparse.ArgumentParser(description="Vietnamese Sarcasm Classifier")

    # Mode: train or test
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
    # Paths to pre-extracted features
    parser.add_argument('--train_features_dir', type=str, default='train_features', help='Directory containing pre-extracted training features')
    parser.add_argument('--test_features_dir', type=str, default='test_features', help='Directory containing pre-extracted testing features')

    # Model paths for testing
    parser.add_argument('--model_paths', type=str, nargs='+', default=['model_epoch_1.pth'], help='Paths to trained models')

    # Common arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate for the optimizer')
    parser.add_argument('--val_size', type=float, default=0.2, help='Val size for train test split')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'attention', 'cross_attention'], help='Method to fuse features: concat (default) or attention, cross_attention')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['focal', 'cross_entropy'], help='Loss type: focal or cross_entropy')

    # Hyperparameters for Focal Loss
    parser.add_argument('--gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss')
    parser.add_argument('--label_smoothing', type=float, default=0.15, help='Label smoothing value (e.g., 0.15)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if args.mode == 'train':
        run_train(
            train_features_dir=args.train_features_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fusion_method=args.fusion_method,
            num_epochs=args.num_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            val_size=args.val_size,
            random_state=args.random_state,
            gamma=args.gamma,
            loss_type=args.loss_type,
            label_smoothing=args.label_smoothing
        )
    elif args.mode == 'test':
        run_test(
            test_features_dir=args.test_features_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_paths=args.model_paths,
            fusion_method=args.fusion_method
        )

if __name__ == "__main__":
    main()
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
        ocr_feature_size = 1000
        combined_feature_size = 2024
        
        self.image_dense1 = nn.Linear(image_feature_size, 1024)
        self.image_dense2 = nn.Linear(1024, 512)

        self.ocr_dense1 = nn.Linear(ocr_feature_size, 1024)
        self.ocr_dense2 = nn.Linear(1024, 512)

        self.image_dense3 = nn.Linear(image_feature_size + 512 + 512, 1024)
        self.image_dense4 = nn.Linear(1024, 512)

        self.text_dense1 = nn.Linear(text_feature_size, 512)
        self.text_dense3 = nn.Linear(512, 256)

        self.text_dense2 = nn.Linear(text_feature_size, 512)
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
                labels=None):
        logging.info(f'Shape: {image_features.shape}')
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
        
        text_out2 = self.text_dense2(text_features)
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
# run_train.py
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import evaluate_model
from sarcasm_model import VietnameseSarcasmClassifier
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from utils import EarlyStopping
from torch.cuda import amp
from tqdm import tqdm
import heapq
import os
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight

def train_model(model, train_dataloader, val_dataloader, device, num_epochs, patience, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)    
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    early_stopping = EarlyStopping(patience=patience)
    scaler = torch.amp.GradScaler()

    best_models = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in train_progress:
            combined_image_features, text_features, ocr_features, image_features, labels = batch
            combined_image_features = combined_image_features.to(device)
            text_features = text_features.to(device)
            ocr_features = ocr_features.to(device)
            image_features = image_features.to(device)
            labels = labels.to(device)

            device_type = "cuda" if torch.cuda.is_available() else "cpu"

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device_type):
                outputs = model(
                    combined_image_features=combined_image_features,
                    text_features=text_features,
                    ocr_features=ocr_features,
                    image_features=image_features,
                    labels=labels
                )
                loss, logits = outputs

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        logging.info(f"\n ####----EPOCH {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}----####")

        f1 = evaluate_model(model, val_dataloader, device)

        model_path = f"model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        if len(best_models) < 5:
            heapq.heappush(best_models, (f1, epoch, model_path))
            logging.info(f"Model saved at epoch {epoch+1}")
        else:
            if f1 > best_models[0][0]:
                _, _, filename_to_remove = heapq.heappop(best_models)
                if os.path.exists(filename_to_remove):
                    os.remove(filename_to_remove)

                heapq.heappush(best_models, (f1, epoch, model_path))
                logging.info(f"Model saved at epoch {epoch+1}")
            else:
                os.remove(model_path)
                logging.info(f"Model at epoch {epoch+1} discarded, not in top 5")

        early_stopping(avg_train_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered")
            break
        
    best_f1, best_epoch, best_model_file = max(best_models, key=lambda x: x[0])
    model.load_state_dict(torch.load(best_model_file))
    logging.info(f"Best model from epoch {best_epoch+1} with F1 score {best_f1:.4f} loaded.")

    return model

def run_train(train_features_dir, device, num_epochs, patience, batch_size, num_workers,
              learning_rate, val_size, random_state, fusion_method, gamma,
              loss_type, label_smoothing):
    logging.info("Starting training and evaluation...")

    # Load pre-extracted features
    train_combined_image_features = np.load(os.path.join(train_features_dir, "combined_image_features.npy"))
    train_text_features = np.load(os.path.join(train_features_dir, "text_features.npy"))
    train_ocr_features = np.load(os.path.join(train_features_dir, "ocr_features.npy"))
    train_image_features = np.load(os.path.join(train_features_dir, "image_features.npy"))

    # Load labels
    with open(os.path.join(train_features_dir, "labels.json"), "r") as f:
        train_labels_data = json.load(f)
    train_labels = [item["label_id"] for item in train_labels_data]

    # --- Compute Class Weights ---
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    # class_weights_normalized = class_weights  / class_weights.sum()
    # class_weights_normalized = 1 / class_weights_normalized
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    # Log class weights
    logging.info("Class Weights:")
    for i, weight in enumerate(class_weights):
        logging.info(f"  Class {i}: {weight:.4f}")

    # Convert to single NumPy array
    train_combined_image_features = np.squeeze(np.array(train_combined_image_features))
    train_text_features = np.squeeze(np.array(train_text_features))
    train_ocr_features = np.squeeze(np.array(train_ocr_features))
    train_image_features = np.squeeze(np.array(train_image_features))

    # Split data into training and validation sets
    (
        train_combined_image_features,
        val_combined_image_features,
        train_text_features,
        val_text_features,
        train_ocr_features,
        val_ocr_features,
        train_image_features,
        val_image_features,
        train_labels,
        val_labels,
    ) = train_test_split(
        train_combined_image_features,
        train_text_features,
        train_ocr_features,
        train_image_features,
        train_labels,
        test_size=val_size,
        stratify=train_labels,
        random_state=random_state,
    )
    logging.info("Finished splitting train/dev indices and features")

    # Create TensorDatasets
    train_dataset = TensorDataset(
        torch.tensor(train_combined_image_features, dtype=torch.float),
        torch.tensor(train_text_features, dtype=torch.float),
        torch.tensor(train_ocr_features, dtype=torch.float),
        torch.tensor(train_image_features, dtype=torch.float),
        torch.tensor(train_labels, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_combined_image_features, dtype=torch.float),
        torch.tensor(val_text_features, dtype=torch.float),
        torch.tensor(val_ocr_features, dtype=torch.float),
        torch.tensor(val_image_features, dtype=torch.float),
        torch.tensor(val_labels, dtype=torch.long),
    )
    logging.info("Finished creating train/dev datasets")

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logging.info('Finished loading DataLoaders')

    # Initialize model
    model = VietnameseSarcasmClassifier(
        mode="train",
        fusion_method=fusion_method,
        class_weight=class_weights_tensor,
        gamma=gamma,
        loss_type='cross_entropy',
        label_smoothing=label_smoothing
    ).to(device)
    logging.info('Model initialized and moved to device')

    # Train the model
    logging.info('Start training model...')
    model = train_model(
        model, train_dataloader, val_dataloader, device, num_epochs, patience, learning_rate
    )
    logging.info('Model training complete')
# run_test.py
import logging
import torch
import os
import json
from torch.utils.data import DataLoader, TensorDataset
from sarcasm_model import VietnameseSarcasmClassifier
from tqdm import tqdm
import numpy as np

def test_model(model, device, dataloader):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            combined_image_features, text_features, ocr_features, image_features = batch
            combined_image_features = combined_image_features.to(device)
            text_features = text_features.to(device)
            ocr_features = ocr_features.to(device)
            image_features = image_features.to(device)
            
            outputs = model(
                combined_image_features=combined_image_features,
                text_features=text_features,
                ocr_features=ocr_features,
                image_features=image_features,
            )
            logits = outputs
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions

def run_test(test_features_dir, device, batch_size, num_workers, model_paths, fusion_method):
    logging.info("Starting testing with multiple models...")

    # Load pre-extracted features
    test_combined_image_features = np.load(os.path.join(test_features_dir, "combined_image_features.npy"))
    test_text_features = np.load(os.path.join(test_features_dir, "text_features.npy"))
    test_ocr_features = np.load(os.path.join(test_features_dir, "ocr_features.npy"))
    test_image_features = np.load(os.path.join(test_features_dir, "image_features.npy"))

    # Convert to a single NumPy array
    test_combined_image_features = np.squeeze(np.array(test_combined_image_features))
    test_text_features = np.squeeze(np.array(test_text_features))
    test_ocr_features = np.squeeze(np.array(test_ocr_features))
    test_image_features = np.squeeze(np.array(test_image_features))
    
    # Create a TensorDataset
    test_dataset = TensorDataset(
        torch.tensor(test_combined_image_features, dtype=torch.float),
        torch.tensor(test_text_features, dtype=torch.float),
        torch.tensor(test_ocr_features, dtype=torch.float),
        torch.tensor(test_image_features, dtype=torch.float),
    )

    # Create DataLoader
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    logging.info('Finished loading Test DataLoader')

    # Initialize model
    model = VietnameseSarcasmClassifier(
        mode="test",
        fusion_method=fusion_method,
    ).to(device)

    # Load and test each model
    for idx, model_path in enumerate(model_paths):
        if not os.path.isfile(model_path):
            logging.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Model loaded from {model_path}")
        
        model.eval()
        logging.info(f'Model set to evaluation mode - {model_path}')
        
        predictions = test_model(model, device, test_dataloader)
        logging.info(f"Predictions generated successfully for model {idx+1}")
        
        id_to_label = {0: 'multi-sarcasm', 1: 'text-sarcasm', 2: 'image-sarcasm', 3: 'not-sarcasm'}
        predicted_labels = [id_to_label.get(pred, 'not-sarcasm') for pred in predictions]
        
        test_data_keys = list(range(len(predicted_labels)))
        
        results = {key: label for key, label in zip(test_data_keys, predicted_labels)}
        
        output = {
            "results": results,
            "phase": "test"
        }
        
        output_filename = f'results_model_{idx + 1}.json'
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logging.info(f"Predictions saved to {output_filename} for model {idx+1}")