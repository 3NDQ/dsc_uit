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

def train_model(model, train_dataloader, val_dataloader, device, num_epochs, patience, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
            image_features, text_features, labels = batch
            image_features = image_features.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)
            
            device_type = "cuda" if torch.cuda.is_available() else "cpu"

            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device_type):
                outputs = model(
                    image_features=image_features,
                    text_features=text_features,
                    labels=labels
                )
                loss, logits = outputs

            scaler.scale(loss).backward()
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
              text_encoder, image_encoder, learning_rate, val_size, random_state, fusion_method):
    logging.info("Starting training and evaluation...")

    # Load pre-extracted features
    train_image_features = np.load(os.path.join(train_features_dir, "image_features.npy"))
    train_text_features = np.load(os.path.join(train_features_dir, "text_features.npy"))

    # Load labels
    with open(os.path.join(train_features_dir, "labels.json"), "r") as f:
        train_labels_data = json.load(f)
    train_labels = [item["label_id"] for item in train_labels_data]

    # Convert to single NumPy array
    train_image_features = np.squeeze(np.array(train_image_features))
    train_text_features = np.squeeze(np.array(train_text_features))

    # Split data into training and validation sets
    train_img_feats, val_img_feats, train_text_feats, val_text_feats, train_labels, val_labels = train_test_split(
        train_image_features, train_text_features, train_labels,
        test_size=val_size, stratify=train_labels, random_state=random_state
    )
    logging.info('Finished splitting train/dev indices and features')

    # Create TensorDatasets
    train_dataset = TensorDataset(
        torch.tensor(train_img_feats, dtype=torch.float),
        torch.tensor(train_text_feats, dtype=torch.float),
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_img_feats, dtype=torch.float),
        torch.tensor(val_text_feats, dtype=torch.float),
        torch.tensor(val_labels, dtype=torch.long)
    )
    logging.info('Finished creating train/dev datasets')

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logging.info('Finished loading DataLoaders')

    # Initialize model
    model = VietnameseSarcasmClassifier(
        mode="train",
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        fusion_method=fusion_method
    ).to(device)
    logging.info('Model initialized and moved to device')

    # Train the model
    logging.info('Start training model...')
    model = train_model(
        model, train_dataloader, val_dataloader, device, num_epochs, patience, learning_rate
    )
    logging.info('Model training complete')