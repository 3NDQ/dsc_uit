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
            combined_image_features, text_features, text_features2, ocr_features, image_features, labels = batch
            combined_image_features = combined_image_features.to(device)
            text_features = text_features.to(device)
            text_features2 = text_features2.to(device)
            ocr_features = ocr_features.to(device)
            image_features = image_features.to(device)
            labels = labels.to(device)

            device_type = "cuda" if torch.cuda.is_available() else "cpu"

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device_type):
                outputs = model(
                    combined_image_features=combined_image_features,
                    text_features=text_features,
                    text_features2= text_features2,
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

def run_train(train_features_dir, train_features_dir2, device, num_epochs, patience, batch_size, num_workers,
              learning_rate, val_size, random_state, fusion_method, gamma,
              loss_type, label_smoothing):
    logging.info("Starting training and evaluation...")

    # Load pre-extracted features
    train_combined_image_features = np.load(os.path.join(train_features_dir, "combined_image_features.npy"))
    train_text_features = np.load(os.path.join(train_features_dir, "text_features.npy"))
    train_text_features2 = np.load(os.path.join(train_features_dir2, "text_features.npy"))
    train_ocr_features = np.load(os.path.join(train_features_dir, "ocr_features.npy"))
    train_image_features = np.load(os.path.join(train_features_dir, "image_features.npy"))
    # Load labels
    with open(os.path.join(train_features_dir, "labels.json"), "r") as f:
        train_labels_data = json.load(f)
    train_labels = [item["label_id"] for item in train_labels_data]

    # Compute Class Weights
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
    train_text_features2 = np.squeeze(np.array(train_text_features2))
    train_ocr_features = np.squeeze(np.array(train_ocr_features))
    train_image_features = np.squeeze(np.array(train_image_features))

    # Split data into training and validation sets
    (
        train_combined_image_features,
        val_combined_image_features,
        train_text_features,
        val_text_features,
        train_text_features2,
        val_text_features2,
        train_ocr_features,
        val_ocr_features,
        train_image_features,
        val_image_features,
        train_labels,
        val_labels,
    ) = train_test_split(
        train_combined_image_features,
        train_text_features,
        train_text_features2,
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
        torch.tensor(train_text_features2, dtype=torch.float),
        torch.tensor(train_ocr_features, dtype=torch.float),
        torch.tensor(train_image_features, dtype=torch.float),
        torch.tensor(train_labels, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_combined_image_features, dtype=torch.float),
        torch.tensor(val_text_features, dtype=torch.float),
        torch.tensor(val_text_features2, dtype=torch.float),
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
        loss_type=loss_type,
        label_smoothing=label_smoothing
    ).to(device)
    logging.info('Model initialized and moved to device')

    # Train the model
    logging.info('Start training model...')
    model = train_model(
        model, train_dataloader, val_dataloader, device, num_epochs, patience, learning_rate
    )
    logging.info('Model training complete')