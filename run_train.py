# #run_train.py
import logging
import json
import torch
from torch.utils.data import DataLoader, Subset
from utils import evaluate_model
from process_datasets import TrainSarcasmDataset
from sarcasm_model import VietnameseSarcasmClassifier
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from utils import EarlyStopping
from torch.cuda import amp
from tqdm import tqdm
import heapq  
from sklearn.metrics import f1_score, precision_score, recall_score

def train_model(model, train_dataloader, val_dataloader, device, num_epochs, patience, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    
    early_stopping = EarlyStopping(patience=patience)
    scaler = amp.GradScaler()  # For mixed precision training
    
    best_models = []  # List to store the top 5 models based on F1 score
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch in train_progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            with amp.autocast():
                outputs = model(**batch)
                loss = outputs['loss']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_progress = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", leave=False)
            for batch in val_progress:
                batch = {k: v.to(device) for k, v in batch.items()}
                with amp.autocast():
                    outputs = model(**batch)
                    loss = outputs['loss']
                    logits = outputs['logits']
                val_loss += loss.item()
                val_progress.set_postfix(loss=loss.item())
                
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {avg_val_loss:.4f}")
        
        # Calculate F1, Precision, Recall for each class and overall
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        logging.info(f"Epoch {epoch+1}/{num_epochs} - F1 Score: {f1:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f}")
        
        # Check for improvement and save top 5 models based on F1 score
        if len(best_models) < 5:
            heapq.heappush(best_models, (f1, epoch, model.state_dict()))  # Store model with F1 score
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            logging.info(f"Model saved at epoch {epoch+1}")
        else:
            # Save the model if its F1 score is better than the lowest one in the heap
            if f1 > best_models[0][0]:
                heapq.heapreplace(best_models, (f1, epoch, model.state_dict()))
                torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
                logging.info(f"Model saved at epoch {epoch+1}")
        
        # Early stopping check based on validation loss
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered")
            break
    
    # Load the best model based on F1 score
    best_f1, best_epoch, best_model_state = max(best_models, key=lambda x: x[0])
    model.load_state_dict(best_model_state)
    logging.info(f"Best model from epoch {best_epoch+1} with F1 score {best_f1:.4f} loaded.")
    
    return model

def run_train(train_json, train_image_folder, tokenizer, device, 
                      num_epochs, patience, batch_size, num_workers, train_ocr_cache_path,
                      text_encoder, image_encoder, learning_rate, 
                      val_size, random_state, fusion_method, use_train_ocr_cache=False, active_ocr=True):
    logging.info("Starting training and evaluation...")
    
    # Create dataset with OCR caching parameters
    dataset = TrainSarcasmDataset(
        data_path=train_json, 
        image_folder=train_image_folder, 
        text_tokenizer=tokenizer, 
        use_ocr_cache=use_train_ocr_cache, 
        ocr_cache_path=train_ocr_cache_path,
        active_ocr=active_ocr
    )
    
    # Extract labels for stratified splitting
    try:
        labels = [dataset[i]['labels'].item() for i in range(len(dataset))]
    except Exception as e:
        logging.error(f"Failed to extract labels for stratified splitting: {e}")
        return
    
    # Split data into training and validation sets
    try:
        train_idx, val_idx = train_test_split(
            range(len(labels)), 
            test_size=val_size, 
            stratify=labels, 
            random_state=random_state
        )
        logging.info('Finished splitting train/dev indices')
    except Exception as e:
        logging.error(f"Failed to split data into train/dev sets: {e}")
        return
    
    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    logging.info('Finished creating train/dev sets')
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    logging.info('Finished loading DataLoaders')
    
    # Initialize model with passed encoders
    try:
        model = VietnameseSarcasmClassifier(text_encoder, image_encoder, fusion_method).to(device)
        logging.info('Model initialized and moved to device')
    except Exception as e:
        logging.error(f"Failed to initialize the model: {e}")
        return
    
    # Train the model
    model = train_model(
        model, 
        train_dataloader, 
        val_dataloader, 
        device, 
        num_epochs=num_epochs, 
        patience=patience,
        learning_rate=learning_rate
    )
    logging.info('Model training complete')
    
    # Save the trained model
    try:
        torch.save(model.state_dict(), 'sarcasm_classifier_model.pth')
        logging.info('Model saved as sarcasm_classifier_model.pth')
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
    
    # Evaluate the model on validation set
    try:
        evaluate_model(model, val_dataloader, device)
    except Exception as e:
        logging.error(f"Failed to evaluate model: {e}")
        return
    
    # Save OCR cache explicitly
    try:
        dataset.save_ocr_cache()
    except Exception as e:
        logging.error(f"Failed to save OCR cache: {e}")