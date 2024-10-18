# train.py 

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from utils import EarlyStopping
from torch.cuda import amp
import logging
from tqdm import tqdm

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=20, patience=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    
    early_stopping = EarlyStopping(patience=patience)
    scaler = amp.GradScaler()  # For mixed precision training
    
    best_val_loss = float('inf')
    best_model = None
    
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
        
        with torch.no_grad():
            val_progress = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", leave=False)
            for batch in val_progress:
                batch = {k: v.to(device) for k, v in batch.items()}
                with amp.autocast():
                    outputs = model(**batch)
                    loss = outputs['loss']
                val_loss += loss.item()
                val_progress.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {avg_val_loss:.4f}")
        
        # Check for improvement
        early_stopping(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            logging.info("Validation loss improved. Saving best model.")
        
        # Early stopping check
        if early_stopping.early_stop:
            logging.info("Early stopping triggered")
            break
    
    # Load best model if available
    if best_model is not None:
        model.load_state_dict(best_model)
        logging.info("Best model loaded.")
    
    return model
