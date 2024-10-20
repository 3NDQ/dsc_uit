# # train.py 
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from utils import EarlyStopping
from torch.cuda import amp
import logging
from tqdm import tqdm
import heapq  # Để lưu 5 mô hình tốt nhất
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

def train_model(model, train_dataloader, val_dataloader, device, num_epochs, patience, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    
    early_stopping = EarlyStopping(patience=patience)
    scaler = amp.GradScaler()  # For mixed precision training
    
    best_models = []  # Danh sách lưu 5 mô hình tốt nhất
    
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
        
        # Tính F1, Precision, Recall cho từng lớp và tổng thể
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        logging.info(f"Epoch {epoch+1}/{num_epochs} - F1 Score: {f1:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f}")
        
        # Check for improvement and save top 5 models
        if len(best_models) < 5:
            heapq.heappush(best_models, (-avg_val_loss, epoch, model.state_dict()))  # Lưu vào danh sách các mô hình tốt nhất
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            logging.info(f"Model saved at epoch {epoch+1}")
        else:
            # Lưu mô hình mới nếu nó tốt hơn mô hình kém nhất trong danh sách
            if -best_models[0][0] > avg_val_loss:
                heapq.heapreplace(best_models, (-avg_val_loss, epoch, model.state_dict()))
                torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
                logging.info(f"Model saved at epoch {epoch+1}")
        
        # Early stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered")
            break
    
    # Load the best model
    best_val_loss, best_epoch, best_model_state = min(best_models, key=lambda x: -x[0])
    model.load_state_dict(best_model_state)
    logging.info(f"Best model from epoch {best_epoch+1} loaded.")
    
    # Lưu toàn bộ mô hình nếu epoch < 5
    if num_epochs < 5:
        for epoch in range(num_epochs):
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    
    return model

