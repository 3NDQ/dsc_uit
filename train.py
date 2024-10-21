# train.py 
import os
import torch
from transformers import get_linear_schedule_with_warmup
from utils import EarlyStopping
from torch.cuda import amp
import logging
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

        model_file = f"model_epoch_{epoch+1}.pth"
        if len(best_models) < 5:
            # If we don't have 5 models yet, always push the current one
            heapq.heappush(best_models, (f1, epoch, model_file))
            
            # Save the current model with the epoch number
            torch.save(model.state_dict(), model_file)
            logging.info(f"Model saved as {model_file} at epoch {epoch+1}")
            
        else:
            # Save the model only if its F1 score is better than the lowest one in the heap
            if f1 > best_models[0][0]:
                # Find the model being replaced (which has the lowest F1 score)
                _, replaced_epoch, replaced_model_file = heapq.heappop(best_models)
                
                # Delete the replaced model's file
                if os.path.exists(replaced_model_file):
                    os.remove(replaced_model_file)
                    logging.info(f"Model from epoch {replaced_epoch+1} deleted: {replaced_model_file}")
                
                # Push the new model into the top 5
                heapq.heappush(best_models, (f1, epoch, model_file))
                
                # Save the new model with the epoch number
                torch.save(model.state_dict(), model_file)
                logging.info(f"Model from epoch {epoch+1} saved, replacing the model from epoch {replaced_epoch+1}")

        
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
