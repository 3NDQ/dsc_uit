# test.py (Ensure no redundant encoder/tokenizer initialization)

import torch
from tqdm import tqdm
def test_model(model, device, dataloader):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    return predictions
