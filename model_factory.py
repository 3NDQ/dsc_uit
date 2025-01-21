from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, AutoModelForImageClassification
import logging

def get_tokenizer(tokenizer_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        logging.info(f"Loaded tokenizer: {tokenizer_name}")
        return tokenizer
    except Exception as e:
        logging.error(f"Failed to load tokenizer '{tokenizer_name}': {e}")
        raise e

def get_text_encoder(model_name):
    try:
        text_encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        logging.info(f"Loaded text encoder: {model_name}")
        return text_encoder
    except Exception as e:
        logging.error(f"Failed to load text encoder '{model_name}': {e}")
        raise e

def get_image_encoder(model_name):
    try:
        image_encoder = AutoModelForImageClassification.from_pretrained(model_name, trust_remote_code=True)
        logging.info(f"Loaded image encoder: {model_name}")
        return image_encoder
    except Exception as e:
        logging.error(f"Failed to load image encoder '{model_name}': {e}")
        raise e
        
def get_image_processor(processor_name):
    try:
        processor = AutoImageProcessor.from_pretrained(processor_name)
        logging.info(f"Loaded image processor: {processor_name}")
        return processor
    except Exception as e:
        logging.error(f"Failed to load image processor '{processor_name}': {e}")
        raise e