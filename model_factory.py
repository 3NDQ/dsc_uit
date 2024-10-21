# model_factory.py

from transformers import AutoModel, AutoTokenizer
import logging

def get_tokenizer(tokenizer_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logging.info(f"Loaded tokenizer: {tokenizer_name}")
        return tokenizer
    except Exception as e:
        logging.error(f"Failed to load tokenizer '{tokenizer_name}': {e}")
        raise e

def get_text_encoder(model_name):
    try:
        text_encoder = AutoModel.from_pretrained(model_name)
        logging.info(f"Loaded text encoder: {model_name}")
        return text_encoder
    except Exception as e:
        logging.error(f"Failed to load text encoder '{model_name}': {e}")
        raise e

def get_image_encoder(model_name):
    try:
        image_encoder = AutoModel.from_pretrained(model_name)
        logging.info(f"Loaded image encoder: {model_name}")
        return image_encoder
    except Exception as e:
        logging.error(f"Failed to load image encoder '{model_name}': {e}")
        raise e

def get_multimodal_model(model_name):
    try:
        # Example for VisoBertModel; replace with actual multimodal model class
        from transformers import VisoBertModel  # Ensure this class exists or adjust accordingly
        multimodal_model = VisoBertModel.from_pretrained(model_name)
        logging.info(f"Loaded multimodal model: {model_name}")
        return multimodal_model
    except ImportError:
        logging.error("VisoBertModel not found in transformers. Ensure you have the correct class or adjust the import.")
        raise
    except Exception as e:
        logging.error(f"Failed to load multimodal model '{model_name}': {e}")
        raise e
