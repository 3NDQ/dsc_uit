import os
import cv2
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from utils import focal_loss, dice_loss
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoTokenizer, AutoModel
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class CombinedSarcasmClassifier:
    def __init__(self, train_path, test_path, ocr_cache_train, ocr_cache_test):
        self.train_path = train_path
        self.test_path = test_path
        self.ocr_cache_train = ocr_cache_train
        self.ocr_cache_test = ocr_cache_test
        self.model = None
        self.vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.vit_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.jina_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.jina_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True, torch_dtype=torch.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.label_mapping = {
            'multi-sarcasm': 0, 
            'text-sarcasm': 1, 
            'image-sarcasm': 2, 
            'not-sarcasm': 3,
        }

        self.vit_model.to(self.device).to(torch.float32)
        self.jina_model.to(self.device).to(torch.float32)

    def preprocess_data(self, images, texts, is_test=0):
        image_features = []
        ocr_features = []
        total_images = len(images)

        input_json_file_path = self.ocr_cache_test if is_test else self.ocr_cache_train

        if os.path.exists(input_json_file_path):
            with open(input_json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            data = []
            for image_path, text in json_data.items():
                image_name = os.path.basename(image_path)
                data.append({"image_path": image_name, "ocr_text": text})
            df = pd.DataFrame(data)
            existing_images = df["image_path"].tolist()
            df["ocr_text"] = df["ocr_text"].fillna("").astype(str)
        else:
            raise FileNotFoundError(f"JSON file not found at {input_json_file_path}")

        for image in images:
            try:
                image_path = os.path.join(self.train_path if not is_test else self.test_path, image)
                img = cv2.imread(image_path)
                inputs = self.vit_processor(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    vit_outputs = self.vit_model(**inputs)
                vit_feature = vit_outputs.logits.cpu().numpy().squeeze()
                image_features.append(vit_feature)

                if image in existing_images:
                    ocr_text = df[df["image_path"] == image]["ocr_text"].values[0]
                else:
                    ocr_text = ""

                if ocr_text.strip():
                    ocr_inputs = self.jina_tokenizer(ocr_text, return_tensors="pt", padding="longest", truncation=True, max_length=512).to(self.device)
                    with torch.no_grad():
                        jina_outputs = self.jina_model(**ocr_inputs)
                    ocr_feature = jina_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    ocr_features.append(ocr_feature)
                else:
                    ocr_features.append(np.zeros(1024))

            except Exception as e:
                print(f"\nError processing image {image}: {str(e)}")
                image_features.append(np.zeros(1000))  # Assuming ViT outputs 1000 features
                ocr_features.append(np.zeros(1024))

        text_features = []
        for text in texts:
            try:
                inputs = self.jina_tokenizer(text, return_tensors="pt", padding="longest", truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    jina_outputs = self.jina_model(**inputs)
                jina_feature = jina_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                text_features.append(jina_feature)
            except Exception as e:
                print(f"\nError processing text: {str(e)}")
                text_features.append(np.zeros(1024))

        return np.array(image_features), np.array(ocr_features), np.array(text_features)

    def encode_labels(self, labels):
        numerical_labels = [self.label_mapping[label] for label in labels]
        return tf.keras.utils.to_categorical(numerical_labels, num_classes=len(self.label_mapping))

    def decode_labels(self, one_hot_labels):
        numerical_labels = np.argmax(one_hot_labels, axis=1)
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        return [reverse_mapping[idx] for idx in numerical_labels]

    def build(self, image_dim=1000, ocr_dim=1024, text_dim=1024):
        image_input = Input(shape=(image_dim,), name='image_input')
        ocr_input = Input(shape=(ocr_dim,), name='ocr_input')
        text_input = Input(shape=(text_dim,), name='text_input')

        image_dense = Dense(2048, activation='relu')(image_input)
        image_dropout = Dropout(0.2)(image_dense)
        image_dense2 = Dense(1024, activation='relu')(image_dropout)
        image_dropout2 = Dropout(0.2)(image_dense2)
        image_dense3 = Dense(512, activation='relu')(image_dropout2)

        ocr_dense = Dense(1024, activation='relu')(ocr_input)
        ocr_dropout = Dropout(0.2)(ocr_dense)
        ocr_dense2 = Dense(512, activation='relu')(ocr_dropout)

        text_dense = Dense(1024, activation='relu')(text_input)
        text_dropout = Dropout(0.2)(text_dense)
        text_dense2 = Dense(512, activation='relu')(text_dropout)

        combined = concatenate([image_dense3, ocr_dense2, text_dense2])

        dense_combined = Dense(1024, activation='relu')(combined)
        dropout_combined = Dropout(0.2)(dense_combined)
        dense_combined2 = Dense(512, activation='relu')(dropout_combined)
        dropout_combined2 = Dropout(0.2)(dense_combined2)
        dense_combined3 = Dense(256, activation='relu')(dropout_combined2)
        dropout_combined3 = Dropout(0.2)(dense_combined3)
        output = Dense(4, activation='softmax', name='output')(dropout_combined3)

        self.model = Model(inputs=[image_input, ocr_input, text_input], outputs=output)

    def learning_rate_schedule(self, epoch, lr):
        if 10 <= epoch < 50:
            return lr * 0.1
        elif 50 <= epoch < 75:
            return lr * 0.01
        elif 75 <= epoch:
            return lr * 0.001
        return lr

    def train(self, x_train_images, x_train_ocr, x_train_texts, y_train, loss_function='focal'):
        x_train_images, x_val_images, x_train_ocr, x_val_ocr, x_train_texts, x_val_texts, y_train, y_val = train_test_split(
            x_train_images, x_train_ocr, x_train_texts, y_train, test_size=0.2, stratify=y_train
        )
        unique, counts = np.unique(y_train, return_counts=True)
        print("Label distribution in training data:")
        print(dict(zip(unique, counts)))

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        print("Class weights:")
        for label, weight in class_weights_dict.items():
            print(f"{list(self.label_mapping.keys())[label]}: {weight:.4f}")

        print("Starting preprocessing for training data...")
        image_features_train, ocr_features_train, text_features_train = self.preprocess_data(x_train_images, x_train_texts)
        image_features_val, ocr_features_val, text_features_val = self.preprocess_data(x_val_images, x_val_texts, is_test=1)
        y_train_encoded = self.encode_labels(y_train)
        y_val_encoded = self.encode_labels(y_val)

        initial_lr = 1e-4

        if loss_function == 'focal':
            loss_fn = focal_loss(gamma=2.0, alpha=[0.1, 0.4, 0.2, 0.1])
        elif loss_function == 'dice':
            loss_fn = dice_loss()
        else:
            raise ValueError("Unsupported loss function. Choose 'focal' or 'dice'.")

        print(f"\nCompiling model with {loss_function} loss...")
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=initial_lr),
            loss=loss_fn,
            metrics=[tf.keras.metrics.F1Score(average="macro", threshold=0.7)]
        )

        class BatchProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                print(f"\nEpoch {epoch + 1} starting...")
            def on_batch_begin(self, batch, logs=None):
                print(f"Training batch {batch + 1}", end='\r')

        lr_scheduler = LearningRateScheduler(self.learning_rate_schedule)

        print("\nStarting training...")
        history = self.model.fit(
            [image_features_train, ocr_features_train, text_features_train], y_train_encoded,
            validation_data=([image_features_val, ocr_features_val, text_features_val], y_val_encoded),
            epochs=100, batch_size=40, class_weight=class_weights_dict,
            callbacks=[BatchProgressCallback(), lr_scheduler]
        )

        print("\nTraining completed!")
        return history

    def predict(self, x_test_images, x_test_texts):
        print("Preprocessing test data...")
        image_features, ocr_features, text_features = self.preprocess_data(x_test_images, x_test_texts, 1)
        print("Making predictions...")
        predictions = self.model.predict([image_features, ocr_features, text_features])
        return self.decode_labels(predictions)

    def load(self, model_file):
        self.model = load_model(model_file)

    def save(self, model_file):
        self.model.save(model_file)

    def summary(self):
        self.model.summary()