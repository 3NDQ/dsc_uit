python extract_features.py \
    --data_path "/kaggle/input/vimmsd/vimmsd-train.json" \
    --image_folder "/kaggle/input/vimmsd/train-images" \
    --ocr_cache_path "/kaggle/input/ocr-cache/paddle_train_ocr_cache.json" \
    --image_model_name "google/vit-base-patch16-224" \
    --text_model_name "jinaai/jina-embeddings-v3"\
    --output_dir "train_features" \
    --mode "train"
    
CUDA_LAUNCH_BLOCKING=1 python main.py \
    --mode train \
    --train_features_dir "/kaggle/input/features-vit-jina-embedding-v3/train_features" \
    --batch_size 32 \
    --num_workers 4 \
    --fusion_method "concat" \
    --num_epochs 35 \
    --patience 10 \
    --learning_rate 3e-5 \
    --val_size 0.2 \
    --random_state 42 \
    --loss_type 'focal' \
    --label_smoothing 0

python extract_features.py \
    --data_path "/kaggle/input/vimmsd/vimmsd-private-test.json" \
    --image_folder "/kaggle/input/vimmsd/test-images" \
    --ocr_cache_path "/kaggle/input/ocr-cache/paddle_test_ocr_cache.json" \
    --image_model_name "google/vit-base-patch16-224" \
    --text_model_name "jinaai/jina-embeddings-v3" \
    --output_dir "test_features" \
    --mode "test"

CUDA_LAUNCH_BLOCKING=1 python main.py \
    --mode test \
    --test_features_dir "/kaggle/input/features-vit-jina-embedding-v3/test_features" \
    --batch_size 16 \
    --num_workers 4 \
    --model_paths "model_epoch_26.pth" "model_epoch_29.pth" "model_epoch_3.pth"\
    --fusion_method "concat"