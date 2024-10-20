!CUDA_LAUNCH_BLOCKING=1 python main.py \
    --mode train \
    --text_encoder "vinai/phobert-base" \
    --image_encoder "google/vit-base-patch16-224" \
    --tokenizer "vinai/phobert-base" \
    --train_json "/kaggle/input/vimmsd-training-dataset/vimmsd-train.json" \
    --train_image_folder "/kaggle/input/vimmsd-training-dataset/training-images/train-images" \
    --num_epochs 10 \
    --patience 5 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --val_size 0.2 \
    --random_state 42 \
    --num_workers 4 \
    --fusion_method 'concat' \
    --use_train_ocr_cache \
    --train_ocr_cache_path "train_ocr_cache.json"


!CUDA_LAUNCH_BLOCKING=1 python main.py \
    --mode test \
    --text_encoder "vinai/phobert-base" \
    --image_encoder "google/vit-base-patch16-224" \
    --tokenizer "vinai/phobert-base" \
    --test_json "/kaggle/input/vimmsd-training-dataset/vimmsd-public-test.json" \
    --test_image_folder "/kaggle/input/vimmsd-public-test/public-test-images/dev-images" \
    --model_path "/kaggle/input/model_temp/pytorch/default/1/model_epoch_1.pth" \
    --batch_size 8 \
    --num_workers 4 \
    --fusion_method 'concat' \
    --use_test_ocr_cache \
    --test_ocr_cache_path "test_ocr_cache.json"


