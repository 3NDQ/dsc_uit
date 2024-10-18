!CUDA_LAUNCH_BLOCK=1 python main.py \
    --mode train \
    --text_encoder "uitnlp/visobert" \
    --image_encoder "google/vit-base-patch16-224" \
    --tokenizer "uitnlp/visobert" \
    --train_json "/kaggle/input/vimmsd-training-dataset/vimmsd-train.json" \
    --train_image_folder "/kaggle/input/vimmsd-training-dataset/training-images/train-images" \
    --num_epochs 10 \
    --patience 5 \
    --batch_size 16 \
    --num_workers 4 \
    --use_train_ocr_cache \
    --train_ocr_cache_path "train_ocr_cache.json"

!CUDA_LAUNCH_BLOCK=1 python main.py \
    --mode test \
    --text_encoder "uitnlp/visobert" \
    --image_encoder "google/vit-base-patch16-224" \
    --tokenizer "uitnlp/visobert" \
    --test_json "/kaggle/input/vimmsd-training-dataset/vimmsd-public-test.json" \
    --test_image_folder "/kaggle/input/vimmsd-training-dataset/public-test-images/dev-images" \
    --model_path "sarcasm_classifier_model.pth" \
    --batch_size 16 \
    --num_workers 4 \
    --use_test_ocr_cache \
    --test_ocr_cache_path "test_ocr_cache.json"

