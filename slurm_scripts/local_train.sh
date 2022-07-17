#!/bin/bash

# BASH COMMANDS TO RUN YOUR PROGRAM BELOW
MODEL_NAME_OR_PATH=dmis-lab/biobert-v1.1
OUTPUT_DIR=./tmp/local
DATA_DIR=./datasets/development
TRAIN_DIR=${DATA_DIR}/processed_dev
DEV_DIR=${DATA_DIR}/processed_dev
TEST_DIR=${DATA_DIR}/processed_dev
DICTIONARY=${DATA_DIR}/dev_dictionary.txt

start_time="$(date -u +%s)"
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py \
    --batch_size 16 \
    --candidates 20 \
    --contextualized 0 \
    --dev_dir ${DEV_DIR} \
    --device "mps" \
    --dictionary_path ${DICTIONARY} \
    --epochs 2 \
    --loss_fn "similarity_nll" \
    --max_length 25 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --similarity_type "binary" \
    --test_dir ${TEST_DIR} \
    --train_dir ${TRAIN_DIR} \
    --umls_path ./umls/processed/
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Prediction finished in $elapsed seconds"
