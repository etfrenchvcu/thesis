#!/bin/bash

# BASH COMMANDS TO RUN YOUR PROGRAM BELOW
MODEL_NAME_OR_PATH=dmis-lab/biobert-v1.1
OUTPUT_DIR=./tmp/local
DATA_DIR=./datasets/development
EVAL_DIR=${DATA_DIR}/processed_dev
DICTIONARY=${DATA_DIR}/dev_dictionary.txt

start_time="$(date -u +%s)"
python predict.py \
    --candidates 1 \
    --data_dir ${EVAL_DIR} \
    --dictionary_path ${DICTIONARY} \
    --device "mps" \
    --max_length 25 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --umls_path ./umls/processed/
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Prediction finished in $elapsed seconds"
