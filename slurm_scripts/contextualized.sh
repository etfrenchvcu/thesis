#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# SBATCH --job-name=etfrench001
#SBATCH --mem=185161
# SBATCH --ntasks=<number_of_parallel-tasks>
# SBATCH --pty
# SBATCH --time=30
#SBATCH --qos=short

echo similarity $1
echo loss_function $2
echo epoch $3

source /home/etfrench/BioSyn/env/bin/activate
echo $PATH
echo $PYTHON_PATH
which python
python -V

# BASH COMMANDS TO RUN YOUR PROGRAM BELOW
MODEL_NAME_OR_PATH=dmis-lab/biobert-v1.1
OUTPUT_DIR=./tmp
DATA_DIR=./datasets/n2c2
TRAIN_DIR=${DATA_DIR}/processed_train
EVAL_DIR=${DATA_DIR}/processed_test
DICTIONARY=${DATA_DIR}/mrconso_dictionary.txt

# Train
start_time="$(date -u +%s)"
python train.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_dictionary_path ${DICTIONARY} \
    --train_dir ${TRAIN_DIR} \
    --eval_dir ${EVAL_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --device "cuda" \
    --topk 20 \
    --epoch $3 \
    --train_batch_size 16\
    --learning_rate 1e-5 \
    --max_length 300 \
    --contextualized \
    --similarity $1 \
    --loss_function $2 \
    --umls_path ./umls/processed/
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Training finished in $elapsed seconds"