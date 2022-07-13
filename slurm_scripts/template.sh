#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# SBATCH --job-name=etfrench001
#SBATCH --mem=185161
# SBATCH --ntasks=<number_of_parallel-tasks>
# SBATCH --pty
# SBATCH --time=30
#SBATCH --qos=short

echo model $1

source /home/etfrench/BioSyn/env/bin/activate
echo $PATH
echo $PYTHON_PATH
which python
python -V

# BASH COMMANDS TO RUN YOUR PROGRAM BELOW
MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=./tmp
DATA_DIR=./datasets/n2c2
EVAL_DIR=${DATA_DIR}/processed_test
DICTIONARY=${DATA_DIR}/dev_dictionary.txt

# Evaluate
start_time="$(date -u +%s)"
python eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dictionary_path ${DICTIONARY}  \
    --data_dir ${EVAL_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --device "cuda" \
    --topk 20 \
    --max_length 25 \
    --umls_path ./umls/processed/ \
    --save_predictions
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Executed in $elapsed seconds"