#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# SBATCH --job-name=etfrench001
#SBATCH --mem=185161
# SBATCH --ntasks=<number_of_parallel-tasks>
# SBATCH --pty
# SBATCH --time=30
#SBATCH --qos=short

# BASH COMMANDS TO RUN YOUR PROGRAM BELOW
MODEL_NAME_OR_PATH=$1
CONTEXTUALIZED=$2
MAX_LENGTH=$3

OUTPUT_DIR=./tmp
DATA_DIR=./datasets/datasets/n2c2
EVAL_DIR=${DATA_DIR}/processed_test
DICTIONARY=${DATA_DIR}/mrconso_dictionary.txt

echo model ${MODEL_NAME_OR_PATH}
echo contextualized ${CONTEXTUALIZED}
echo max_length ${MAX_LENGTH}

source /home/etfrench/BioSyn/env/bin/activate
echo $PATH
echo $PYTHON_PATH
which python
python -V

python predict.py \
    --candidates 5 \
    --contextualized ${CONTEXTUALIZED} \
    --data_dir ${EVAL_DIR} \
    --dictionary_path ${DICTIONARY} \
    --device "cuda" \
    --max_length ${MAX_LENGTH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --umls_path ./umls/processed/
