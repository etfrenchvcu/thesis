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
SIMILARITY=$4
LOSS_FN=$5
LR=$6
EXPERIMENT=$7

OUTPUT_DIR=./tmp/${EXPERIMENT}
DATA_DIR=./datasets/n2c2
TRAIN_DIR=${DATA_DIR}/processed_train
DEV_DIR=${DATA_DIR}/processed_dev
TEST_DIR=${DATA_DIR}/processed_test
DICTIONARY=${DATA_DIR}/mrconso_dictionary.txt

echo experiment ${EXPERIMENT}
echo model ${MODEL_NAME_OR_PATH}
echo contextualized ${CONTEXTUALIZED}
echo max_length ${MAX_LENGTH}
echo similarity ${SIMILARITY}
echo loss_fn ${LOSS_FN}
echo learning_rate ${LR}

source /home/etfrench/BioSyn/env/bin/activate
echo $PATH
echo $PYTHON_PATH
which python
python -V

python train.py \
    --batch_size 16 \
    --candidates 20 \
    --contextualized ${CONTEXTUALIZED} \
    --dev_dir ${DEV_DIR} \
    --device "cuda" \
    --dictionary_path ${DICTIONARY} \
    --epochs 30 \
    --loss_fn ${LOSS_FN} \
    --lr ${LR} \
    --max_length ${MAX_LENGTH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --similarity_type ${SIMILARITY} \
    --test_dir ${TEST_DIR} \
    --train_dir ${TRAIN_DIR} \
    --umls_path ./umls/processed/
