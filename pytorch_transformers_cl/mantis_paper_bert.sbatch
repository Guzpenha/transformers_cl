#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=12000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

TASK_NAME=mantis_10
NUM_EPOCHS=1.0
BATCH_SIZE=32
LOGGING_STEPS=400
DATA_DIR=/tudelft.net/staff-umbrella/domaincss/wsdm19/bert_ranker/data

source /home/nfs/gustavopenha/env_transformers/bin/activate

for RANDOM_SEED in 10 100 1000 10000 100000
do
    srun python ./examples/run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $DATA_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=$BATCH_SIZE  \
        --per_gpu_train_batch_size=$BATCH_SIZE  \
        --learning_rate 2e-5 \
        --num_train_epochs $NUM_EPOCHS \
        --output_dir $DATA_DIR/${TASK_NAME}_output \
        --evaluate_during_training \
        --overwrite_output_dir \
        --logging_steps $LOGGING_STEPS \
        --eval_all_checkpoints \
        --seed $RANDOM_SEED \
        --percentage_data_by_epoch 0.40
done

TASK_NAME=mantis_50
for RANDOM_SEED in 10 100 1000 10000 100000
do
    srun python ./examples/run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $DATA_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=$BATCH_SIZE  \
        --per_gpu_train_batch_size=$BATCH_SIZE  \
        --learning_rate 2e-5 \
        --num_train_epochs $NUM_EPOCHS \
        --output_dir $DATA_DIR/${TASK_NAME}_output \
        --evaluate_during_training \
        --overwrite_output_dir \
        --logging_steps $LOGGING_STEPS \
        --eval_all_checkpoints \
        --seed $RANDOM_SEED \
        --percentage_data_by_epoch 0.2
done