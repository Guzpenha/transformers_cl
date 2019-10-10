#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_transformers/bin/activate

RANDOM_SEED=111
DATA_DIR=/tudelft.net/staff-umbrella/conversationalsearch/transformers_cl/pytorch_transformers_cl/data
BATCH_SIZE=32

TASK_NAME=faq
NUM_EPOCHS=2.0
LOGGING_STEPS=50

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
    --save_aps