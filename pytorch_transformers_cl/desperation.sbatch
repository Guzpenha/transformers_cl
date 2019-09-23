#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
source /home/nfs/gustavopenha/env_transformers/bin/activate

# TASK_NAME=ms_v2
# NUM_EPOCHS=3.0
# BATCH_SIZE=32
# LOGGING_STEPS=200
# DATA_DIR=/tudelft.net/staff-umbrella/conversationalsearch/transformers_cl/pytorch_transformers_cl/data
# PACING_FUNCTION=root_2

# srun python ./examples/run_glue.py \
#     --model_type bert \
#     --model_name_or_path bert-base-uncased \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_eval \
#     --do_lower_case \
#     --data_dir $DATA_DIR/$TASK_NAME \
#     --max_seq_length 128 \
#     --per_gpu_eval_batch_size=$BATCH_SIZE  \
#     --per_gpu_train_batch_size=$BATCH_SIZE  \
#     --learning_rate 2e-5 \
#     --num_train_epochs $NUM_EPOCHS \
#     --output_dir $DATA_DIR/${TASK_NAME}_output \
#     --evaluate_during_training \
#     --overwrite_output_dir \
#     --logging_steps $LOGGING_STEPS \
#     --curriculum_file $DATA_DIR/$TASK_NAME/${TASK_NAME}_random_c_3 \
#     --eval_all_checkpoints \
#     --seed $RANDOM_SEED \
#     --use_additive_cl \
#     --pacing_function $PACING_FUNCTION

# srun python ./examples/run_glue.py \
#     --model_type bert \
#     --model_name_or_path bert-base-uncased \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_eval \
#     --do_lower_case \
#     --data_dir $DATA_DIR/$TASK_NAME \
#     --max_seq_length 128 \
#     --per_gpu_eval_batch_size=$BATCH_SIZE  \
#     --per_gpu_train_batch_size=$BATCH_SIZE  \
#     --learning_rate 2e-5 \
#     --num_train_epochs $NUM_EPOCHS \
#     --output_dir $DATA_DIR/${TASK_NAME}_output \
#     --evaluate_during_training \
#     --overwrite_output_dir \
#     --logging_steps $LOGGING_STEPS \
#     --curriculum_file $DATA_DIR/$TASK_NAME/${TASK_NAME}_bert_preds_dif_c_3 \
#     --eval_all_checkpoints \
#     --seed $RANDOM_SEED \
#     --use_additive_cl \
#     --pacing_function $PACING_FUNCTION \
#     --invert_cl_values

# srun python ./examples/run_glue.py \
#     --model_type bert \
#     --model_name_or_path bert-base-uncased \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_eval \
#     --do_lower_case \
#     --data_dir $DATA_DIR/$TASK_NAME \
#     --max_seq_length 128 \
#     --per_gpu_eval_batch_size=$BATCH_SIZE  \
#     --per_gpu_train_batch_size=$BATCH_SIZE  \
#     --learning_rate 2e-5 \
#     --num_train_epochs $NUM_EPOCHS \
#     --output_dir $DATA_DIR/${TASK_NAME}_output \
#     --evaluate_during_training \
#     --overwrite_output_dir \
#     --logging_steps $LOGGING_STEPS \
#     --curriculum_file $DATA_DIR/$TASK_NAME/${TASK_NAME}_bert_avg_loss_c_3 \
#     --eval_all_checkpoints \
#     --seed $RANDOM_SEED \
#     --use_additive_cl \
#     --pacing_function $PACING_FUNCTION

TASK_NAME=mantis_10
NUM_EPOCHS=2.0
BATCH_SIZE=32
LOGGING_STEPS=600
DATA_DIR=/tudelft.net/staff-umbrella/conversationalsearch/transformers_cl/pytorch_transformers_cl/data
PACING_FUNCTION=root_2
PERCT_BY_EPOCH=1.0

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
    --curriculum_file $DATA_DIR/$TASK_NAME/${TASK_NAME}_random_c_3 \
    --eval_all_checkpoints \
    --seed $RANDOM_SEED \
    --use_additive_cl \
    --pacing_function $PACING_FUNCTION

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
    --curriculum_file $DATA_DIR/$TASK_NAME/${TASK_NAME}_bert_preds_dif_c_3 \
    --eval_all_checkpoints \
    --seed $RANDOM_SEED \
    --use_additive_cl \
    --pacing_function $PACING_FUNCTION \
    --invert_cl_values

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
    --curriculum_file $DATA_DIR/$TASK_NAME/${TASK_NAME}_bert_avg_loss_c_3 \
    --eval_all_checkpoints \
    --seed $RANDOM_SEED \
    --use_additive_cl \
    --pacing_function $PACING_FUNCTION
