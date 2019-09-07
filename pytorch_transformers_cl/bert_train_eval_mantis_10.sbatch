#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=28:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

TASK_NAME=mantis_10
NUM_EPOCHS=1
BATCH_SIZE=32
LOGGING_STEPS=200
DATA_DIR=/tudelft.net/staff-umbrella/domaincss/wsdm19/bert_ranker/data
PACING_FUNCTION=root_2
PERCT_BY_EPOCH=0.20

source /home/nfs/gustavopenha/env_transformers/bin/activate

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
    --pacing_function $PACING_FUNCTION \
    --percentage_data_by_epoch $PERCT_BY_EPOCH

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
    --curriculum_file $DATA_DIR/$TASK_NAME/${TASK_NAME}_turns_c_3 \
    --eval_all_checkpoints \
    --seed $RANDOM_SEED \
    --use_additive_cl \
    --pacing_function $PACING_FUNCTION \
    --percentage_data_by_epoch $PERCT_BY_EPOCH

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
    --curriculum_file $DATA_DIR/$TASK_NAME/${TASK_NAME}_utt_avg_words_c_3 \
    --eval_all_checkpoints \
    --seed $RANDOM_SEED \
    --use_additive_cl \
    --pacing_function $PACING_FUNCTION \
    --percentage_data_by_epoch $PERCT_BY_EPOCH

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
    --curriculum_file $DATA_DIR/$TASK_NAME/${TASK_NAME}_cand_docs_avg_words_c_3 \
    --eval_all_checkpoints \
    --seed $RANDOM_SEED \
    --use_additive_cl \
    --pacing_function $PACING_FUNCTION \
    --percentage_data_by_epoch $PERCT_BY_EPOCH

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
    --curriculum_file $DATA_DIR/$TASK_NAME/${TASK_NAME}max_dif_exact_match_q_d_c_3 \
    --eval_all_checkpoints \
    --seed $RANDOM_SEED \
    --use_additive_cl \
    --pacing_function $PACING_FUNCTION \
    --percentage_data_by_epoch $PERCT_BY_EPOCH

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
    --curriculum_file $DATA_DIR/$TASK_NAME/${TASK_NAME}max_dif_semantic_match_q_d_c_3 \
    --eval_all_checkpoints \
    --seed $RANDOM_SEED \
    --use_additive_cl \
    --pacing_function $PACING_FUNCTION \
    --percentage_data_by_epoch $PERCT_BY_EPOCH

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
    --curriculum_file $DATA_DIR/$TASK_NAME/${TASK_NAME}_bert_avg_pred_scores_c_3 \
    --eval_all_checkpoints \
    --seed $RANDOM_SEED \
    --use_additive_cl \
    --invert_cl_values \
    --pacing_function $PACING_FUNCTION \
    --percentage_data_by_epoch $PERCT_BY_EPOCH

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
    --pacing_function $PACING_FUNCTION \
    --percentage_data_by_epoch $PERCT_BY_EPOCH