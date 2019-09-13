sbatch --export=ALL,RANDOM_SEED=1 desperation.sbatch
sbatch --export=ALL,RANDOM_SEED=2 desperation.sbatch
sbatch --export=ALL,RANDOM_SEED=3 desperation.sbatch
sbatch --export=ALL,RANDOM_SEED=4 desperation.sbatch
sbatch --export=ALL,RANDOM_SEED=5 desperation.sbatch

sbatch --export=ALL,RANDOM_SEED=1 bert_train_eval_ms_v2.sbatch
sbatch --export=ALL,RANDOM_SEED=2 bert_train_eval_ms_v2.sbatch
sbatch --export=ALL,RANDOM_SEED=3 bert_train_eval_ms_v2.sbatch
sbatch --export=ALL,RANDOM_SEED=4 bert_train_eval_ms_v2.sbatch
sbatch --export=ALL,RANDOM_SEED=5 bert_train_eval_ms_v2.sbatch

sbatch --export=ALL,RANDOM_SEED=1 bert_train_eval_mantis_10.sbatch
sbatch --export=ALL,RANDOM_SEED=2 bert_train_eval_mantis_10.sbatch
sbatch --export=ALL,RANDOM_SEED=3 bert_train_eval_mantis_10.sbatch
sbatch --export=ALL,RANDOM_SEED=4 bert_train_eval_mantis_10.sbatch
sbatch --export=ALL,RANDOM_SEED=5 bert_train_eval_mantis_10.sbatch