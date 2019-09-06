#!/bin/bash

#SBATCH --time=60:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=14
#SBATCH -p lyceum

source activate projenv
python /lyceum/gw2g17/project/code/main.py --model_type $MODEL --test_subject $SLURM_ARRAY_TASK_ID --experiment_num $EXPNUM --std_method $METHOD --lyceum
