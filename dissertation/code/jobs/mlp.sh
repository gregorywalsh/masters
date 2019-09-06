#!/bin/bash

#SBATCH --job-name=mlp_x_mm
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=14
#SBATCH -p lyceum

source activate projenv
python /lyceum/gw2g17/project/code/main.py --model_type mlp --experiment_num 500 --std_method minmax --lyceum
