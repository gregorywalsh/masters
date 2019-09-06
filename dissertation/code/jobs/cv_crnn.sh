#!/bin/bash

#SBATCH --job-name=crnn_c_mm
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=14
#SBATCH -p lyceum

source activate projenv
python /lyceum/gw2g17/project/code/crossval.py --model_type crnn --experiment_num 10 --std_method minmax --lyceum
