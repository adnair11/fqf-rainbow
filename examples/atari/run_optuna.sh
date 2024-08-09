#!/usr/bin/env bash
#SBATCH --job-name=OptunaDB_SpaceInvaders
#SBATCH --output=runs/OptunaDB_SpaceInvaders%j.log
#SBATCH --error=runs/OptunaDB_SpaceInvaders%j.err
#SBATCH --mail-user=nair@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

srun python3 -u atari_fqf_rainbow_optuna.py 
