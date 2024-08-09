#!/usr/bin/env bash
#SBATCH --job-name=c51_Montezuma
#SBATCH --output=runs/c51_Montezuma%j.log
#SBATCH --error=runs/c51_Montezuma%j.err
#SBATCH --mail-user=nair@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

srun python3 -u atari_c51.py \
 --task "MontezumaRevengeNoFrameskip-v4" --seed 3128 
