#!/usr/bin/env bash
#SBATCH --job-name=FQF-Rainbow-pacman
#SBATCH --output=runs/FQF-Rainbow-pacman%j.log
#SBATCH --error=runs/FQF-Rainbow-pacman%j.err
#SBATCH --mail-user=nair@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:4

srun python3 -u atari_fqf_rainbow.py \
 --task "MsPacmanNoFrameskip-v4"
 