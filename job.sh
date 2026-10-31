#!/bin/bash
#SBATCH --job-name=tema2_ia
#SBATCH --partition=ucsx
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt

apptainer exec --nv env.sif python main.py "$@"