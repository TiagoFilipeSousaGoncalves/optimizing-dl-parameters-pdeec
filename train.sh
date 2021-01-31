#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o fich-%j.out
#SBATCH -e fich-%j.err

# other options you might want
python code/run_genetic_algorithm.py
# etc
