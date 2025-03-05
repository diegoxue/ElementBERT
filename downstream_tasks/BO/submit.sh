#!/bin/sh
#SBATCH -J llm_features
#SBATCH -p queue
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --array=1-6 #

module load intel18u4

conda run -n xian_py310 python3 bo_botorch_grad_opt.py $(($SLURM_ARRAY_TASK_ID + 0))
conda run -n xian_py310 python3 bo_botorch_grad_opt.py $(($SLURM_ARRAY_TASK_ID + 1))
conda run -n xian_py310 python3 bo_botorch_grad_opt.py $(($SLURM_ARRAY_TASK_ID + 2))
conda run -n xian_py310 python3 bo_botorch_grad_opt.py $(($SLURM_ARRAY_TASK_ID + 3))
conda run -n xian_py310 python3 bo_botorch_grad_opt.py $(($SLURM_ARRAY_TASK_ID + 4))
conda run -n xian_py310 python3 bo_botorch_grad_opt.py $(($SLURM_ARRAY_TASK_ID + 5))
conda run -n xian_py310 python3 bo_botorch_grad_opt.py $(($SLURM_ARRAY_TASK_ID + 6))
conda run -n xian_py310 python3 bo_botorch_grad_opt.py $(($SLURM_ARRAY_TASK_ID + 7))
