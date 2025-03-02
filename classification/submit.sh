#!/bin/sh
#SBATCH -J class_1    
#SBATCH -p queue        
#SBATCH -N 1             
#SBATCH -n 48           
#SBATCH --array=0-8     

module load miniforge/24.11
module load intel/18.0.4


f_n=$((SLURM_ARRAY_TASK_ID + 1))    
env_name="gpr"


conda run -n $env_name python3 verify_feature_eff.py $f_n
