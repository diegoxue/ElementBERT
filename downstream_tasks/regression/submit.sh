#!/bin/sh
#SBATCH -J feature_selection    
#SBATCH -p queue           
#SBATCH -N 1                 
#SBATCH -n 48            
#SBATCH --array=0           

module load anaconda/3-2023.07-2-hxl
module load intel/18.0.4

env_name="gpr"

f_n=4

conda run -n $env_name python3 verify_feature_eff.py ${SLURM_ARRAY_TASK_ID} $f_n