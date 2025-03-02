#!/bin/sh
#SBATCH -J verify_feat    
#SBATCH -p queue         
#SBATCH -N 1             
#SBATCH -n 48           
#SBATCH --array=0-26    


module load anaconda/3-2023.07-2-hxl
module load intel/18.0.4


env_name="gpr"

prop_id=$((SLURM_ARRAY_TASK_ID / 9))    
f_n=$((SLURM_ARRAY_TASK_ID % 9 + 1))   


conda run -n $env_name python3 verify_feature_eff.py $prop_id $f_n