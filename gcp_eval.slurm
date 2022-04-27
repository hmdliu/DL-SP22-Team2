#!/bin/bash

## Team ID
#SBATCH --account=csci_ga_2572_2022sp_02

#SBATCH --job-name=torch
#SBATCH --partition=n1s8-v100-1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --time=1-00:00:00
#SBATCH --exclusive
#SBATCH --requeue

# evaluation
singularity exec --nv \
--bind /scratch \
--overlay /scratch/hl3797/conda.ext3:ro \
--overlay /scratch/DL22SP/labeled.sqsh \
/share/apps/images/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
python eval.py $1 > eval_$1.log 2>&1
"