#!/bin/bash

## Team ID
#SBATCH --account=csci_ga_2572_2022sp_02

#SBATCH --job-name=torch
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=32GB

#SBATCH --time=7-00:00:00
#SBATCH --output=mae-base-200.out
#SBATCH --error=mae-base-200.err
#SBATCH --exclusive
#SBATCH --requeue

# pretrain
singularity exec --nv \
--bind /scratch \
--overlay /scratch/hl3797/conda.ext3:ro \
--overlay /scratch/DL22SP/unlabeled_224.sqsh \
--overlay /scratch/DL22SP/labeled.sqsh \
/share/apps/images/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "source /ext3/env.sh; python main_pretrain.py --epochs 1 --warmup_epochs 0"