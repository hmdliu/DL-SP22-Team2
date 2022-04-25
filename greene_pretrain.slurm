#!/bin/bash

#SBATCH --job-name=torch
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=32GB
#SBATCH --time=5-00:00:00
#SBATCH --output=mae-base-200.out
#SBATCH --error=mae-base-200.err
#SBATCH --exclusive
#SBATCH --requeue

# env & dataset
ext3_path=/scratch/$USER/conda.ext3
dataset_path=/scratch/$USER/dataset/unlabeled_224.sqsh
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# pretrain
singularity exec --nv --bind /scratch/$USER \
--overlay ${ext3_path}:ro \
--overlay ${dataset_path} \
${sif_path} \
/bin/bash -c "
source /ext3/env.sh
python main_pretrain.py
"