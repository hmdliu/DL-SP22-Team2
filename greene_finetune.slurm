#!/bin/bash

#SBATCH --job-name=torch
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=20GB
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --requeue

# exp setting
exp_id=$1
num_epochs=$2
checkpoint_type=$3
checkpoint_path=$4

# env & dataset
ext3_path=/scratch/$USER/conda.ext3
dataset_path=/scratch/$USER/dataset/labeled.sqsh
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# finetune
singularity exec --nv --bind /scratch/$USER \
--overlay ${ext3_path}:ro \
--overlay ${dataset_path} \
${sif_path} \
/bin/bash -c "
source /ext3/env.sh
python main_finetune.py ${exp_id} ${num_epochs} ${checkpoint_type} ${checkpoint_path} > ${exp_id}.log 2>&1
"