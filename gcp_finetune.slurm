#!/bin/bash

## Team ID
#SBATCH --account=csci_ga_2572_2022sp_02

#SBATCH --job-name=torch
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --exclusive
#SBATCH --requeue

# exp setting
exp_id=$1
num_epochs=$2
checkpoint_path=$3

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

# copy the dataset
cp -rp /scratch/DL22SP/labeled.sqsh /tmp
echo "Dataset is copied to /tmp"

# pretrain
singularity exec --nv \
--bind /scratch \
--overlay /scratch/hl3797/conda.ext3:ro \
--overlay /tmp/labeled.sqsh \
/share/apps/images/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
python main_finetune.py ${exp_id} ${num_epochs} ${checkpoint_path} > ${exp_id}.log 2>&1
"