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

# exp setting
exp_id=$1
num_epochs=$2
checkpoint_type=$3
checkpoint_path=$4

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
python main_finetune.py ${exp_id} ${num_epochs} ${checkpoint_type} ${checkpoint_path} > ${exp_id}.log 2>&1
"