#!/bin/bash

## Team ID
#SBATCH --account=csci_ga_2572_2022sp_02

#SBATCH --job-name=torch
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=32GB

#SBATCH --time=1-00:00:00
#SBATCH --output=mae-day2.out
#SBATCH --error=mae-day2.err
#SBATCH --exclusive
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

# copy the dataset
cp -rp /scratch/DL22SP/unlabeled_224.sqsh /tmp
echo "Dataset is copied to /tmp"

# pretrain with default settings
singularity exec --nv \
--bind /scratch \
--overlay /scratch/hl3797/conda.ext3:ro \
--overlay /tmp/unlabeled_224.sqsh \
/share/apps/images/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
python main_pretrain.py \
        --resume ./output_dir/mae-day1/checkpoint-40.pth \
        --output_dir ./output_dir/mae-day2 \
        --log_dir ./output_dir/mae-day2
"