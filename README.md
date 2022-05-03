# Deep Learning Final Competition - Team 2

**References: competition start code, [MAE](https://github.com/facebookresearch/mae), and [ViTDet](https://github.com/ViTAE-Transformer/ViTDet).**

---

## Group Members
- Hammond Liu
- Wenbin Qi
- Harry Lee

---

## Requisites
- Test Env: Python 3.9.7 (Singularity)
- Major Packages:
    - torch (1.11.0), torchvision (0.12.0)
    - timm (0.3.2) with a [manual fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842)
    - pycocotools, numpy, addict
- Env Path (on NYU GCP): /scratch/hl3797/conda.ext3

---

## Clone codebase
```
cd /scratch/$USER
git clone https://github.com/hmdliu/DL-SP22-Team2 && cd DL-SP22-Team2
```

---

## MAE Pre-training
By default, we use a per device batch size of 64 and train on 4 GPU for 80 epochs. The pre-training configs are specified in *main_pretrain.py*. Since jobs on GCP have a 24 hours time limit, *gcp_pretrain_day2.slurm* demonstrates how to resume from checkpoints.
```
# before you start: modify the account in the slurm scripts

# pre-training day 1 (~40 epochs)
sbatch gcp_pretrain_day1.slurm
# => Output logs: mae-day1.out & mae-day1.err
# => Output dir: ./output_dir/mae-day1

# pre-training day 2 (~40 epochs)
# by default, this loads: ./output_dir/mae-day1/checkpoint-40.pth
sbatch gcp_pretrain_day2.slurm
# => Output logs: mae-day2.out & mae-day2.err
# => Output dir: ./output_dir/mae-day2

# => final pre-train weights: ./output_dir/mae-day2/checkpoint-80.pth
# => expected train loss: ~0.27
```
