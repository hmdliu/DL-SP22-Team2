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
By default, we use a per device batch size of 64 and train on 4 GPUs. The pre-training configs are specified in *main_pretrain.py*. Since jobs on GCP have a 24 hours time limit, *gcp_pretrain_day2.slurm* demonstrates how to resume from checkpoints. In this competition, we pre-train for 80 epochs in total.
```
# switch to codebase root
cd cd /scratch/$USER/DL-SP22-Team2

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

# => final pre-trained weights: ./output_dir/mae-day2/checkpoint-80.pth
# => expected train loss: ~0.27
mkdir checkpoints
cp ./output_dir/mae-day2/checkpoint-80.pth ./checkpoints/pretrain-mae-base-80.pth
```

---

## ViTDet FPN + Faster R-CNN Fine-tuning
In the fine-tuning stage, we first freeze the backbone and train the newly appended FPN and detection head for 20 epochs. Then, we use a per-layer decayed learning rate and a strong jitter transform to further fine-tune the whole model. Again, due to the job time constraint on GCP, we resume from checkpoints do the second step iteratively.
```
# switch to codebase root
cd cd /scratch/$USER/DL-SP22-Team2

# before you start: modify the account in the slurm scripts

# fine-tuning with the backbone frozen
sbatch gcp_finetune.slurm ft_freeze
# => Output logs: ./ft_freeze.log
# => Checkpoints: ./checkpoints/ft_freeze-[epoch_num].pth

# pick the best checkpoint from [ft_freeze] and fill into ./configs/ft_resume_1.yaml
sbatch gcp_finetune.slurm ft_resume_1
# => Output logs: ./ft_resume_1.log
# => Checkpoints: ./checkpoints/ft_resume_1-[epoch_num].pth

# pick the best checkpoint from [ft_resume_1] and fill into ./configs/ft_resume_2.yaml
sbatch gcp_finetune.slurm ft_resume_2
# => Output logs: ./ft_resume_2.log
# => Checkpoints: ./checkpoints/ft_resume_2-[epoch_num].pth

# pick the best checkpoint from [ft_resume_2] and fill into ./configs/ft_resume_3.yaml
sbatch gcp_finetune.slurm ft_resume_3
# => Output logs: ./ft_resume_3.log
# => Checkpoints: ./checkpoints/ft_resume_3-[epoch_num].pth

# => final fine-tuned weights: ./checkpoints/ft_resume_3-[best_epoch].pth
# => expected best mAP for [ft_freeze]: ~0.151
# => expected best mAP for [ft_resume_1]: ~0.294
# => expected best mAP for [ft_resume_2]: ~0.314
# => expected best mAP for [ft_resume_3]: ~0.320
```

---

## Evaluation
```
# switch to codebase root
cd cd /scratch/$USER/DL-SP22-Team2

# before you start: modify the account in the slurm scripts

# set the checkpoint arg in ./configs/eval.yaml
sbatch gcp_finetune.slurm eval
# => Output logs: ./eval.log
```

---

## Supplementary Materials
You can find our training logs and checkpoints via [this link](https://drive.google.com/drive/folders/1Y1P4y313Ey0sdvBuDaPzXFD7oIMw6-hV?usp=sharing).
