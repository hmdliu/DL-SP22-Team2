# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os
import sys
import pdb
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor

import models_vit
from engine_finetune import train_one_epoch, evaluate

import utils
import transforms as T
from dataset import UnlabeledDataset, LabeledDataset


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.Jitter())
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes, checkpoint_path):

    backbone = models_vit.__dict__['vit_base_patch16'](
        img_size=512,
        num_classes=num_classes,
        drop_path_rate=0.1
    )
    # test_img = torch.randn(1, 3, 224, 224)
    # test_out = backbone.forward_features(test_img)
    # print(test_out.size())

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % checkpoint_path)
    checkpoint_model = checkpoint['model']
    utils.interpolate_pos_embed(backbone, checkpoint_model)

    # load pre-trained model
    msg = backbone.load_state_dict(checkpoint_model, strict=False)
    assert len(msg.missing_keys) == 0
    # print(msg)

    backbone_with_fpn = models_vit.MyFPN(backbone)

    test_img = torch.randn(1, 3, 512, 512)
    test_out = backbone_with_fpn(test_img)
    print('Test FPN:', [v.size() for k, v in test_out.items()])

    model = FasterRCNN(
        backbone=backbone_with_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=AnchorGenerator(
            sizes=((16,), (32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        ),
        box_roi_pool=MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3", "pool"],
            output_size=8,
            sampling_ratio=2
        )
    )
    model.transform = GeneralizedRCNNTransform(
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        min_size=512,
        max_size=512,
        fixed_size=(512, 512),
    )

    return model

def main():
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(sys.argv) == 1:
        exp_id, num_epochs, checkpoint_path = 'test', 1, './checkpoints/pretrain-mae-base-80.pth'
    else:
        exp_id, num_epochs, checkpoint_path = sys.argv[1], int(sys.argv[2]), os.path.abspath(sys.argv[3])

    num_classes = 100
    train_dataset = LabeledDataset(root='/labeled', split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)
    print(f'exp_id={exp_id}; seed={seed}; batch_size=2; num_epochs={num_epochs}; device={device}.')

    # checkpoint_path = '/scratch/hl3797/DL-S2022/Deep-Learning-S22/checkpoints/mae-base-80.pth'
    model = get_model(num_classes, checkpoint_path)
    model_without_ddp = model
    model.to(device)
    print(model)

    optimizer = torch.optim.AdamW(utils.get_params(model, mode='decay'), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    # print('optimizer:', optimizer)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=500)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)

        # save checkpoint
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(to_save, os.path.abspath(f'./checkpoints/{exp_id}-{epoch}.pth'))

    print("That's it!")

if __name__ == "__main__":
    main()
