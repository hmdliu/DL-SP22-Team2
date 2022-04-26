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

if len(sys.argv) == 1:
    EXP_ID = 'test'
    NUM_EPOCHS = 1
    CHECKPOINT_TYPE = 'pretrained'
    CHECKPOINT_PATH = './checkpoints/pretrain-mae-base-80.pth'
else:
    EXP_ID = sys.argv[1]
    NUM_EPOCHS = int(sys.argv[2])
    CHECKPOINT_TYPE = sys.argv[3]
    CHECKPOINT_PATH = os.path.abspath(sys.argv[4])

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.Jitter())
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):

    backbone = models_vit.__dict__['vit_base_patch16'](
        img_size=512,
        num_classes=num_classes,
        drop_path_rate=0.1
    )
    # test_img = torch.randn(1, 3, 224, 224)
    # test_out = backbone.forward_features(test_img)
    # print(test_out.size())

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

    # load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    print(f'Load {CHECKPOINT_TYPE} checkpoint from: {CHECKPOINT_PATH}')
    checkpoint_model = checkpoint['model']
    utils.interpolate_pos_embed(model.backbone.backbone, checkpoint_model)
    if CHECKPOINT_TYPE == 'pretrained':
        msg = model.backbone.backbone.load_state_dict(checkpoint_model, strict=False)
    else:
        msg = model.load_state_dict(checkpoint_model)
        print('Loading status:', msg)
    assert len(msg.missing_keys) == 0    

    return model

def main():
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_classes = 100
    train_dataset = LabeledDataset(root='/labeled', split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)
    print(f'exp_id={EXP_ID}; seed={seed}; batch_size=2; num_epochs={NUM_EPOCHS}; device={device}.')

    model = get_model(num_classes)
    model_without_ddp = model
    model.to(device)
    print(model)

    optimizer = torch.optim.AdamW(utils.get_params(model, mode='decay'), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    # print('optimizer:', optimizer)

    for epoch in range(NUM_EPOCHS):
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
        torch.save(to_save, os.path.abspath(f'./checkpoints/{EXP_ID}-{epoch}.pth'))

    print("That's it!")

if __name__ == "__main__":
    main()
