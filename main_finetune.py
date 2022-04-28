# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os
import sys
import yaml
import random
import numpy as np
from addict import Dict

import torch
import torch.backends.cudnn as cudnn

import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import models_vit
from engine_finetune import train_one_epoch, evaluate

import utils
import transforms as T
from dataset import LabeledDataset

EXP_ID = sys.argv[1]

def init_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.Jitter())
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(args):

    # init backbone
    backbone = models_vit.__dict__['vit_base_patch16'](
        img_size=512,
        num_classes=args.num_classes,
        drop_path_rate=0.1
    )
    # test_img = torch.randn(1, 3, 224, 224)
    # test_out = backbone.forward_features(test_img)
    # print(test_out.size())

    # build fpn network
    backbone_with_fpn = models_vit.MyFPN(
        backbone=backbone,
        embed_dim=args.embed_dim,
        out_dim=args.out_dim,
        extra_pool=args.extra_pool
    )
    test_img = torch.randn(1, 3, 512, 512)
    test_out = backbone_with_fpn(test_img)
    print('Test FPN:', [v.size() for k, v in test_out.items()])

    # wrap up detector
    sizes = tuple((s,) for s in args.anchor_sizes)
    aspect_ratios = tuple(tuple(args.anchor_aspect_ratios) for i in range(len(sizes)))
    model = FasterRCNN(
        backbone=backbone_with_fpn,
        num_classes=args.num_classes,
        rpn_anchor_generator=AnchorGenerator(
            sizes=sizes,
            aspect_ratios=aspect_ratios
        ),
        box_roi_pool=MultiScaleRoIAlign(
            featmap_names=args.roi_align_feats,
            output_size=args.roi_align_output_size,
            sampling_ratio=args.roi_align_sampling_ratio
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
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    print(f'Load {args.checkpoint_type} checkpoint from: {args.checkpoint_path}')
    checkpoint_model = checkpoint['model']
    utils.interpolate_pos_embed(model.backbone.backbone, checkpoint_model)
    if args.checkpoint_type == 'pretrained':
        msg = model.backbone.backbone.load_state_dict(checkpoint_model, strict=False)
    else:
        msg = model.load_state_dict(checkpoint_model)
        print('Loading status:', msg)
    assert len(msg.missing_keys) == 0    

    return model

def main():
    
    # init config
    config_path = os.path.abspath(f'./configs/{EXP_ID}.yaml')
    assert os.path.isfile(config_path)
    args = Dict(yaml.safe_load(open(config_path)))
    print('Loading Args:')
    for k, v in args.items():
        print(f'[{k}]: {v}')

    # init training
    init_seed(args.seed)
    cuda_flag = torch.cuda.is_available() and (args.device == 'cuda')
    args.device = torch.device('cuda') if cuda_flag else torch.device('cpu')

    # init dataloaders
    train_dataset = LabeledDataset(root='/labeled', split="training", transforms=get_transform(train=True))
    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn
    )

    # init model
    model = get_model(args)
    model.to(args.device)
    print(model)

    # init optimizer & scheduler
    optimizer = torch.optim.AdamW(utils.get_params(model, mode=args.optim_mode), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    # print('optimizer:', optimizer)

    for epoch in range(1, args.num_epochs+1):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=args.device,
            epoch=epoch,
            print_freq=args.train_print_freq,
            warmup_iter=args.warmup_iter
        )
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=args.device)

        # save checkpoint
        if epoch >= args.export_bound:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(to_save, os.path.abspath(f'./checkpoints/{args.exp_id}-{epoch}.pth'))

    print("That's it!")

if __name__ == "__main__":
    main()
