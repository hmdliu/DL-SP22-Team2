
# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os
import sys
import torch
import numpy as np

import utils
import models_vit
from dataset import LabeledDataset
from main_finetune import *
from engine_finetune import evaluate

def get_eval_model(num_classes, eval_checkpoint_path):

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
        # rpn_anchor_generator=AnchorGenerator(
        #     # sizes=((128, 64), (64, 32), (32, 16), (16, 8), (8, 4)),
        #     sizes=((512,), (256,), (128,), (64,), (32,)),
        #     aspect_ratios=((0.5, 1.0, 2.0),) * 5
        # ),
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

    checkpoint = torch.load(eval_checkpoint_path, map_location='cpu')
    print("Load fine-tuned checkpoint from: %s" % eval_checkpoint_path)
    checkpoint_model = checkpoint['model']

    # load fine-tuned model
    msg = model.load_state_dict(checkpoint_model)
    assert len(msg.missing_keys) == 0
    print('Loading status:', msg)

    return model

def main():
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_classes = 100
    exp_id, checkpoint_path = sys.argv[1], os.path.abspath(sys.argv[2])
    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)
    print(f'[Eval]: exp_id={exp_id}; seed={seed}; batch_size=2; device={device}.')

    # model = get_model(num_classes, checkpoint_path)
    model = get_eval_model(num_classes, os.path.abspath(checkpoint_path))
    model.to(device)
    print(model)

    print('Evaluation Start:')
    evaluate(model, valid_loader, device=device)
    print('That\'s it!')

if __name__ == "__main__":
    main()
