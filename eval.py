
# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os
import sys
import yaml
import torch
from addict import Dict

import utils
from main_finetune import init_seed, get_model, get_transform
from dataset import LabeledDataset
from engine_finetune import evaluate

EXP_ID = sys.argv[1]

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
    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
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

    # evaluation
    print('Evaluation Start:')
    evaluate(model, valid_loader, device=args.device)
    print('That\'s it!')

if __name__ == "__main__":
    main()
