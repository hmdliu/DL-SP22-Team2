# Credits: https://github.com/facebookresearch/mae/blob/main/models_vit.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import timm.models.vision_transformer

from models_fpn import *

# from torchvision.ops.feature_pyramid_network_new import FeaturePyramidNetwork
# # from torchvision.ops.misc import Conv2dNormActivation
# from torchvision.models.detection.backbone_utils import *

class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

# ViTDet FPN by Kaiming He
# Reference: https://github.com/ViTAE-Transformer/ViTDet/blob/main/mmdet/models/backbones/vit.py
class MyFPN(nn.Module):
    def __init__(self, backbone, embed_dim=768, out_dim=256):
        super().__init__()
        self.backbone = backbone
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            Norm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fpn_head = FeaturePyramidNetwork(
            in_channels_list=[768, 768, 768, 768],
            out_channels=256,
            norm_layer=Norm2d,
            extra_blocks=LastLevelMaxPool()
        )

        self.out_channels = out_dim

        # self.inner_blocks = nn.ModuleList()
        # self.layer_blocks = nn.ModuleList()
        # for i in range(4):
        #     self.inner_blocks.append(Conv2dNormActivation(embed_dim, out_dim, kernel_size=1, padding=0, norm_layer=Norm2d, activation_layer=None))
        #     self.layer_blocks.append(Conv2dNormActivation(embed_dim, out_dim, kernel_size=3, norm_layer=Norm2d, activation_layer=None))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_uniform_(m.weight, a=1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        # forward backbone
        x = self.backbone(x)
        # forward fpn
        fp = OrderedDict()
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(ops)):
            fp[i] = ops[i](x)
        fp = self.fpn_head(fp)
        return fp

    def forward(self, x):
        x = self.forward_features(x)
        return x

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, embed_dim=768, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.out_channels = embed_dim
        self.head = nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)[:, 1:, :]  # remove cls token
        x = x.permute(0, 2, 1).reshape(B, -1, 32, 32)

        return x

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model