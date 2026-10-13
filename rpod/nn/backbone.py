"""Backbone: a timm network with ImageNet weights, returning {level: feature map} (stride = 2^level).

Images come in as float tensors in [0, 1], shape [B, 3, H, W]. The ImageNet normalization the weights expect
happens inside, so it is part of the exported model and the serving side only has to divide by 255.
"""
import timm
import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self, timm_name, levels, pretrained=True):
        super().__init__()
        probe = timm.create_model(timm_name.split('.')[0], pretrained=False, features_only=True)
        stride_to_index = {r: i for i, r in enumerate(probe.feature_info.reduction())}
        missing = [l for l in levels if 2 ** l not in stride_to_index]
        if missing:
            raise ValueError(f'{timm_name} has no feature map for levels {missing}')
        out_indices = [stride_to_index[2 ** l] for l in levels]
        self.body = timm.create_model(timm_name, pretrained=pretrained, features_only=True, out_indices=out_indices)
        self.levels = list(levels)
        self.channels = dict(zip(self.levels, self.body.feature_info.channels()))
        cfg = self.body.pretrained_cfg
        self.register_buffer('mean', torch.tensor(cfg.get('mean', (0.485, 0.456, 0.406))).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(cfg.get('std', (0.229, 0.224, 0.225))).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return dict(zip(self.levels, self.body((x - self.mean) / self.std)))
