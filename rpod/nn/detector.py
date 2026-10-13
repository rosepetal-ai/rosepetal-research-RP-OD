# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The whole network: backbone -> neck -> query selection + decoder. Built from `rpod.cfg.schema.derive(...)`."""
import torch.nn as nn

from .backbone import Backbone
from .decoder import Decoder
from .neck import Neck


class Detector(nn.Module):
    def __init__(self, d, pretrained_backbone=True):
        super().__init__()
        self.backbone = Backbone(d['timm_name'], d['levels'], pretrained_backbone)
        self.neck = Neck({l: self.backbone.channels[l] for l in d['levels']}, d['hidden_dim'], d['attn_layers'],
                         d['attn_heads'], d['attn_ffn'], d['fusion_blocks'], d['fusion_expansion'])
        self.decoder = Decoder(d['num_classes'], d['hidden_dim'], d['num_queries'], d['dec_heads'], len(d['levels']),
                               d['dec_points'], d['dec_layers'], d['dec_ffn'], 0.0, d['num_denoising'],
                               d['label_noise'], d['box_noise'], d['anchor_size'])

    def forward(self, images, targets=None):
        """images: [B, 3, H, W] in [0, 1]. Eval: {'pred_logits': [B, K, C], 'pred_boxes': [B, K, 4] cxcywh 0-1}."""
        return self.decoder(self.neck(self.backbone(images)), targets)

    def fuse(self):
        """Merge BatchNorms and RepVgg branches of the neck for export. Eval-mode outputs are unchanged."""
        self.neck.fuse()
        return self
