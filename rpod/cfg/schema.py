# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Model sizes and everything derived from them.

RP-OD has three sizes and nothing else to choose: classes come from the dataset and the input size is chosen when
training. Each size fixes the backbone, the depth of the neck and the decoder layers, with the balance of D-FINE's
S / M / L (the backbone grows with the size; neck and decoder are 256 channels wide in all of them).

| size | backbone   | neck fusion blocks | decoder layers | parameters (80 classes) | GFLOPs at 640 |
| s    | HGNetv2-B0 | 1                  | 3              | 9.8 M                   | 26            |
| m    | HGNetv2-B2 | 2                  | 4              | 16.0 M                  | 45            |
| l    | HGNetv2-B4 | 3                  | 6              | 33.1 M                  | 110           |
"""
from dataclasses import asdict, dataclass


class ConfigError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ fixed choices
# Backbones: timm model with ImageNet weights, and the width of the neck's fusion blocks that goes with it.
BACKBONES = {
    'hgnetv2-b0': dict(timm='hgnetv2_b0.ssld_stage1_in22k_in1k', expansion=0.5),
    'hgnetv2-b2': dict(timm='hgnetv2_b2.ssld_stage1_in22k_in1k', expansion=0.5),
    'hgnetv2-b4': dict(timm='hgnetv2_b4.ssld_stage1_in22k_in1k', expansion=1.0),
}


@dataclass(frozen=True)
class Architecture:
    """What a size fixes. Users choose a size; checkpoints store the architecture, so any checkpoint loads."""
    backbone: str
    neck_blocks: int
    decoder_layers: int


SIZES = {
    's': Architecture('hgnetv2-b0', 1, 3),
    'm': Architecture('hgnetv2-b2', 2, 4),
    'l': Architecture('hgnetv2-b4', 3, 6),
}
DEFAULT_SIZE = 'm'
HIDDEN_DIM = 256                        # every feature map and query
ATTENTION = dict(layers=1, heads=8, ffn_dim=1024)
DECODER = dict(heads=8, points=4, ffn_dim=1024)
DENOISING = dict(num=100, label_noise=0.5, box_noise=1.0)
MIN_QUERIES = 300


def model_config(size=DEFAULT_SIZE):
    if size not in SIZES:
        raise ConfigError(f"unknown model size {size!r}; RP-OD has {', '.join(map(repr, SIZES))}")
    return SIZES[size]


def size_of(arch):
    """The name of the standard size with this architecture, or None (a checkpoint made before the sizes existed)."""
    return next((name for name, a in SIZES.items() if a == arch), None)


def describe(arch):
    name = size_of(arch)
    parts = f'{arch.backbone}, {arch.neck_blocks} fusion block(s), {arch.decoder_layers} decoder layers'
    return f'size {name} ({parts})' if name else f'not a standard size ({parts})'


def to_dict(arch):
    return {'size': size_of(arch), **asdict(arch)}


def from_dict(data):
    """Checkpoint config -> Architecture. Also reads the knobs of checkpoints made before the sizes
    ({'backbone', 'neck': {'size'}, 'decoder': {'layers'}}, where neck size s / m / l meant 1 / 2 / 3 blocks)."""
    if 'neck' in data:
        arch = Architecture(data['backbone'], {'s': 1, 'm': 2, 'l': 3}[data['neck']['size']], data['decoder']['layers'])
    else:
        arch = Architecture(data['backbone'], data['neck_blocks'], data['decoder_layers'])
    if arch.backbone not in BACKBONES:
        raise ConfigError(f'backbone {arch.backbone!r} is not part of this version of RP-OD')
    return arch


# ------------------------------------------------------------------------------------------------ derived
def derive(arch, num_classes, num_queries=MIN_QUERIES):
    """Everything the network needs, computed from the architecture. Nothing here is user-facing."""
    levels = [3, 4, 5]                       # strides 8, 16, 32
    return dict(
        timm_name=BACKBONES[arch.backbone]['timm'], levels=levels, hidden_dim=HIDDEN_DIM,
        attn_layers=ATTENTION['layers'], attn_heads=ATTENTION['heads'], attn_ffn=ATTENTION['ffn_dim'],
        fusion_blocks=arch.neck_blocks, fusion_expansion=BACKBONES[arch.backbone]['expansion'],
        num_classes=num_classes, num_queries=max(MIN_QUERIES, num_queries),
        dec_layers=arch.decoder_layers, dec_heads=DECODER['heads'], dec_points=DECODER['points'],
        dec_ffn=DECODER['ffn_dim'], anchor_size=0.05,
        num_denoising=DENOISING['num'], label_noise=DENOISING['label_noise'], box_noise=DENOISING['box_noise'])


def queries_for(max_objects_per_image):
    """Candidates the decoder refines: at least 300, and at least 5x the busiest image of the dataset."""
    return max(MIN_QUERIES, 5 * int(max_objects_per_image))


def positions(levels, imgsz):
    """Candidate positions (tokens) the query selection chooses from, for an input of imgsz [height, width]."""
    return sum((imgsz[0] // 2 ** l) * (imgsz[1] // 2 ** l) for l in levels)


def check_input_size(levels, imgsz, num_queries):
    """The decoder keeps num_queries positions, so the input must have at least that many."""
    if positions(levels, imgsz) < num_queries:
        side = next(s for s in range(32, 100000, 32) if positions(levels, [s, s]) >= num_queries)
        raise ValueError(f'imgsz {imgsz[0]}x{imgsz[1]} gives {positions(levels, imgsz)} candidate positions but the '
                         f'decoder keeps {num_queries}; use a larger input (at least {side}x{side} for a square one)')
