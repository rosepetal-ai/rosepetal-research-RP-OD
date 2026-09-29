"""Checkpoints and partial weight loading.

A checkpoint carries everything needed to use the model: weights, architecture, class names, input size, number of
queries and the recommended confidence threshold.

Partial loading matches weights by name and shape, so a pretrained model can initialize any combination of knobs:
the neck modules are named by level, extra or missing decoder layers are simply skipped, and the deformable attention
is re-sliced when the number of feature levels changes (a new level starts as a copy of its nearest neighbour).
"""
import datetime

import torch

from ..cfg.schema import DECODER, derive, from_dict, to_dict

FORMAT = 1
_LEVEL_SLICED = ('cross_attn.sampling_offsets.', 'cross_attn.attention_weights.')


def save_checkpoint(path, net, cfg, names, imgsz, num_queries, threshold=None, metrics=None, extra=None):
    ckpt = {'rpod_format': FORMAT, 'config': to_dict(cfg), 'names': {int(k): str(v) for k, v in names.items()},
            'imgsz': [int(imgsz[0]), int(imgsz[1])], 'num_queries': int(num_queries),
            'threshold': None if threshold is None else float(threshold), 'metrics': metrics or {},
            'date': datetime.datetime.now().isoformat(timespec='seconds'),
            'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()}}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('rpod_format') != FORMAT:
        raise ValueError(f'{path} is not an RP-OD checkpoint (or comes from an incompatible version)')
    ckpt['config'] = from_dict(ckpt['config'])
    ckpt['names'] = {int(k): v for k, v in ckpt['names'].items()}
    return ckpt


def build_from_checkpoint(ckpt):
    from ..nn import Detector
    net = Detector(derive(ckpt['config'], len(ckpt['names']), ckpt['num_queries']), pretrained_backbone=False)
    net.load_state_dict(ckpt['state_dict'])
    return net


def _reslice_levels(src, src_levels, dst_levels):
    """Re-slice a sampling_offsets / attention_weights tensor from src_levels to dst_levels (heads first, then levels)."""
    heads = DECODER['heads']
    shape = src.shape
    per_level = shape[0] // (heads * len(src_levels))
    s = src.reshape(heads, len(src_levels), per_level, *shape[1:])
    pick = [src_levels.index(l) if l in src_levels else min(range(len(src_levels)), key=lambda i: abs(src_levels[i] - l))
            for l in dst_levels]
    return s[:, pick].reshape(heads * len(dst_levels) * per_level, *shape[1:])


def load_weights(net, state_dict, src_levels=None, dst_levels=None):
    """Load whatever fits into `net`; returns a report of what was loaded and what kept its fresh initialization."""
    own = net.state_dict()
    new, reinit, adapted = {}, [], []
    for k, v in own.items():
        s = state_dict.get(k)
        if s is not None and s.shape == v.shape:
            new[k] = s
        elif s is not None and src_levels and dst_levels and any(t in k for t in _LEVEL_SLICED):
            new[k] = _reslice_levels(s, list(src_levels), list(dst_levels))
            adapted.append(k)
        else:
            reinit.append(k)
    net.load_state_dict({**own, **new})
    total = sum(v.numel() for v in own.values())
    loaded = sum(own[k].numel() for k in new)
    groups = sorted({'.'.join(k.split('.')[:2]) for k in reinit})
    return {'loaded_fraction': loaded / total, 'reinitialized': groups, 'adapted_levels': len(adapted),
            'summary': f"loaded {100 * loaded / total:.1f}% of the weights"
                       + (f" · re-sliced {len(adapted)} tensors for levels {list(dst_levels)}" if adapted else '')
                       + (f" · kept fresh: {', '.join(groups)}" if groups else '')}
