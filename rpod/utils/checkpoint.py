"""Checkpoints and partial weight loading.

A checkpoint carries everything needed to use the model: weights, architecture, class names, input size, number of
queries and the recommended confidence threshold.

Partial loading matches weights by name and shape, so a pretrained model can initialize any combination of knobs:
a smaller neck takes the first fusion blocks, fewer decoder layers take the first layers, and only what does not fit
(other classes) keeps its fresh initialization.
"""
import datetime

import torch

from ..cfg.schema import derive, from_dict, to_dict

FORMAT = 2     # 2: boxes as edge distributions (D-FINE); 1: RP-OD <= ba7e126


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
        raise ValueError(f'{path} is not an RP-OD checkpoint (or comes from an incompatible version'
                         + (': format 1, trained before the box head became edge distributions; use RP-OD <= ba7e126)'
                            if ckpt.get('rpod_format') == 1 else ')'))
    ckpt['config'] = from_dict(ckpt['config'])
    ckpt['names'] = {int(k): v for k, v in ckpt['names'].items()}
    return ckpt


def build_from_checkpoint(ckpt):
    from ..nn import Detector
    net = Detector(derive(ckpt['config'], len(ckpt['names']), ckpt['num_queries']), pretrained_backbone=False)
    net.load_state_dict(ckpt['state_dict'])
    return net


def load_weights(net, state_dict):
    """Load whatever fits into `net`; returns a report of what was loaded and what kept its fresh initialization."""
    own = net.state_dict()
    new, reinit = {}, []
    for k, v in own.items():
        s = state_dict.get(k)
        if s is not None and s.shape == v.shape:
            new[k] = s
        else:
            reinit.append(k)
    net.load_state_dict({**own, **new})
    total = sum(v.numel() for v in own.values())
    loaded = sum(own[k].numel() for k in new)
    groups = sorted({'.'.join(k.split('.')[:2]) for k in reinit})
    return {'loaded_fraction': loaded / total, 'reinitialized': groups,
            'summary': f"loaded {100 * loaded / total:.1f}% of the weights"
                       + (f" · kept fresh: {', '.join(groups)}" if groups else '')}
