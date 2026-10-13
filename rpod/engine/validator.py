# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Validation: run the model on a dataset split and compute the metrics (optionally the plots).

Detections go through the same post-processing as predict() (top 100 per image) and are compared with the ground
truth in pixels of the original image.
"""
import contextlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import collate
from ..utils.metrics import evaluate
from ..utils.plots import GRID_IMAGES, save_plots
from .predictor import postprocess


@torch.no_grad()
def validate(net, dataset, names, device, batch=16, threshold=None, plots_dir=None, workers=0, amp_dtype=None):
    """threshold None: choose the best-F1 threshold (training validation). A number: use it (test, re-validation)."""
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate)
    was_training = net.training
    net.eval()
    preds, gts, scales, first = [], [], [], []
    for images, _, metas in loader:
        amp = torch.autocast(str(device).split(':')[0], dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
        with amp:
            out = net(images.to(device, non_blocking=True))
        # one copy per batch; the per-image work runs on the CPU (on the GPU every image would wait for it)
        logits, boxes = out['pred_logits'].float().cpu(), out['pred_boxes'].float().cpu()
        for j, meta in enumerate(metas):
            xyxy, scores, cls = postprocess(logits[j], boxes[j], meta['geom'], meta['orig_shape'], dataset.imgsz, 0.0, 100)
            pred = (xyxy.cpu().numpy(), scores.cpu().numpy(), cls.cpu().numpy())
            gt = dataset.ground_truth(meta['index'])
            preds.append(pred)
            gts.append(gt)
            scales.append(meta['geom'][0])          # object sizes measured at the model input
            if plots_dir is not None and len(first) < GRID_IMAGES:
                first.append((meta['index'], gt, pred))
    net.train(was_training)
    metrics = evaluate(preds, gts, names, threshold, scales)
    if plots_dir is not None:
        items = [(dataset.read(dataset.records[i]).image, gb, gc, pb, ps, pc) for i, (gb, gc), (pb, ps, pc) in first]
        save_plots(metrics, names, items, plots_dir)
        (Path(plots_dir) / 'metrics.json').write_text(json.dumps(
            {**metrics.to_dict(), 'per_class': metrics.per_class,
             'confusion_matrix': metrics.confusion_matrix.tolist()}, indent=2))
    return metrics
