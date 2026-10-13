# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The COCO numbers of a checkpoint as papers report them: pycocotools on the official instances_val2017.json (crowd
regions ignored, up to 100 detections per image, AP by object size).

--imgsz evaluates at another input size (the network has no size-specific weights) and --queries keeps another number
of candidates for the decoder; both without retraining. APs50 / APs75: small objects at IoU 0.5 and 0.75.

    python eval_coco.py runs/train/rpod_s/weights/best.pt
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from rpod import RPOD  # noqa: E402
from rpod.cfg.schema import describe  # noqa: E402
from rpod.data import DetectionDataset, check_dataset, collate  # noqa: E402
from rpod.engine.predictor import postprocess  # noqa: E402

STATS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']


@torch.no_grad()
def evaluate(ckpt, data='coco/coco.yaml', ann='coco/annotations/instances_val2017.json', imgsz=None, queries=None,
             batch=32, workers=8):
    model = RPOD(ckpt)
    imgsz = [imgsz, imgsz] if isinstance(imgsz, int) else list(imgsz or model.imgsz)
    net = model.net.cuda().eval()
    if queries:
        net.decoder.num_queries = queries
    root = Path(data).resolve().parent
    cat_ids = json.loads((root / 'classes.json').read_text())['category_ids']
    ds = check_dataset(data)
    dataset = DetectionDataset(ds['records']['val'], imgsz, len(model.names))
    results, image_ids = [], []
    for images, _, metas in DataLoader(dataset, batch_size=batch, num_workers=workers, collate_fn=collate):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(images.cuda(non_blocking=True))
        for j, meta in enumerate(metas):
            xyxy, scores, cls = postprocess(out['pred_logits'][j].float(), out['pred_boxes'][j].float(), meta['geom'],
                                            meta['orig_shape'], imgsz, 0.0, 100)
            image_id = int(Path(meta['path']).stem)
            image_ids.append(image_id)
            for (x0, y0, x1, y1), s, c in zip(xyxy.tolist(), scores.tolist(), cls.tolist()):
                results.append({'image_id': image_id, 'category_id': cat_ids[c], 'bbox': [x0, y0, x1 - x0, y1 - y0],
                                'score': s})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(ann)
        ev = COCOeval(gt, gt.loadRes(results), 'bbox')
        ev.params.imgIds = image_ids          # the whole val2017, or the subset in the data file
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    stats = dict(zip(STATS, (round(100 * s, 1) for s in ev.stats)))
    prec = ev.eval['precision'][:, :, :, 1, 2]                   # [IoU, recall, class] for small objects, 100 detections
    for name, t in (('APs50', 0), ('APs75', 5)):
        p = prec[t]
        stats[name] = round(100 * float(p[p > -1].mean()), 1)
    params = sum(p.numel() for p in net.parameters()) / 1e6
    row = {'model': describe(model.cfg),
           'params_M': round(params, 1), 'imgsz': imgsz, 'queries': net.decoder.num_queries, **stats,
           'checkpoint': str(ckpt)}
    return row


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('--data', default='coco/coco.yaml')
    p.add_argument('--ann', default='coco/annotations/instances_val2017.json')
    p.add_argument('--imgsz', type=int, default=None)
    p.add_argument('--queries', type=int, default=None)
    p.add_argument('--batch', type=int, default=32)
    row = evaluate(**{k: v for k, v in vars(p.parse_args()).items()})
    print(json.dumps(row))
