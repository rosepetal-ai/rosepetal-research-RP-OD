"""The COCO numbers of a checkpoint as papers report them: pycocotools on the official instances_val2017.json (crowd
regions ignored, up to 100 detections per image, AP by object size).

--decoder-layers N evaluates the same weights keeping only the first N decoder layers, without retraining: every layer
was trained with its own loss, so each one is already a complete detector.

    python eval_coco.py runs/train/max_b2/weights/best.pt
    python eval_coco.py runs/train/max_b2/weights/best.pt --decoder-layers 3
"""
import argparse
import contextlib
import dataclasses
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
from rpod.cfg.schema import derive  # noqa: E402
from rpod.data import DetectionDataset, check_dataset, collate  # noqa: E402
from rpod.engine.predictor import postprocess  # noqa: E402
from rpod.nn import Detector  # noqa: E402
from rpod.utils.checkpoint import load_weights  # noqa: E402

STATS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']


@torch.no_grad()
def evaluate(ckpt, data='coco/coco.yaml', ann='coco/annotations/instances_val2017.json', decoder_layers=None, batch=32,
             workers=8):
    model = RPOD(ckpt)
    cfg = model.cfg
    if decoder_layers and decoder_layers != cfg.decoder.layers:
        if decoder_layers > cfg.decoder.layers:
            raise ValueError(f'the checkpoint has {cfg.decoder.layers} decoder layers')
        cfg = dataclasses.replace(cfg, decoder=dataclasses.replace(cfg.decoder, layers=decoder_layers))
        net = Detector(derive(cfg, len(model.names), model.num_queries), pretrained_backbone=False)
        load_weights(net, model.net.state_dict())
    else:
        net = model.net
    net = net.cuda().eval()
    root = Path(data).resolve().parent
    cat_ids = json.loads((root / 'classes.json').read_text())['category_ids']
    ds = check_dataset(data)
    dataset = DetectionDataset(ds['records']['val'], model.imgsz, len(model.names))
    results, image_ids = [], []
    for images, _, metas in DataLoader(dataset, batch_size=batch, num_workers=workers, collate_fn=collate):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(images.cuda(non_blocking=True))
        for j, meta in enumerate(metas):
            xyxy, scores, cls = postprocess(out['pred_logits'][j].float(), out['pred_boxes'][j].float(), meta['geom'],
                                            meta['orig_shape'], model.imgsz, 0.0, 100)
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
    params = sum(p.numel() for p in net.parameters()) / 1e6
    c = cfg
    row = {'model': f'{c.backbone} · neck {c.neck.size}{" · P2" if c.neck.small_objects else ""} · {c.decoder.layers} layers',
           'params_M': round(params, 1), **stats, 'checkpoint': str(ckpt)}
    return row


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('--data', default='coco/coco.yaml')
    p.add_argument('--ann', default='coco/annotations/instances_val2017.json')
    p.add_argument('--decoder-layers', type=int, default=None)
    row = evaluate(**{k: v for k, v in vars(p.parse_args()).items()})
    print(json.dumps(row))
