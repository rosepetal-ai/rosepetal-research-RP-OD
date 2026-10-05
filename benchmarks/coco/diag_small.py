"""Where small objects get lost, measured inside a trained model on COCO val2017 (no training).

1. Query selection: does one of the K candidates the decoder starts from cover each object (IoU >= 0.5)? Compared with
   the best of ALL tokens: if all tokens cover it but the top-K do not, the selection drops it; if no token does, the
   dense head never proposes it.
2. Levels: which pyramid level the selected candidates come from, and the one covering each small object.
3. Attention: share of the decoder's deformable attention that lands on each level, per layer, for all queries and
   for the queries that end up as small detections.
4. Final recall at IoU 0.5 (same class, top 100 detections, any score), per object size.

Sizes follow COCO on box area in the original image: small < 32², medium < 96², large otherwise.

    python diag_small.py runs/train/max_b2/weights/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from rpod import RPOD  # noqa: E402
from rpod.data import DetectionDataset, check_dataset, collate  # noqa: E402
from rpod.engine.predictor import postprocess  # noqa: E402
from rpod.utils.metrics import box_iou  # noqa: E402

SIZES = ('small', 'medium', 'large')


def size_of(boxes):
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return np.where(area < 32 ** 2, 0, np.where(area < 96 ** 2, 1, 2))


def cxcywh_to_xyxy(b):
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


@torch.no_grad()
def diagnose(ckpt, data='coco/coco.yaml', queries=None, batch=16, workers=8, limit=None):
    model = RPOD(ckpt)
    net = model.net.cuda().eval()
    dec = net.decoder
    if queries:
        dec.num_queries = queries
    levels = net.backbone.levels
    imgsz = model.imgsz
    ds = check_dataset(data)
    records = ds['records']['val'][:limit] if limit else ds['records']['val']
    dataset = DetectionDataset(records, imgsz, len(model.names))

    captured = {}
    hooks = [dec.register_forward_pre_hook(lambda m, a: captured.update(shapes=[(f.shape[2], f.shape[3]) for f in a[0]])),
             dec.enc_score_head.register_forward_hook(lambda m, a, o: captured.update(logits=o.float())),
             dec.enc_bbox_head.register_forward_hook(lambda m, a, o: captured.update(delta=o.float()))]
    attn = []
    for i, layer in enumerate(dec.layers):
        hooks.append(layer.cross_attn.attention_weights.register_forward_hook(
            lambda m, a, o, i=i: captured.setdefault('attn', {}).update({i: o.float()})))

    sel_cover = {s: [0, 0] for s in SIZES}       # covered by the selected top-K, total
    all_cover = {s: 0 for s in SIZES}            # covered by any token
    final_hit = {s: 0 for s in SIZES}
    sel_levels = np.zeros(len(levels))
    small_levels = np.zeros(len(levels))
    attn_all = np.zeros((len(dec.layers), len(levels)))
    attn_small = np.zeros((len(dec.layers), len(levels)))
    n_q_all = n_q_small = 0

    for images, _, metas in DataLoader(dataset, batch_size=batch, num_workers=workers, collate_fn=collate):
        captured.clear()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(images.cuda(non_blocking=True))
        shapes = captured['shapes']
        anchors, _ = dec._anchors(shapes, captured['delta'].device, torch.float32)
        boxes_all = (captured['delta'] + anchors).sigmoid()                         # [B, S, 4] cxcywh, canvas 0-1
        top = dec._select(captured['logits'].max(-1).values, shapes)                  # [B, K], as the model does
        level_of = np.concatenate([np.full(h * w, l) for l, (h, w) in zip(range(len(levels)), shapes)])
        np.add.at(sel_levels, level_of[top.cpu().numpy().ravel()], 1)
        H, W = imgsz
        for j, meta in enumerate(metas):
            gt, gcls = dataset.ground_truth(meta['index'])
            if not len(gt):
                continue
            scale, _, _, top_px, left_px = meta['geom']
            gt_canvas = (gt * scale + np.array([left_px, top_px, left_px, top_px])) / np.array([W, H, W, H])
            sizes = size_of(gt)
            b_all = boxes_all[j].cpu().numpy()
            finite = np.isfinite(b_all).all(1) & (b_all[:, 2] < 1) & (b_all[:, 3] < 1)
            iou_all = box_iou(gt_canvas, cxcywh_to_xyxy(b_all[finite]))
            sel = top[j].cpu().numpy()
            iou_sel = box_iou(gt_canvas, cxcywh_to_xyxy(b_all[sel]))
            xyxy, scores, cls = postprocess(out['pred_logits'][j].float(), out['pred_boxes'][j].float(), meta['geom'],
                                            meta['orig_shape'], imgsz, 0.0, 100)
            iou_fin = box_iou(gt, xyxy.cpu().numpy().astype(np.float64))
            same = gcls[:, None] == cls.cpu().numpy()[None]
            for g, s in enumerate(sizes):
                name = SIZES[s]
                sel_cover[name][1] += 1
                sel_cover[name][0] += int(iou_sel[g].max() >= 0.5)
                all_cover[name] += int(iou_all.shape[1] and iou_all[g].max() >= 0.5)
                final_hit[name] += int((iou_fin[g] * same[g] >= 0.5).any())
                if s == 0 and iou_sel[g].max() >= 0.5:
                    small_levels[level_of[sel[iou_sel[g].argmax()]]] += 1
            # attention per level: all queries, and those whose final box is a confident small detection
            final_boxes = out['pred_boxes'][j].float().cpu().numpy() * np.array([W, H, W, H]) / scale
            final_score = out['pred_logits'][j].float().sigmoid().max(-1).values.cpu().numpy()
            small_q = (final_boxes[:, 2] * final_boxes[:, 3] < 32 ** 2) & (final_score > 0.3)
            for i, a in captured['attn'].items():
                w = a[j].view(dec.num_queries, dec.layers[i].cross_attn.n_heads, -1).softmax(-1)
                w = w.view(dec.num_queries, dec.layers[i].cross_attn.n_heads, len(levels), -1).sum(-1).mean(1).cpu().numpy()
                attn_all[i] += w.sum(0)
                attn_small[i] += w[small_q].sum(0)
            n_q_all += dec.num_queries
            n_q_small += int(small_q.sum())
    for h in hooks:
        h.remove()

    names = [f'P{l}' for l in levels]
    return {
        'checkpoint': str(ckpt), 'imgsz': imgsz, 'queries': dec.num_queries, 'images': len(dataset),
        'objects': {s: sel_cover[s][1] for s in SIZES},
        'selected_top_k_cover': {s: round(sel_cover[s][0] / max(sel_cover[s][1], 1), 3) for s in SIZES},
        'any_token_cover': {s: round(all_cover[s] / max(sel_cover[s][1], 1), 3) for s in SIZES},
        'final_recall_iou50': {s: round(final_hit[s] / max(sel_cover[s][1], 1), 3) for s in SIZES},
        'selected_queries_by_level': dict(zip(names, np.round(sel_levels / sel_levels.sum(), 3).tolist())),
        'small_objects_covered_from_level': dict(zip(names, np.round(small_levels / max(small_levels.sum(), 1), 3).tolist())),
        'attention_by_level_all_queries': {f'layer {i + 1}': dict(zip(names, np.round(attn_all[i] / n_q_all, 3).tolist()))
                                           for i in range(len(dec.layers))},
        'attention_by_level_small_detections': {f'layer {i + 1}': dict(zip(names, np.round(attn_small[i] / max(n_q_small, 1), 3).tolist()))
                                                for i in range(len(dec.layers))},
        'small_detection_queries': n_q_small,
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('--data', default='coco/coco.yaml')
    p.add_argument('--queries', type=int, default=None)
    p.add_argument('--limit', type=int, default=None)
    print(json.dumps(diagnose(**vars(p.parse_args())), indent=1))
