# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Detection metrics, all in original image pixels.

- mAP50-95 and mAP50: the COCO definition (greedy matching by score, 101 recall points, up to 100 detections per
  image), so the numbers match pycocotools and any paper.
- mAP50-95 per object size: small < 32x32, medium < 96x96, large, measured in pixels at the model input (the size
  that matters for detection), with the COCO rules: objects of other sizes, and the detections that land on them,
  are ignored. With scale 1 (original pixels) the numbers are those of pycocotools.
- At a confidence threshold, with IoU >= 0.5: precision, recall, F1, mean IoU of the hits, false positives per image
  and the confusion matrix. The recommended threshold is the one that maximizes the mean F1 over classes.
"""
from dataclasses import dataclass, field

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
THRESHOLDS = np.linspace(0.0, 1.0, 1001)[1:-1]
MATCH_IOU = 0.5
DEFAULT_THRESHOLD = 0.5
SIZES = (('small', 0.0, 32.0 ** 2), ('medium', 32.0 ** 2, 96.0 ** 2), ('large', 96.0 ** 2, 1e10))   # COCO, ends included
RANGES = (('all', 0.0, float('inf')),) + SIZES


def box_iou(a, b):
    """Pairwise IoU of xyxy boxes, numpy [N, 4] x [M, 4] -> [N, M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def _match(ious, iou_thr, gt_ignore=None):
    """COCO greedy matching for one image and class. ious: [detections sorted by score, objects].
    Returns (matched object or -1, its IoU, matched object is ignored) per detection.

    Each detection, in score order, takes the free object with the highest IoU >= iou_thr; objects that count come
    first, ignored ones (another size) only when no counted object qualifies."""
    n_det, n_gt = ious.shape
    gt_idx, gt_iou, on_ignored = np.full(n_det, -1), np.zeros(n_det), np.zeros(n_det, bool)
    if n_det == 0 or n_gt == 0:
        return gt_idx, gt_iou, on_ignored
    thr = min(iou_thr, 1 - 1e-10)
    ignored = np.zeros(n_gt, bool) if gt_ignore is None else gt_ignore
    taken = np.zeros(n_gt, bool)
    for d in np.flatnonzero(ious.max(1) >= thr):          # only detections that touch some object can match
        for pool in (~ignored, ignored):
            row = np.where(taken | ~pool, -1.0, ious[d])
            g = int(row.argmax())
            if row[g] >= thr:
                taken[g] = True
                gt_idx[d], gt_iou[d], on_ignored[d] = g, ious[d, g], ignored[g]
                break
    return gt_idx, gt_iou, on_ignored


def _area(b):
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def _class_matches(preds, gts, c, scales):
    """Every detection of class c over the dataset, matched at every IoU threshold, for all objects and per size.

    Returns {range: (scores [T] arrays sorted high first, true positives [T] arrays, n_objects)} plus the IoU of each
    hit at IoU 0.5 for all objects. For a size, the detections matched to objects of another size, and the unmatched
    ones of another size, do not count (COCO); so each threshold keeps its own list."""
    T = len(IOU_THRESHOLDS)
    acc = {name: ([[] for _ in range(T)], [[] for _ in range(T)], [0]) for name, _, _ in RANGES}
    iou05 = []
    for (pb, ps, pc), (gb, gc), s in zip(preds, gts, scales):
        m = pc == c
        order = np.argsort(-ps[m], kind='mergesort')
        db, ds, g = pb[m][order], ps[m][order], gb[gc == c]
        if len(db) == 0 and len(g) == 0:
            continue
        ious = box_iou(db, g)
        d_area, g_area = _area(db) * s * s, _area(g) * s * s
        for name, lo, hi in RANGES:
            scores, tps, n = acc[name]
            g_ign = None if name == 'all' else (g_area < lo) | (g_area > hi)
            n[0] += len(g) if g_ign is None else int((~g_ign).sum())
            d_out = np.zeros(len(db), bool) if name == 'all' else (d_area < lo) | (d_area > hi)
            for t, thr in enumerate(IOU_THRESHOLDS):
                idx, iou, on_ign = _match(ious, thr, g_ign)
                if name == 'all' and t == 0:
                    iou05.append(iou)
                counts = ~(on_ign | ((idx < 0) & d_out))
                scores[t].append(ds[counts])
                tps[t].append((idx >= 0)[counts])
    out = {}
    for name, (scores, tps, n) in acc.items():
        s_sorted, t_sorted = [], []
        for t in range(T):
            sc = np.concatenate(scores[t]) if scores[t] else np.zeros(0)
            tp = np.concatenate(tps[t]).astype(bool) if tps[t] else np.zeros(0, bool)
            order = np.argsort(-sc, kind='mergesort')
            s_sorted.append(sc[order])
            t_sorted.append(tp[order])
        out[name] = (s_sorted, t_sorted, n[0])
    sc_all = np.concatenate(acc['all'][0][0]) if acc['all'][0][0] else np.zeros(0)
    ious_all = np.concatenate(iou05) if iou05 else np.zeros(0)
    out['iou05'] = ious_all[np.argsort(-sc_all, kind='mergesort')]
    return out


def average_precision(tp, n_gt):
    """COCO AP from true-positive flags sorted by score: precision envelope sampled at 101 recall points."""
    if n_gt == 0:
        return np.nan
    if len(tp) == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, RECALL_POINTS, side='left')
    q = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(q.mean())


@dataclass
class Metrics:
    map50_95: float
    map50: float
    precision: float
    recall: float
    f1: float
    threshold: float
    mean_iou: float
    fp_per_image: float
    fitness: float
    map_small: float = -1.0       # mAP50-95 per object size at the model input; -1: no object of that size
    map_medium: float = -1.0
    map_large: float = -1.0
    size_objects: dict = field(default_factory=dict)
    per_class: list = field(default_factory=list)
    confusion_matrix: np.ndarray = None
    curves: dict = field(default_factory=dict, repr=False)

    def to_dict(self):
        keys = ('map50_95', 'map50', 'map_small', 'map_medium', 'map_large', 'precision', 'recall', 'f1', 'threshold',
                'mean_iou', 'fp_per_image', 'fitness')
        return {k: float(getattr(self, k)) for k in keys}

    def table(self, names):
        head = f"{'class':16}{'images':>8}{'objects':>9}{'P':>7}{'R':>7}{'F1':>7}{'mAP50':>8}{'mAP50-95':>10}{'IoU':>7}"
        rows = [head]
        for r in [dict(self.to_dict(), name='all', images=self.curves.get('images', 0),
                       instances=sum(p['instances'] for p in self.per_class))] + self.per_class:
            if r['instances'] == 0:   # nothing of this class in the split: no metric makes sense
                rows.append(f"{r['name'][:15]:16}{r['images']:8d}{0:9d}" + ''.join(f'{"-":>{w}}' for w in (7, 7, 7, 8, 10, 7)))
                continue
            rows.append(f"{r['name'][:15]:16}{r['images']:8d}{r['instances']:9d}{r['precision']:7.3f}{r['recall']:7.3f}"
                        f"{r['f1']:7.3f}{r['map50']:8.3f}{r['map50_95']:10.3f}{r['mean_iou']:7.3f}")
        sizes = [f"{name} {self.size_objects.get(name, 0)} objects " + (f'{v:.3f}' if v >= 0 else '-')
                 for name, v in (('small', self.map_small), ('medium', self.map_medium), ('large', self.map_large))]
        rows.append('mAP50-95 by size at the model input: ' + ' · '.join(sizes))
        return '\n'.join(rows)


def evaluate(preds, gts, names, threshold=None, scales=None):
    """preds: per image (boxes xyxy [N, 4], scores [N], classes [N]); gts: per image (boxes xyxy [M, 4], classes [M]).

    threshold None: pick the one with the best mean F1 (training validation). A number: use it as given (test).
    scales: per image, model-input pixels per original pixel (the letterbox scale), to measure object sizes at the
    model input; None: original pixels.
    """
    num_classes = len(names)
    n_images = len(gts)
    scales = np.ones(n_images) if scales is None else np.asarray(scales, np.float64)
    ap = np.full((len(RANGES), len(IOU_THRESHOLDS), num_classes), np.nan)
    n_by_size = np.zeros((len(RANGES), num_classes), int)
    per_class_05 = []
    for c in range(num_classes):
        m = _class_matches(preds, gts, c, scales)
        for r, (name, _, _) in enumerate(RANGES):
            scores, tps, n = m[name]
            n_by_size[r, c] = n
            for t in range(len(IOU_THRESHOLDS)):
                ap[r, t, c] = average_precision(tps[t], n)
        per_class_05.append((m['all'][0][0], m['all'][1][0], m['iou05']))
    n_gt = n_by_size[0]
    by_size = {name: float(np.nanmean(ap[r][:, n_by_size[r] > 0])) if (n_by_size[r] > 0).any() else -1.0
               for r, (name, _, _) in enumerate(RANGES) if name != 'all'}
    ap = ap[0]
    present = n_gt > 0

    # precision / recall / F1 of every class for every candidate threshold (detection kept if score > threshold)
    P = np.zeros((num_classes, len(THRESHOLDS)))
    R = np.zeros_like(P)
    for c, (scores, tp, _) in enumerate(per_class_05):
        kept = len(scores) - np.searchsorted(scores[::-1], THRESHOLDS, side='right')   # detections with score > t
        tps = np.concatenate([[0], np.cumsum(tp)])[kept]                               # scores are sorted, high first
        P[c] = tps / np.maximum(kept, 1)
        R[c] = tps / max(n_gt[c], 1)
    F = 2 * P * R / np.maximum(P + R, 1e-12)
    mean_f1 = F[present].mean(0) if present.any() else np.zeros(len(THRESHOLDS))
    if threshold is None:
        if mean_f1.max() <= 0:
            threshold = DEFAULT_THRESHOLD                               # nothing found yet: no basis to choose
        else:
            best = np.flatnonzero(mean_f1 >= mean_f1.max() - 1e-12)[-1]   # ties: the highest, fewer false alarms
            threshold = float(THRESHOLDS[best])

    rows, all_ious, fp = [], [], 0
    for c, (scores, tp, ious) in enumerate(per_class_05):
        keep = scores > threshold
        tp_k = tp & keep
        n_det = int(keep.sum())
        n_tp = int(tp_k.sum())
        prec = n_tp / max(n_det, 1)
        rec = n_tp / max(n_gt[c], 1)
        fp += n_det - n_tp
        all_ious.append(ious[tp_k])
        rows.append({'name': names[c], 'images': int(sum(1 for _, gc in gts if (gc == c).any())),
                     'instances': int(n_gt[c]), 'precision': prec, 'recall': rec,
                     'f1': 2 * prec * rec / max(prec + rec, 1e-12),
                     'map50': float(ap[0, c]) if present[c] else 0.0,
                     'map50_95': float(ap[:, c].mean()) if present[c] else 0.0,
                     'mean_iou': float(ious[tp_k].mean()) if n_tp else 0.0})
    valid = [r for r, p in zip(rows, present) if p]
    mean = (lambda k: float(np.mean([r[k] for r in valid])) if valid else 0.0)
    all_ious = np.concatenate(all_ious) if all_ious else np.zeros(0)
    map50_95 = float(np.nanmean(ap[:, present])) if present.any() else 0.0
    recall = mean('recall')
    return Metrics(
        map50_95=map50_95, map50=float(np.nanmean(ap[0, present])) if present.any() else 0.0,
        precision=mean('precision'), recall=recall, f1=mean('f1'), threshold=threshold,
        mean_iou=float(all_ious.mean()) if len(all_ious) else 0.0, fp_per_image=fp / max(n_images, 1),
        fitness=0.5 * map50_95 + 0.5 * recall, map_small=by_size['small'], map_medium=by_size['medium'],
        map_large=by_size['large'], size_objects={name: int(n_by_size[r].sum()) for r, (name, _, _) in enumerate(RANGES)
                                                  if name != 'all'}, per_class=rows,
        confusion_matrix=confusion_matrix(preds, gts, num_classes, threshold),
        curves={'thresholds': THRESHOLDS, 'precision': P, 'recall': R, 'f1': F, 'present': present,
                'images': n_images, 'pr50': [(s, t, int(n)) for (s, t, _), n in zip(per_class_05, n_gt)]})


def confusion_matrix(preds, gts, num_classes, threshold):
    """Rows: true class (+ background = false alarms); columns: predicted class (+ background = missed objects).

    Class-agnostic matching at IoU >= 0.5, pairs taken by decreasing IoU, so a detection with the wrong class is
    counted as a confusion and not as a miss plus a false alarm."""
    bg = num_classes
    m = np.zeros((num_classes + 1, num_classes + 1), np.int64)
    for (pb, ps, pc), (gb, gc) in zip(preds, gts):
        keep = ps > threshold
        pb, pc = pb[keep], pc[keep]
        ious = box_iou(gb, pb)
        used_g, used_p = set(), set()
        for g, p in zip(*np.unravel_index(np.argsort(-ious, axis=None), ious.shape)) if ious.size else []:
            if ious[g, p] < MATCH_IOU:
                break
            if g in used_g or p in used_p:
                continue
            used_g.add(g)
            used_p.add(p)
            m[gc[g], pc[p]] += 1
        for g in range(len(gb)):
            if g not in used_g:
                m[gc[g], bg] += 1
        for p in range(len(pb)):
            if p not in used_p:
                m[bg, pc[p]] += 1
    return m
