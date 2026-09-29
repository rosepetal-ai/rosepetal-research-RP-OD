"""Detection metrics, all in original image pixels.

- mAP50-95 and mAP50: the COCO definition (greedy matching by score, 101 recall points, up to 100 detections per
  image), so the numbers match pycocotools and any paper.
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


def _match(det_boxes, gt_boxes, iou_thr):
    """COCO greedy matching for one image and class; det_boxes sorted by score. Returns (matched gt or -1, its IoU).

    Each detection, in score order, takes the free ground truth with the highest IoU >= iou_thr."""
    gt_idx, gt_iou = np.full(len(det_boxes), -1), np.zeros(len(det_boxes))
    if len(det_boxes) == 0 or len(gt_boxes) == 0:
        return gt_idx, gt_iou
    ious = box_iou(det_boxes, gt_boxes)
    thr = min(iou_thr, 1 - 1e-10)
    taken = np.zeros(len(gt_boxes), bool)
    for d in np.flatnonzero(ious.max(1) >= thr):          # only detections that touch some object can match
        row = np.where(taken, -1.0, ious[d])
        g = int(row.argmax())
        if row[g] >= thr:
            taken[g] = True
            gt_idx[d], gt_iou[d] = g, ious[d, g]
    return gt_idx, gt_iou


def _per_class_matches(preds, gts, c, iou_thr):
    """All detections of class c over the dataset: (scores, is_true_positive, iou_with_its_gt), sorted by score."""
    scores, tps, ious = [], [], []
    for (pb, ps, pc), (gb, gc) in zip(preds, gts):
        m = pc == c
        order = np.argsort(-ps[m], kind='mergesort')
        db, ds = pb[m][order], ps[m][order]
        gt_idx, gt_iou = _match(db, gb[gc == c], iou_thr)
        scores.append(ds)
        tps.append(gt_idx >= 0)
        ious.append(gt_iou)
    scores, tps, ious = (np.concatenate(x) if x else np.zeros(0) for x in (scores, tps, ious))
    order = np.argsort(-scores, kind='mergesort')
    return scores[order], tps[order].astype(bool), ious[order]


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
    per_class: list = field(default_factory=list)
    confusion_matrix: np.ndarray = None
    curves: dict = field(default_factory=dict, repr=False)

    def to_dict(self):
        keys = ('map50_95', 'map50', 'precision', 'recall', 'f1', 'threshold', 'mean_iou', 'fp_per_image', 'fitness')
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
        return '\n'.join(rows)


def evaluate(preds, gts, names, threshold=None):
    """preds: per image (boxes xyxy [N, 4], scores [N], classes [N]); gts: per image (boxes xyxy [M, 4], classes [M]).

    threshold None: pick the one with the best mean F1 (training validation). A number: use it as given (test).
    """
    num_classes = len(names)
    n_images = len(gts)
    ap = np.full((len(IOU_THRESHOLDS), num_classes), np.nan)
    n_gt = np.array([sum(int((gc == c).sum()) for _, gc in gts) for c in range(num_classes)])
    per_class_05 = []
    for c in range(num_classes):
        for t, iou_thr in enumerate(IOU_THRESHOLDS):
            scores, tp, ious = _per_class_matches(preds, gts, c, iou_thr)
            ap[t, c] = average_precision(tp, n_gt[c])
            if t == 0:
                per_class_05.append((scores, tp, ious))
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
        fitness=0.5 * map50_95 + 0.5 * recall, per_class=rows,
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
