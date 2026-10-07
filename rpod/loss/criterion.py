"""Losses (training only), as in D-FINE.

- Classification: MAL (DEIM's matchability-aware loss): a matched query learns the score IoU^gamma, with full weight
  however low its IoU (VFL weighted it by the IoU, so poor matches barely learned), unmatched ones learn 0 weighted by
  their score^gamma. On every decoder layer, the initial boxes, the dense head (every token, so the token that fits an
  object best learns to score high even if the top-K selection missed it) and the denoising queries, each with its own
  Hungarian matching.
- Boxes: L1 + GIoU, plus two losses on the edge distributions of the decoder layers: FGL (fine-grained localization:
  each true edge as the two bins around it, weighted by the IoU) and DDF (GO-LSD self-distillation: every layer's
  distributions pulled towards the last layer's, weighted by its confidence).
  Boxes are supervised on the union of the matchings of all layers (GO-LSD): a query that any layer matched learns its
  box, not only the one the last layer picked.

targets: list (one per image) of {'labels': int64 [N], 'boxes': float [N, 4] cxcywh normalized}.
"""
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.decoder import REG_MAX, bin_positions, box_to_distance
from ..utils.boxes import cxcywh_to_xyxy, paired_giou, paired_iou
from .matcher import HungarianMatcher

DDF_TEMPERATURE = 5.0


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher=None, weight_dict=None, gamma=1.5):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        self.weight_dict = weight_dict or {'cls': 1.0, 'bbox': 5.0, 'giou': 2.0, 'fgl': 0.15, 'ddf': 1.5}
        self.gamma = gamma
        self._bins = {}

    @staticmethod
    def _to_device(matchings, sizes, device):
        """Matchings (per image: CPU query and object indices) -> (image, query, object) index tensors on the device,
        object = position among the objects of the whole batch. One asynchronous copy for all of them: indexing a
        device tensor with CPU indices copies them one at a time, each copy waiting for the GPU to finish."""
        offsets = [0, *itertools.accumulate(sizes)]
        parts = [torch.stack([torch.cat([torch.full_like(q, i) for i, (q, _) in enumerate(m)]),
                              torch.cat([q for q, _ in m]),
                              torch.cat([g + offsets[i] for i, (_, g) in enumerate(m)])]) for m in matchings]
        flat = torch.cat(parts, 1)
        if device.type == 'cuda':
            flat = flat.pin_memory().to(device, non_blocking=True)
        return [tuple(p.unbind(0)) for p in flat.split([p.shape[1] for p in parts], 1)]

    def _cls(self, outputs, gt, idx, num_boxes):
        """MAL: target IoU^gamma for the matched class of a matched query, 0 elsewhere."""
        logits, pred_boxes = outputs['pred_logits'], outputs['pred_boxes']
        b, q, g = idx
        labels = gt[0][g]
        ious = paired_iou(cxcywh_to_xyxy(pred_boxes[b, q]), cxcywh_to_xyxy(gt[1][g]))
        positive = torch.zeros_like(logits)            # built in place: on the dense head, logits are [B, tokens, C]
        positive[b, q, labels] = 1.0
        tgt_score = torch.zeros_like(logits)
        tgt_score[b, q, labels] = ious.detach().to(logits.dtype).pow(self.gamma)
        weight = logits.sigmoid().detach().pow(self.gamma) * (1 - positive) + positive
        loss = F.binary_cross_entropy_with_logits(logits, tgt_score, weight=weight, reduction='none')
        return loss.sum() / num_boxes

    def _box_losses(self, outputs, gt, idx, num_boxes, edge_targets=None, ddf_norm=None):
        """L1 and GIoU; FGL when the output has edge distributions; DDF when it also has a teacher."""
        pred_boxes = outputs['pred_boxes']
        b, q, g = idx
        src_boxes, tgt_boxes = pred_boxes[b, q], gt[1][g]
        iou = paired_iou(cxcywh_to_xyxy(src_boxes), cxcywh_to_xyxy(tgt_boxes)).detach()
        if len(tgt_boxes):
            losses = {'bbox': F.l1_loss(src_boxes, tgt_boxes, reduction='sum') / num_boxes,
                      'giou': (1 - paired_giou(cxcywh_to_xyxy(src_boxes), cxcywh_to_xyxy(tgt_boxes))).sum() / num_boxes}
        else:
            losses = {'bbox': pred_boxes.sum() * 0.0, 'giou': pred_boxes.sum() * 0.0}
        if 'pred_corners' in outputs:
            corners = outputs['pred_corners']
            if len(tgt_boxes):
                left, w_left, w_right = edge_targets
                pred = corners[b, q].reshape(-1, REG_MAX + 1)
                ce = (F.cross_entropy(pred, left, reduction='none') * w_left
                      + F.cross_entropy(pred, left + 1, reduction='none') * w_right)
                losses['fgl'] = (ce * iou.repeat_interleave(4)).sum() / num_boxes
            else:
                losses['fgl'] = corners.sum() * 0.0
            if 'teacher_corners' in outputs:
                losses['ddf'] = self._ddf(outputs, (b, q), iou, *ddf_norm)
        return losses

    @staticmethod
    def _ddf(outputs, idx, iou, w_matched, w_unmatched, scale):
        """KL from the teacher's edge distributions to this layer's, per edge, weighted by the teacher's confidence
        (by the IoU on matched queries); matched and unmatched queries averaged apart and mixed by sqrt of their counts."""
        t = DDF_TEMPERATURE
        pred = outputs['pred_corners'].reshape(-1, REG_MAX + 1)
        teacher = outputs['teacher_corners'].detach().reshape(-1, REG_MAX + 1)
        weight = outputs['teacher_logits'].detach().sigmoid().max(-1).values
        weight[idx] = iou.to(weight.dtype)
        matched = torch.zeros_like(weight)
        matched[idx] = 1.0
        kl = F.kl_div(F.log_softmax(pred / t, -1), F.softmax(teacher / t, -1), reduction='none').sum(-1) * t ** 2
        kl = kl * weight.reshape(-1).repeat_interleave(4)
        matched = matched.reshape(-1).repeat_interleave(4)
        # means of the matched and the unmatched edges (0 if there are none), without reading anything from the GPU
        mean = lambda m: (kl * m).sum() / m.sum().clamp(min=1)
        return (mean(matched) * w_matched + mean(1 - matched) * w_unmatched) / (w_matched + w_unmatched) * scale

    def _edge_targets(self, ref_points, gt, idx):
        """FGL target of each matched query: every true edge, as distance from the initial box, split between the two
        bins around it in proportion to how close it is to each -> (left bin, its weight, the right bin's weight)."""
        b, q, g = idx
        dist = box_to_distance(ref_points[b, q].detach(), cxcywh_to_xyxy(gt[1][g])).reshape(-1)
        if dist.device not in self._bins:
            self._bins[dist.device] = bin_positions().to(dist)
        bins = self._bins[dist.device].to(dist.dtype)
        left = (bins[None] <= dist[:, None]).sum(1) - 1
        below, above = left < 0, left >= REG_MAX
        left = left.clamp(0, REG_MAX - 1)
        d_left, d_right = (dist - bins[left]).abs(), (bins[left + 1] - dist).abs()
        w_right = torch.where(below, 0.0, torch.where(above, 1.0, d_left / (d_left + d_right)))
        return left, 1 - w_right, w_right

    @staticmethod
    def _union(matchings):
        """GO-LSD's matching: every (query, object) pair that any layer matched; a query matched to several objects
        keeps the one most layers agree on."""
        out = []
        for per_image in zip(*matchings):
            pairs = torch.stack([torch.cat([s for s, _ in per_image]), torch.cat([t for _, t in per_image])], 1)
            if not len(pairs):
                out.append(per_image[0])
                continue
            pairs, counts = pairs.unique(dim=0, return_counts=True)
            pairs = pairs[counts.argsort(descending=True, stable=True)]
            queries, inverse = pairs[:, 0].unique(return_inverse=True)
            first = torch.full((len(queries),), len(pairs), dtype=torch.int64).scatter_reduce(
                0, inverse, torch.arange(len(pairs)), 'amin')
            out.append((pairs[first, 0], pairs[first, 1]))
        return out

    def forward(self, outputs, targets, num_boxes=None, num_images=None):
        """num_boxes, num_images: objects and images of the whole effective batch when gradients are accumulated over
        several steps, so the summed losses match those of one big batch: exactly for MAL, L1 and GIoU; for the terms
        normalized by the GO-LSD matches (L1, GIoU and FGL of the decoder) and for DDF, by this step's share of the
        objects and images (exact when the steps are alike). Default: this batch."""
        sizes = [len(t['labels']) for t in targets]
        batch_boxes = sum(sizes)
        bs, nq = outputs['pred_logits'].shape[:2]
        dev = outputs['pred_logits'].device
        num_boxes = max(1.0, float(batch_boxes if num_boxes is None else num_boxes))
        num_images = float(bs if num_images is None else num_images)
        gt = (torch.cat([t['labels'] for t in targets]).to(dev), torch.cat([t['boxes'] for t in targets]).to(dev))
        losses = {}

        def add(values, suffix):
            for k, v in values.items():
                losses[k + suffix] = v

        aux, enc = outputs.get('aux_outputs', []), outputs.get('enc_aux_outputs', [])
        pre = [outputs['pre_outputs']] if 'pre_outputs' in outputs else []
        outs = {'main': [outputs], 'aux': aux, 'pre': pre, 'enc': enc}
        found = iter(self.matcher.match_all([o for os in outs.values() for o in os], targets))
        match = {k: [next(found) for _ in os] for k, os in outs.items()}
        match['go'] = [self._union(match['main'] + match['aux'] + match['pre'])]
        n_go = sum(len(s) for s, _ in match['go'][0])
        num_go = max(1.0, n_go * num_boxes / max(batch_boxes, 1))
        match['dn'] = []
        if 'dn_meta' in outputs:
            # denoising queries already know their GT: no matching needed
            g = outputs['dn_meta']['dn_num_group']
            match['dn'] = [[(pos, torch.arange(n).tile(g)) for pos, n in zip(outputs['dn_meta']['dn_positive_idx'], sizes)]]
        on_device = iter(self._to_device([m for ms in match.values() for m in ms], sizes, dev))
        idx = {k: [next(on_device) for _ in ms] for k, ms in match.items()}
        main_i, go_i = idx['main'][0], idx['go'][0]
        go_edges = self._edge_targets(outputs['ref_points'], gt, go_i) if 'pred_corners' in outputs else None
        # D-FINE mixes the matched and unmatched DDF means by the square root of their edge counts per 8 images
        ddf_norm = (math.sqrt(4 * n_go * 8 / bs), math.sqrt(4 * (bs * nq - n_go) * 8 / bs), bs / num_images)

        add({'cls': self._cls(outputs, gt, main_i, num_boxes),
             **self._box_losses(outputs, gt, go_i, num_go, go_edges, ddf_norm)}, '')
        for i, (a, m) in enumerate(zip(aux, idx['aux'])):
            add({'cls': self._cls(a, gt, m, num_boxes), **self._box_losses(a, gt, go_i, num_go, go_edges, ddf_norm)},
                f'_aux{i}')
        for p, m in zip(pre, idx['pre']):
            add({'cls': self._cls(p, gt, m, num_boxes), **self._box_losses(p, gt, go_i, num_go)}, '_pre')
        for i, (e, m) in enumerate(zip(enc, idx['enc'])):
            add({'cls': self._cls(e, gt, m, num_boxes), **self._box_losses(e, gt, m, num_boxes)}, f'_enc{i}')
        if idx['dn']:
            dn_i, num_dn = idx['dn'][0], num_boxes * outputs['dn_meta']['dn_num_group']
            dn_layers = outputs['dn_aux_outputs']
            dn_edges = self._edge_targets(dn_layers[0]['ref_points'], gt, dn_i)
            for i, a in enumerate(dn_layers):
                add({'cls': self._cls(a, gt, dn_i, num_dn),
                     **self._box_losses(a, gt, dn_i, num_dn, dn_edges, ddf_norm)}, f'_dn{i}')
            p = outputs['dn_pre_outputs']
            add({'cls': self._cls(p, gt, dn_i, num_dn), **self._box_losses(p, gt, dn_i, num_dn)}, '_dnpre')
        losses['loss'] = sum(self.weight_dict[k.split('_')[0]] * v for k, v in losses.items())
        return losses
