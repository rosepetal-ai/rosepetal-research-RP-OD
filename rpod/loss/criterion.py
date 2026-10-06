"""Losses (training only): IoU-aware classification (VFL) + L1 + GIoU, on every decoder layer, the top-K proposals
and the denoising queries.

targets: list (one per image) of {'labels': int64 [N], 'boxes': float [N, 4] cxcywh normalized}.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.boxes import cxcywh_to_xyxy, paired_giou, paired_iou
from .matcher import HungarianMatcher


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher=None, weight_dict=None, alpha=0.75, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        self.weight_dict = weight_dict or {'vfl': 1.0, 'bbox': 5.0, 'giou': 2.0}
        self.alpha, self.gamma = alpha, gamma

    @staticmethod
    def _src_idx(indices):
        batch = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        return batch, torch.cat([src for src, _ in indices])

    def _losses(self, outputs, targets, indices, num_boxes):
        logits, pred_boxes = outputs['pred_logits'], outputs['pred_boxes']
        idx = self._src_idx(indices)
        dev = logits.device
        src_boxes = pred_boxes[idx]
        tgt_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)]).to(dev)
        tgt_labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)]).to(dev)
        # VFL: the target score of a matched query is its IoU with the GT (IoU-aware classification)
        ious = paired_iou(cxcywh_to_xyxy(src_boxes), cxcywh_to_xyxy(tgt_boxes)).detach()
        tgt_cls = torch.full(logits.shape[:2], self.num_classes, dtype=torch.int64, device=dev)
        tgt_cls[idx] = tgt_labels
        onehot = F.one_hot(tgt_cls, self.num_classes + 1)[..., :-1].to(logits.dtype)
        score = torch.zeros(logits.shape[:2], dtype=logits.dtype, device=dev)
        score[idx] = ious.to(logits.dtype)
        tgt_score = score[..., None] * onehot
        weight = self.alpha * logits.sigmoid().detach().pow(self.gamma) * (1 - onehot) + tgt_score
        vfl = F.binary_cross_entropy_with_logits(logits, tgt_score, weight=weight, reduction='none')
        losses = {'vfl': vfl.mean(1).sum() * logits.shape[1] / num_boxes}
        if len(tgt_boxes):
            losses['bbox'] = F.l1_loss(src_boxes, tgt_boxes, reduction='sum') / num_boxes
            losses['giou'] = (1 - paired_giou(cxcywh_to_xyxy(src_boxes), cxcywh_to_xyxy(tgt_boxes))).sum() / num_boxes
        else:
            losses['bbox'] = losses['giou'] = pred_boxes.sum() * 0.0
        return losses

    def forward(self, outputs, targets, num_boxes=None):
        """num_boxes: objects of the whole effective batch when gradients are accumulated over several steps, so the
        summed losses equal those of one big batch. Defaults to the objects of this batch."""
        if num_boxes is None:
            num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = max(1.0, float(num_boxes))
        losses = {}

        def add(out, indices, suffix, nb=num_boxes):
            for k, v in self._losses(out, targets, indices, nb).items():
                losses[k + suffix] = v

        add(outputs, self.matcher(outputs, targets), '')
        for i, aux in enumerate(outputs.get('aux_outputs', [])):
            add(aux, self.matcher(aux, targets), f'_aux{i}')
        for i, aux in enumerate(outputs.get('enc_aux_outputs', [])):
            add(aux, self.matcher(aux, targets), f'_enc{i}')
        if 'dn_meta' in outputs:
            # denoising queries already know their GT: no matching needed
            meta = outputs['dn_meta']
            g = meta['dn_num_group']
            dn_idx = [(pos.cpu(), torch.arange(len(t['labels'])).tile(g)) for pos, t in zip(meta['dn_positive_idx'], targets)]
            for i, aux in enumerate(outputs['dn_aux_outputs']):
                add(aux, dn_idx, f'_dn{i}', num_boxes * g)
        losses['loss'] = sum(self.weight_dict[k.split('_')[0]] * v for k, v in losses.items())
        return losses
