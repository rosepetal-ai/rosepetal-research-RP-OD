"""Hungarian matching (training only): each ground-truth object gets exactly one query.

targets: list (one per image) of {'labels': int64 [N], 'boxes': float [N, 4] cxcywh normalized}.
"""
import itertools

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from ..utils.boxes import cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """cost = w_class * focal cost + w_bbox * L1 + w_giou * (-GIoU); each GT gets exactly one prediction."""

    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou
        self.alpha, self.gamma = alpha, gamma

    def _cost(self, logits, boxes, tgt_ids, tgt_boxes):
        """[N predictions, M objects]"""
        p = logits.sigmoid()[:, tgt_ids]
        neg = (1 - self.alpha) * p ** self.gamma * -(1 - p + 1e-8).log()
        pos = self.alpha * (1 - p) ** self.gamma * -(p + 1e-8).log()
        return (self.cost_class * (pos - neg) + self.cost_bbox * torch.cdist(boxes, tgt_boxes, p=1)
                - self.cost_giou * generalized_box_iou(cxcywh_to_xyxy(boxes), cxcywh_to_xyxy(tgt_boxes)))

    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.match_all([outputs], targets)[0]

    @torch.no_grad()
    def match_all(self, outputs_list, targets):
        """The matchings of several outputs (every decoder layer, ...) with one transfer of the costs to the CPU, which
        has to wait for the GPU to finish: one wait per training step instead of one per output.

        An output with 'valid' [1 or B, N, 1] has a prediction per token of the dense head: its costs are computed
        image by image (the cost matrix of the whole batch would pair every token with every object of every image),
        and only valid tokens can be matched."""
        sizes = [len(t['boxes']) for t in targets]
        empty = torch.empty(0, dtype=torch.int64)
        if sum(sizes) == 0:
            return [[(empty, empty) for _ in targets] for _ in outputs_list]
        offsets = [0, *itertools.accumulate(sizes)]
        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_boxes = torch.cat([t['boxes'] for t in targets])
        pieces = []                                        # per output and image with objects: [queries x objects]
        for out in outputs_list:
            logits, boxes = out['pred_logits'], out['pred_boxes']
            if 'valid' in out:
                for i, t in enumerate(targets):
                    if sizes[i]:
                        c = self._cost(logits[i], boxes[i], t['labels'], t['boxes'])
                        pieces.append(c.masked_fill(~out['valid'][min(i, len(out['valid']) - 1)], 1e6).flatten())
            else:
                c = self._cost(logits.flatten(0, 1), boxes.flatten(0, 1), tgt_ids, tgt_boxes).view(*logits.shape[:2], -1)
                pieces += [c[i, :, offsets[i]:offsets[i + 1]].flatten() for i in range(len(targets)) if sizes[i]]
        costs = torch.cat(pieces).float().cpu().split(
            [out['pred_logits'].shape[1] * n for out in outputs_list for n in sizes if n])
        costs = iter(costs)
        result = []
        for out in outputs_list:
            nq = out['pred_logits'].shape[1]
            result.append([self._assign(next(costs).view(nq, n)) if n else (empty, empty) for n in sizes])
        return result

    @staticmethod
    def _assign(cost):
        r, c = linear_sum_assignment(cost.numpy())
        return torch.as_tensor(r, dtype=torch.int64), torch.as_tensor(c, dtype=torch.int64)
