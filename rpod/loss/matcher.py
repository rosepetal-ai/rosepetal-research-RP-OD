"""Hungarian matching (training only): each ground-truth object gets exactly one query.

targets: list (one per image) of {'labels': int64 [N], 'boxes': float [N, 4] cxcywh normalized}.
"""
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

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, nq = outputs['pred_logits'].shape[:2]
        sizes = [len(t['boxes']) for t in targets]
        if sum(sizes) == 0:
            empty = torch.empty(0, dtype=torch.int64)
            return [(empty, empty) for _ in range(bs)]
        prob = outputs['pred_logits'].flatten(0, 1).sigmoid()
        boxes = outputs['pred_boxes'].flatten(0, 1)
        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_boxes = torch.cat([t['boxes'] for t in targets])
        p = prob[:, tgt_ids]
        neg = (1 - self.alpha) * p ** self.gamma * -(1 - p + 1e-8).log()
        pos = self.alpha * (1 - p) ** self.gamma * -(p + 1e-8).log()
        c = (self.cost_class * (pos - neg) + self.cost_bbox * torch.cdist(boxes, tgt_boxes, p=1)
             - self.cost_giou * generalized_box_iou(cxcywh_to_xyxy(boxes), cxcywh_to_xyxy(tgt_boxes)))
        c = c.view(bs, nq, -1).float().cpu()
        out = []
        for i, ci in enumerate(c.split(sizes, -1)):
            r, col = linear_sum_assignment(ci[i].numpy())
            out.append((torch.as_tensor(r, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)))
        return out
