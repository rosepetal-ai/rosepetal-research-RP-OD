"""Box helpers. Boxes are normalized to [0, 1]; cxcywh = (center x, center y, width, height)."""
import torch


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(b):
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def box_area(b):
    return (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)


def box_iou(a, b):
    """Pairwise IoU between xyxy boxes a [..., N, 4] and b [..., M, 4] -> (iou [..., N, M], union [..., N, M])."""
    lt = torch.max(a[..., :, None, :2], b[..., None, :, :2])
    rb = torch.min(a[..., :, None, 2:], b[..., None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = box_area(a)[..., :, None] + box_area(b)[..., None, :] - inter
    return inter / union.clamp(min=1e-7), union


def generalized_box_iou(a, b):
    """Pairwise GIoU between xyxy boxes a [..., N, 4] and b [..., M, 4], in [-1, 1]."""
    iou, union = box_iou(a, b)
    lt = torch.min(a[..., :, None, :2], b[..., None, :, :2])
    rb = torch.max(a[..., :, None, 2:], b[..., None, :, 2:])
    enclose = (rb - lt).clamp(min=0).prod(-1)
    return iou - (enclose - union) / enclose.clamp(min=1e-7)


def paired_iou(a, b):
    """IoU of a[i] with b[i] (xyxy, same length)."""
    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    return inter / (box_area(a) + box_area(b) - inter).clamp(min=1e-7)


def paired_giou(a, b):
    """GIoU of a[i] with b[i] (xyxy, same length)."""
    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = box_area(a) + box_area(b) - inter
    iou = inter / union.clamp(min=1e-7)
    elt = torch.min(a[:, :2], b[:, :2])
    erb = torch.max(a[:, 2:], b[:, 2:])
    enclose = (erb - elt).clamp(min=0).prod(-1)
    return iou - (enclose - union) / enclose.clamp(min=1e-7)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))
