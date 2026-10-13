# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Prediction: letterbox -> network -> top detections -> boxes back in original image coordinates. No NMS.

Post-processing, the same a serving pipeline applies to the exported model: sigmoid, top-k over queries x classes,
confidence threshold, cxcywh -> xyxy, undo the letterbox scale and padding, clip to the image.
"""
import colorsys

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..data import letterbox, load_images
from ..utils.boxes import cxcywh_to_xyxy


class Boxes:
    def __init__(self, xyxy, conf, cls, orig_shape):
        h, w = orig_shape
        self.xyxy = xyxy                                                    # [N, 4] pixels of the original image
        self.xyxyn = xyxy / torch.tensor([w, h, w, h], dtype=xyxy.dtype)    # [N, 4] normalized 0-1
        self.conf = conf                                                    # [N]
        self.cls = cls                                                      # [N] class index

    def __len__(self):
        return len(self.conf)


class Result:
    def __init__(self, image, boxes, names, path=None):
        self.orig_img = image       # the RGB image that was analysed
        self.orig_shape = image.shape[:2]
        self.boxes = boxes
        self.names = names
        self.path = path

    def __len__(self):
        return len(self.boxes)

    def __repr__(self):
        found = ', '.join(f'{self.names[int(c)]} {float(s):.2f}' for c, s in zip(self.boxes.cls, self.boxes.conf))
        return f"Result({self.path or 'array'}, {self.orig_shape[1]}x{self.orig_shape[0]}: {found or 'nothing'})"

    def plot(self):
        """RGB image (numpy) with the detections drawn."""
        im = Image.fromarray(np.ascontiguousarray(self.orig_img))
        draw = ImageDraw.Draw(im)
        lw = max(2, round(min(self.orig_shape) / 300))
        for (x0, y0, x1, y1), c, s in zip(self.boxes.xyxy.tolist(), self.boxes.cls.tolist(), self.boxes.conf.tolist()):
            col = class_color(int(c))
            draw.rectangle([x0, y0, x1, y1], outline=col, width=lw)
            label = f'{self.names[int(c)]} {s:.2f}'
            tb = draw.textbbox((x0, y0), label)
            ty = y0 - (tb[3] - tb[1]) - 2 * lw if y0 - (tb[3] - tb[1]) - 2 * lw >= 0 else y1
            draw.rectangle([x0, ty, x0 + tb[2] - tb[0] + 2 * lw, ty + tb[3] - tb[1] + 2 * lw], fill=col)
            draw.text((x0 + lw, ty + lw - tb[1] + y0), label, fill=(0, 0, 0))
        return np.asarray(im)

    def save(self, path):
        Image.fromarray(self.plot()).save(path)
        return path


def class_color(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.65, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def postprocess(logits, boxes, geom, orig_shape, imgsz, conf, max_det):
    """logits [K, C], boxes [K, 4] cxcywh 0-1 for one image -> (xyxy pixels, scores, classes)."""
    num_classes = logits.shape[-1]
    scores = logits.sigmoid().flatten()
    top, idx = scores.topk(min(max_det, scores.numel()))
    keep = top > conf
    top, idx = top[keep], idx[keep]
    cls, query = idx % num_classes, idx // num_classes
    scale, _, _, pad_top, pad_left = geom
    out_h, out_w = imgsz
    xyxy = cxcywh_to_xyxy(boxes[query]) * torch.tensor([out_w, out_h, out_w, out_h], dtype=boxes.dtype, device=boxes.device)
    xyxy = (xyxy - torch.tensor([pad_left, pad_top, pad_left, pad_top], dtype=boxes.dtype, device=boxes.device)) / scale
    h, w = orig_shape
    xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w)
    xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h)
    return xyxy, top, cls


@torch.no_grad()
def predict(net, names, imgsz, source, color='RGB', conf=0.5, max_det=100, device='cpu', batch=16):
    items = load_images(source, color)
    net = net.to(device).eval()
    results = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        prepared = [letterbox(img, imgsz[0], imgsz[1], device) for img, _ in chunk]
        out = net(torch.stack([t for t, _ in prepared]))
        for j, ((img, path), (_, geom)) in enumerate(zip(chunk, prepared)):
            xyxy, scores, cls = postprocess(out['pred_logits'][j].float(), out['pred_boxes'][j].float(), geom,
                                            img.shape[:2], imgsz, conf, max_det)
            results.append(Result(img, Boxes(xyxy.cpu(), scores.cpu(), cls.cpu(), img.shape[:2]), names, path))
    return results
