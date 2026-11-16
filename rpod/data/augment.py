# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""RP-OD's default training augmentation: the recipe of its COCO pretraining, applied to training images before the
letterbox. RPOD.train uses it unless it gets augment=None (no augmentation) or another function(sample, ctx).

Every phase is a fraction of ctx.epochs, the epoch where training ends: the `epochs` given to train, or earlier when
patience starts the final learning-rate decay sooner (the trainer moves ctx.epochs with it). So the recipe scales with
any number of epochs. With a time budget the phases are fractions of the time instead, and the trainer tells every
epoch's phase (ctx.mosaic, ctx.clean).

  first mosaic_fraction (25 %)   mosaic with probability `mosaic` (0.5)
  every epoch but the last 4 %   photometric distortion 0.5, zoom-out 0.5 (not after a mosaic), IoU crop 0.8, flip 0.5
  last clean_fraction (4 %)      clean images, at least one epoch (the same ones where multi-scale returns to imgsz)

The values are the fields of Augment; `default_augment = Augment()` is the recipe as measured, and another run takes
another one, e.g. `RPOD.train(..., augment=Augment(hflip=0.0, mosaic_fraction=0.4))`. The trainer reads the two phase
fractions from it, so its clean stretch (back to the best weights) and its time planning follow the same phases.

The RT-DETR / D-FINE recipe: photometric distortion, zoom-out, IoU crop and horizontal flip, with a short clean phase
at the end. A longer clean phase (10 %, 7 of 72 epochs) made the first COCO run overfit: training loss fell from 14.5
to 12.0 while validation dropped from 50.6 to 49.8. It runs on the original image, before RP-OD's letterbox, so
changing the image content is what gives scale variety with a fixed input: zooming out makes the objects smaller on
the canvas, cropping makes them bigger.

Mosaic (YOLO, DEIM) only early on, the same epochs where multi-scale training uses its widest range (DEIM keeps it for
the first half): four training images in a 2x2 grid around a random centre, on a canvas twice the model input, so
objects arrive at about half their size: more small objects, more objects and more context per image. After a mosaic
come the photometric distortion, the IoU crop and the flip, not the zoom-out (the objects are already halved). In
6-epoch COCO fine-tunes it was the only change that helped at 640: +0.3 AP, +0.4 AP small.

Differences with the RT-DETR pipeline: the image is letterboxed (aspect ratio kept, like YOLO and like production)
instead of stretched, and the zoom-out fill is the letterbox gray.
"""
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageEnhance

from .dataset import Sample

FILL = 114                                           # same gray as the letterbox padding
CROP_SCALE, CROP_ASPECT, CROP_TRIALS = (0.3, 1.0), (0.5, 2.0), 40
CROP_MIN_IOUS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)  # SSD sampler; 1.0 means "no crop this time"


@dataclass(frozen=True)
class Augment:
    """function(sample, ctx) -> sample for RPOD.train (see the module docstring): probabilities and phase fractions."""
    mosaic: float = 0.5                  # benchmarks/coco/train_coco.py --mosaic 0 sets 0
    mosaic_fraction: float = 0.25        # first quarter of the epochs = rpod's widest multi-scale phase
    photometric: float = 0.5
    zoom_out: float = 0.5
    zoom_out_max: float = 4.0            # canvas up to 4x the image side
    iou_crop: float = 0.8
    hflip: float = 0.5
    clean_fraction: float = 0.04         # at least one epoch, whatever the fraction

    def __post_init__(self):
        for k in ('mosaic', 'photometric', 'zoom_out', 'iou_crop', 'hflip', 'mosaic_fraction', 'clean_fraction'):
            if not 0.0 <= getattr(self, k) <= 1.0:
                raise ValueError(f'augment {k} must be between 0 and 1, got {getattr(self, k)}')
        if self.zoom_out_max < 1.0:
            raise ValueError(f'augment zoom_out_max must be at least 1, got {self.zoom_out_max}')
        if self.mosaic_fraction + self.clean_fraction > 1.0:
            raise ValueError(f'the mosaic phase ({self.mosaic_fraction}) and the clean one ({self.clean_fraction}) '
                             f'overlap: together they must be at most 1')

    def __call__(self, sample, ctx):
        clean = getattr(ctx, 'clean', None)
        if ctx.epoch >= ctx.epochs - max(1, round(self.clean_fraction * ctx.epochs)) if clean is None else clean:
            return sample
        rng = ctx.rng
        told = getattr(ctx, 'mosaic', None)
        mosaic_phase = ctx.epoch < round(self.mosaic_fraction * ctx.epochs) if told is None else told
        mosaic_done = mosaic_phase and rng.random() < self.mosaic
        if mosaic_done:
            sample = mosaic([sample] + ctx.sample(3), ctx.imgsz, rng)
        if rng.random() < self.photometric:
            sample = Sample(photometric(sample.image, rng), sample.boxes, sample.labels, sample.path)
        if not mosaic_done and rng.random() < self.zoom_out:
            sample = zoom_out(sample, rng, self.zoom_out_max)
        if rng.random() < self.iou_crop:
            sample = iou_crop(sample, rng)
        if rng.random() < self.hflip:
            sample = hflip(sample)
        return sample

    def to_dict(self):
        return asdict(self)


default_augment = Augment()


def photometric(image, rng):
    """Brightness, contrast, saturation and hue, each with probability 0.5 (torchvision RandomPhotometricDistort)."""
    im = Image.fromarray(image)
    contrast_first = rng.random() < 0.5
    if rng.random() < 0.5:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.875, 1.125))
    if contrast_first and rng.random() < 0.5:
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.5, 1.5))
    if rng.random() < 0.5:
        im = ImageEnhance.Color(im).enhance(rng.uniform(0.5, 1.5))
    if rng.random() < 0.5:
        shift = int(round(rng.uniform(-0.05, 0.05) * 255))
        h, s, v = im.convert('HSV').split()
        im = Image.merge('HSV', (h.point(lambda x: (x + shift) % 256), s, v)).convert('RGB')
    if not contrast_first and rng.random() < 0.5:
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.5, 1.5))
    return np.asarray(im)


def mosaic(samples, imgsz, rng):
    """Four samples in a 2x2 grid on a (2h, 2w) canvas, h x w = the model input; the grid centre is random in the
    middle half. Each image is resized so its longer side fits the input (as in YOLO) and placed against the centre;
    what falls outside the canvas is cut, and the boxes with it (the dataset drops those left under 1 pixel)."""
    h, w = imgsz
    canvas = np.full((2 * h, 2 * w, 3), FILL, np.uint8)
    yc, xc = int(rng.uniform(0.5 * h, 1.5 * h)), int(rng.uniform(0.5 * w, 1.5 * w))
    boxes, labels = [], []
    for k, s in enumerate(samples):
        ih, iw = s.image.shape[:2]
        r = min(h / ih, w / iw)
        nh, nw = max(1, round(ih * r)), max(1, round(iw * r))
        img = np.asarray(Image.fromarray(s.image).resize((nw, nh), Image.BILINEAR))
        x0, y0 = (xc - nw, yc - nh) if k == 0 else (xc, yc - nh) if k == 1 else (xc - nw, yc) if k == 2 else (xc, yc)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + nw, 2 * w), min(y0 + nh, 2 * h)
        canvas[cy0:cy1, cx0:cx1] = img[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        b = s.boxes * r + np.array([x0, y0, x0, y0], np.float32)
        b[:, 0::2] = b[:, 0::2].clip(cx0, cx1)
        b[:, 1::2] = b[:, 1::2].clip(cy0, cy1)
        boxes.append(b)
        labels.append(s.labels)
    return Sample(canvas, np.concatenate(boxes).astype(np.float32), np.concatenate(labels), samples[0].path)


def zoom_out(sample, rng, max_ratio=4.0):
    """Place the image at a random spot of a canvas up to max_ratio times bigger."""
    h, w = sample.image.shape[:2]
    r = rng.uniform(1.0, max_ratio)
    big_h, big_w = int(h * r), int(w * r)
    top, left = int(rng.uniform(0, big_h - h)), int(rng.uniform(0, big_w - w))
    canvas = np.full((big_h, big_w, 3), FILL, np.uint8)
    canvas[top:top + h, left:left + w] = sample.image
    return Sample(canvas, sample.boxes + np.array([left, top, left, top], np.float32), sample.labels, sample.path)


def iou_crop(sample, rng, max_rounds=50):
    """SSD crop: a random window that overlaps some object by at least a random IoU; objects whose center falls
    outside are dropped, the rest clipped. Images without objects are left as they are."""
    boxes = sample.boxes
    if len(boxes) == 0:
        return sample
    h, w = sample.image.shape[:2]
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    for _ in range(max_rounds):
        min_iou = CROP_MIN_IOUS[rng.integers(len(CROP_MIN_IOUS))]
        if min_iou >= 1.0:
            return sample
        for _ in range(CROP_TRIALS):
            sw, sh = rng.uniform(*CROP_SCALE, 2)
            cw, ch = int(w * sw), int(h * sh)
            if cw < 1 or ch < 1 or not CROP_ASPECT[0] <= cw / ch <= CROP_ASPECT[1]:
                continue
            left, top = int((w - cw) * rng.random()), int((h - ch) * rng.random())
            right, bottom = left + cw, top + ch
            inside = (centers[:, 0] > left) & (centers[:, 0] < right) & (centers[:, 1] > top) & (centers[:, 1] < bottom)
            if not inside.any() or _iou(boxes[inside], [left, top, right, bottom]).max() < min_iou:
                continue
            kept = boxes[inside].copy()
            kept[:, 0::2] = kept[:, 0::2].clip(left, right) - left
            kept[:, 1::2] = kept[:, 1::2].clip(top, bottom) - top
            return Sample(np.ascontiguousarray(sample.image[top:bottom, left:right]), kept,
                          sample.labels[inside], sample.path)
    return sample


def hflip(sample):
    w = sample.image.shape[1]
    boxes = sample.boxes.copy()
    boxes[:, [0, 2]] = w - sample.boxes[:, [2, 0]]
    return Sample(np.ascontiguousarray(sample.image[:, ::-1]), boxes, sample.labels, sample.path)


def _iou(boxes, window):
    lt = np.maximum(boxes[:, :2], window[:2])
    rb = np.minimum(boxes[:, 2:], window[2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    area = (boxes[:, 2:] - boxes[:, :2]).prod(1)
    return inter / (area + (window[2] - window[0]) * (window[3] - window[1]) - inter)
