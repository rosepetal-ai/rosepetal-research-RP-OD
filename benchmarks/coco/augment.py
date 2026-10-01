"""COCO training augmentations for the RP-OD augment hook (the library itself has none).

The RT-DETR / D-FINE recipe: photometric distortion, zoom-out, IoU crop and horizontal flip, switched off for the last
10 % of the epochs so training ends on clean images. It runs on the original image, before RP-OD's letterbox, so
changing the image content is what gives scale variety with a fixed input: zooming out makes the objects smaller on the
canvas, cropping makes them bigger.

Differences with the RT-DETR pipeline: the image is letterboxed (aspect ratio kept, like YOLO and like production)
instead of stretched to 640x640, the zoom-out fill is the letterbox gray, and there is no multi-scale input size.
"""
import numpy as np
from PIL import Image, ImageEnhance

from rpod.data.dataset import Sample

P_PHOTOMETRIC, P_ZOOM_OUT, P_IOU_CROP, P_FLIP = 0.5, 0.5, 0.8, 0.5
ZOOM_OUT_MAX = 4.0                                   # canvas up to 4x the image side
FILL = 114                                           # same gray as the letterbox padding
CROP_SCALE, CROP_ASPECT, CROP_TRIALS = (0.3, 1.0), (0.5, 2.0), 40
CROP_MIN_IOUS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)  # SSD sampler; 1.0 means "no crop this time"
NO_AUG_FRACTION = 0.1                                # D-FINE: the last 8 of 80 epochs


def augment(sample, ctx):
    if ctx.epoch >= ctx.epochs * (1 - NO_AUG_FRACTION):
        return sample
    rng = ctx.rng
    if rng.random() < P_PHOTOMETRIC:
        sample = Sample(photometric(sample.image, rng), sample.boxes, sample.labels, sample.path)
    if rng.random() < P_ZOOM_OUT:
        sample = zoom_out(sample, rng)
    if rng.random() < P_IOU_CROP:
        sample = iou_crop(sample, rng)
    if rng.random() < P_FLIP:
        sample = hflip(sample)
    return sample


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


def zoom_out(sample, rng):
    """Place the image at a random spot of a canvas up to ZOOM_OUT_MAX times bigger."""
    h, w = sample.image.shape[:2]
    r = rng.uniform(1.0, ZOOM_OUT_MAX)
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
