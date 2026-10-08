"""Dataset for training and validation: read -> (optional) augment -> letterbox exactly like production.

No augmentation happens here. An external library can plug in through one function, applied to training samples only:

    def augment(sample, ctx):
        # sample.image   uint8 RGB [H, W, 3] at original resolution
        # sample.boxes   float [N, 4] xyxy in pixels of that image
        # sample.labels  int [N]
        # sample.path    where it comes from
        # ctx.sample(k)  k more random training samples, read and not augmented (mosaic, mixup, copy-paste...)
        # ctx.epoch, ctx.epochs, ctx.imgsz ([height, width] the image will be letterboxed into), ctx.rng
        return sample   # any size: the letterbox comes after
"""
import math
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, get_worker_info

from .check import DatasetError
from .letterbox import letterbox
from .sources import read_image

MIN_BOX_PIXELS = 1.0


@dataclass
class Sample:
    image: np.ndarray
    boxes: np.ndarray
    labels: np.ndarray
    path: str = None


class AugmentError(RuntimeError):
    pass


class AugmentContext:
    def __init__(self, dataset, epoch, epochs, imgsz, rng):
        self._dataset = dataset
        self.epoch, self.epochs, self.imgsz, self.rng = epoch, epochs, list(imgsz), rng

    def sample(self, k=1):
        """k random training samples, read from disk and not augmented."""
        pool = self._dataset.pool
        picks = self.rng.choice(len(pool), size=k, replace=len(pool) < k)
        return [self._dataset.read(pool[i]) for i in picks]


def class_map(dataset_names, model_names):
    """Array mapping dataset class ids to model class ids, by name. The sets must be identical; the order may differ."""
    ds, md = set(dataset_names.values()), set(model_names.values())
    if ds != md:
        missing, extra = sorted(md - ds), sorted(ds - md)
        raise DatasetError('the dataset classes do not match the model classes'
                           + (f'; missing in the dataset: {missing}' if missing else '')
                           + (f'; not in the model: {extra}' if extra else ''))
    by_name = {n: i for i, n in model_names.items()}
    return np.array([by_name[dataset_names[i]] for i in range(len(dataset_names))], np.int64)


class DetectionDataset(Dataset):
    def __init__(self, records, imgsz, num_classes, augment=None, pool=None, epochs=1, seed=0, id_map=None):
        """records: from check_dataset. pool: records ctx.sample() draws from (the training split)."""
        self.records, self.imgsz, self.num_classes = records, list(imgsz), num_classes
        self.augment, self.pool = augment, pool if pool is not None else records
        self.epochs, self.seed, self.id_map = epochs, seed, id_map
        self._epoch = mp.Value('i', 0)   # shared with the dataloader workers

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self._epoch.value = int(epoch)

    def ground_truth(self, i):
        """Boxes xyxy in pixels of the original image and model class ids, without reading the image."""
        r = self.records[i]
        b = r['boxes'].astype(np.float64) * np.array([r['w'], r['h'], r['w'], r['h']])
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        labels = r['cls'] if self.id_map is None else self.id_map[r['cls']]
        return xyxy.reshape(-1, 4), labels

    def read(self, record):
        """Sample at original resolution, boxes in pixels xyxy. Fresh arrays: augmentations can change them freely."""
        image = read_image(record['path'])
        h, w = image.shape[:2]
        b = record['boxes'].astype(np.float32)
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        labels = record['cls'] if self.id_map is None else self.id_map[record['cls']]
        return Sample(image, xyxy.reshape(-1, 4) * np.array([w, h, w, h], np.float32), labels.copy(), record['path'])

    def __getitem__(self, item):
        i, size = item if isinstance(item, tuple) else (item, self.imgsz)   # (index, [h, w]) from multi-scale batches
        size = list(size)
        record = self.records[i]
        sample = self.read(record)
        if self.augment is not None:
            epoch = self._epoch.value
            worker = get_worker_info()
            rng = np.random.default_rng([self.seed, epoch, worker.id if worker else 0, i])
            sample = self._checked(self.augment(sample, AugmentContext(self, epoch, self.epochs, size, rng)), record)
        image, geom = letterbox(sample.image, *size)
        scale, _, _, top, left = geom
        out_h, out_w = size
        b = sample.boxes * scale + np.array([left, top, left, top], np.float32)
        b = b / np.array([out_w, out_h, out_w, out_h], np.float32)
        cxcywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        target = {'labels': torch.from_numpy(np.asarray(sample.labels, np.int64)),
                  'boxes': torch.from_numpy(cxcywh.reshape(-1, 4).astype(np.float32))}
        meta = {'index': i, 'path': record['path'], 'orig_shape': sample.image.shape[:2], 'geom': geom}
        return image, target, meta

    def _checked(self, sample, record):
        """Validate what the augmentation returned: clip boxes to the image, drop the degenerate ones."""
        where = f"augment() on {record['path']}"
        if not isinstance(sample, Sample):
            raise AugmentError(f'{where}: must return the Sample it received (or a new rpod Sample)')
        img = sample.image
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise AugmentError(f'{where}: image must be uint8 RGB [H, W, 3], got '
                               f"{getattr(img, 'dtype', type(img))} {getattr(img, 'shape', '')}")
        boxes = np.asarray(sample.boxes, np.float32).reshape(-1, 4)
        labels = np.asarray(sample.labels).reshape(-1)
        if len(boxes) != len(labels):
            raise AugmentError(f'{where}: {len(boxes)} boxes but {len(labels)} labels')
        if len(labels) and (labels.dtype.kind not in 'iu' or labels.min() < 0 or labels.max() >= self.num_classes):
            raise AugmentError(f'{where}: labels must be integers in 0..{self.num_classes - 1}')
        h, w = img.shape[:2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
        keep = ((boxes[:, 2] - boxes[:, 0]) >= MIN_BOX_PIXELS) & ((boxes[:, 3] - boxes[:, 1]) >= MIN_BOX_PIXELS)
        return Sample(img, boxes[keep], labels[keep].astype(np.int64), sample.path)


def collate(batch):
    images, targets, metas = zip(*batch)
    return torch.stack(images), list(targets), list(metas)


# Multi-scale training: (until this fraction of the epochs, each side from low x to high x imgsz). Width and height
# are drawn separately for every batch; after the last phase, every batch at imgsz.
MULTISCALE = ((0.25, 0.5, 1.5),          # first quarter: wide range, also extreme aspect ratios (up to 3:1)
              (5 / 6, 0.75, 1.25))       # then a moderate range (480-800 at 640), until the last sixth


def side_choices(side, low, high, stride=32, minimum=32):
    """The multiples of `stride` from low x side to high x side, none below `minimum`."""
    lo = max(minimum, math.ceil(side * low / stride) * stride)
    return list(range(lo, max(lo, math.floor(side * high / stride) * stride) + 1, stride))


class MultiScaleBatchSampler(Sampler):
    """Shuffled batches of (index, size) where all the images of a batch share one size, its height and width drawn
    separately from the range of the phase the epoch is in (`phases`, see MULTISCALE); after the last phase, `base`.

    Each image is then letterboxed directly from the original at that size (as in production, never upscaled from
    a smaller letterbox). The trainer sets `epoch` (it seeds the order and the sizes, so a resumed run gets the same
    batches) and `fixed` (True: every batch at `base`, for the clean last epochs)."""

    def __init__(self, n, batch, base, epochs, phases=MULTISCALE, seed=0, drop_last=True, stride=32, min_side=32):
        self.n, self.batch, self.base = n, batch, tuple(base)
        self.phases = [(round(f * epochs), side_choices(base[0], lo, hi, stride, min_side),
                        side_choices(base[1], lo, hi, stride, min_side)) for f, lo, hi in phases]
        self.seed, self.drop_last, self.epoch, self.fixed = seed, drop_last, 0, False

    def __len__(self):
        return self.n // self.batch if self.drop_last else math.ceil(self.n / self.batch)

    def choices(self, epoch):
        """(heights, widths) the batches of this epoch are drawn from; None: every batch at `base`."""
        return next(((hs, ws) for end, hs, ws in self.phases if epoch < end), None)

    def largest(self):
        return max([self.base] + [(max(hs), max(ws)) for _, hs, ws in self.phases], key=lambda x: x[0] * x[1])

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(self.n)
        choices = None if self.fixed else self.choices(self.epoch)
        for k in range(len(self)):
            size = self.base if choices is None else (int(rng.choice(choices[0])), int(rng.choice(choices[1])))
            yield [(int(i), size) for i in order[k * self.batch:(k + 1) * self.batch]]
