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
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

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

    def read(self, record):
        """Sample at original resolution, boxes in pixels xyxy. Fresh arrays: augmentations can change them freely."""
        image = read_image(record['path'])
        h, w = image.shape[:2]
        b = record['boxes'].astype(np.float32)
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        labels = record['cls'] if self.id_map is None else self.id_map[record['cls']]
        return Sample(image, xyxy.reshape(-1, 4) * np.array([w, h, w, h], np.float32), labels.copy(), record['path'])

    def __getitem__(self, i):
        record = self.records[i]
        sample = self.read(record)
        if self.augment is not None:
            epoch = self._epoch.value
            worker = get_worker_info()
            rng = np.random.default_rng([self.seed, epoch, worker.id if worker else 0, i])
            sample = self._checked(self.augment(sample, AugmentContext(self, epoch, self.epochs, self.imgsz, rng)), record)
        image, geom = letterbox(sample.image, *self.imgsz)
        scale, _, _, top, left = geom
        out_h, out_w = self.imgsz
        b = sample.boxes * scale + np.array([left, top, left, top], np.float32)
        b = b / np.array([out_w, out_h, out_w, out_h], np.float32)
        cxcywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        target = {'labels': torch.from_numpy(np.asarray(sample.labels, np.int64)),
                  'boxes': torch.from_numpy(cxcywh.reshape(-1, 4).astype(np.float32))}
        meta = {'path': record['path'], 'orig_shape': sample.image.shape[:2], 'geom': geom}
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
