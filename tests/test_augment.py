# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The default augmentation (rpod.data.augment, the COCO recipe) keeps every box on its object, and its phases follow
the end of training."""
import inspect
from types import SimpleNamespace

import numpy as np
import yaml

import rpod.data.augment as A
from rpod import RPOD
from rpod.data import DetectionDataset, check_dataset
from rpod.data.dataset import Sample

from .test_dataset import make_dataset

COLORS = [(220, 30, 30), (30, 200, 40), (40, 60, 230), (230, 210, 20)]


def scene():
    image = np.full((240, 320, 3), 60, np.uint8)
    boxes = np.array([[10, 20, 90, 100], [120, 40, 200, 90], [210, 150, 300, 230], [40, 160, 100, 220]], np.float32)
    for (x0, y0, x1, y1), c in zip(boxes.astype(int), COLORS):
        image[y0:y1, x0:x1] = c
    return Sample(image, boxes, np.arange(4), 'scene')


def test_geometric_augmentations_keep_boxes_on_their_objects(monkeypatch):
    monkeypatch.setattr(A, 'P_PHOTOMETRIC', 0.0)            # colors must stay exact to find the objects again
    monkeypatch.setattr(A, 'P_MOSAIC', 0.0)                 # mosaic has its own test
    seen = set()
    for i in range(300):
        out = A.default_augment(scene(), SimpleNamespace(epoch=0, epochs=10, rng=np.random.default_rng(i)))
        h, w = out.image.shape[:2]
        seen.add((h, w))
        assert len(out.boxes) == len(out.labels)
        for (x0, y0, x1, y1), c in zip(out.boxes, out.labels):
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
            inner = out.image[int(np.ceil(y0)):int(y1), int(np.ceil(x0)):int(x1)].reshape(-1, 3)
            if len(inner):
                assert (np.abs(inner.astype(int) - COLORS[c]).max(1) == 0).mean() > 0.99
    assert len(seen) > 50                                   # zoom-out and crops really change the image


def test_photometric_changes_colors_not_geometry():
    s = scene()
    out = A.photometric(s.image, np.random.default_rng(3))
    assert out.shape == s.image.shape and out.dtype == np.uint8


def test_last_epochs_are_clean(monkeypatch):
    monkeypatch.setattr(A, 'P_MOSAIC', 0.0)
    s = scene()
    for epoch, epochs in ((96, 100), (99, 100), (69, 72), (5, 6)):           # last 4 %, at least one epoch
        assert A.default_augment(s, SimpleNamespace(epoch=epoch, epochs=epochs, rng=np.random.default_rng(0))) is s
    changed = [A.default_augment(s, SimpleNamespace(epoch=95, epochs=100, rng=np.random.default_rng(k))) is not s
               for k in range(10)]
    assert any(changed)                                                        # epoch 95 of 100 still augments



def test_mosaic_keeps_boxes_on_their_objects(monkeypatch):
    monkeypatch.setattr(A, 'P_PHOTOMETRIC', 0.0)
    for i in range(100):
        rng = np.random.default_rng(i)
        ctx = SimpleNamespace(epoch=0, epochs=10, rng=rng, imgsz=[160, 200], sample=lambda k: [scene() for _ in range(k)])
        out = A.mosaic([scene()] + ctx.sample(3), ctx.imgsz, rng)
        assert out.image.shape == (320, 400, 3) and len(out.boxes) == len(out.labels) == 16
        for (x0, y0, x1, y1), c in zip(out.boxes, out.labels):
            if x1 - x0 < 2 or y1 - y0 < 2:                          # cut by the canvas edge: the dataset drops it
                continue
            inner = out.image[int(np.ceil(y0)) + 1:int(y1) - 1, int(np.ceil(x0)) + 1:int(x1) - 1].reshape(-1, 3)
            if len(inner):
                assert (np.abs(inner.astype(int) - COLORS[c]).max(1) <= 2).mean() > 0.95   # bilinear resize at edges


def test_mosaic_only_in_the_first_quarter(monkeypatch):
    calls = []
    monkeypatch.setattr(A, 'P_MOSAIC', 1.0)
    monkeypatch.setattr(A, 'P_PHOTOMETRIC', 0.0)
    monkeypatch.setattr(A, 'mosaic', lambda samples, imgsz, rng: calls.append(1) or samples[0])
    for epoch in range(100):
        A.default_augment(scene(), SimpleNamespace(epoch=epoch, epochs=100, rng=np.random.default_rng(epoch),
                                                   imgsz=[160, 160], sample=lambda k: [scene() for _ in range(k)]))
    assert len(calls) == 25


def test_train_augments_by_default():
    default = inspect.signature(RPOD.train).parameters['augment'].default
    assert default is A.default_augment


def test_the_end_of_training_reaches_the_augmentation(tmp_path):
    ds = check_dataset(make_dataset(tmp_path), use_cache=False)
    seen = []
    train = DetectionDataset(ds['records']['train'], [64, 64], 3, lambda s, ctx: seen.append(ctx.epochs) or s,
                             epochs=100)
    train[0]
    train.set_epochs(40)                  # patience brought the end forward
    train[0]
    assert seen == [100, 40] and train.epochs == 40


def test_patience_moves_the_augmentation_end(tmp_path, monkeypatch):
    """patience 0 starts the final decay after the first epoch: 10 epochs become 1 + 2 (20 % final phase)."""
    monkeypatch.setenv('RPOD_WORKERS', '0')               # the augmentation runs here, so it can record
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 2, 'val': 1})
    seen = {}

    def record(sample, ctx):
        seen.setdefault(ctx.epoch, set()).add(ctx.epochs)
        return sample

    res = RPOD('s').train(data, imgsz=128, epochs=10, patience=0, batch=2, device='cpu', augment=record,
                          pretrained=False)
    assert seen == {0: {10}, 1: {3}, 2: {3}}
    assert yaml.safe_load(open(res['dir'] / 'args.yaml'))['augment'] == 'record'
