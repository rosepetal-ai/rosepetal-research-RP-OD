"""The COCO benchmark augmentations (benchmarks/coco/augment.py) keep every box on its object."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rpod.data.dataset import Sample

spec = importlib.util.spec_from_file_location('augment', Path(__file__).parents[1] / 'benchmarks/coco/augment.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

COLORS = [(220, 30, 30), (30, 200, 40), (40, 60, 230), (230, 210, 20)]


def scene():
    image = np.full((240, 320, 3), 60, np.uint8)
    boxes = np.array([[10, 20, 90, 100], [120, 40, 200, 90], [210, 150, 300, 230], [40, 160, 100, 220]], np.float32)
    for (x0, y0, x1, y1), c in zip(boxes.astype(int), COLORS):
        image[y0:y1, x0:x1] = c
    return Sample(image, boxes, np.arange(4), 'scene')


def test_geometric_augmentations_keep_boxes_on_their_objects():
    A.P_PHOTOMETRIC = 0.0                                   # colors must stay exact to find the objects again
    seen = set()
    for i in range(300):
        out = A.augment(scene(), SimpleNamespace(epoch=0, epochs=10, rng=np.random.default_rng(i)))
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


def test_last_epochs_are_clean():
    s = scene()
    out = A.augment(s, SimpleNamespace(epoch=95, epochs=100, rng=np.random.default_rng(0)))
    assert out is s
