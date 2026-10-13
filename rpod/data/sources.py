# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Turn whatever `predict` receives into RGB uint8 arrays, without ever modifying the caller's images.

Accepted: a numpy array (H x W, or H x W x 1/3/4) with a color tag, a file path, a folder, or a list of these.
Only 8-bit images are accepted, as in production.
"""
from pathlib import Path

import numpy as np
from PIL import Image

COLORS = ('RGB', 'BGR', 'GRAY')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def to_rgb(image, color='RGB'):
    """uint8 H x W x 3 RGB view or copy of `image`. Read-only and non-contiguous arrays are fine."""
    color = color.upper()
    if color not in COLORS:
        raise ValueError(f"color must be one of {', '.join(COLORS)}, got {color!r}")
    if image.dtype != np.uint8:
        raise ValueError(f'only 8-bit images are supported, got dtype {image.dtype}')
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError(f'expected an image H x W, H x W x 1, 3 or 4, got shape {image.shape}')
    c = image.shape[2]
    if c == 1:
        return np.repeat(image, 3, axis=2)
    if color == 'GRAY':
        raise ValueError(f'color="GRAY" but the image has {c} channels')
    return image[..., 2::-1] if color == 'BGR' else image[..., :3]


def read_image(path):
    with Image.open(path) as im:
        if im.mode in ('I', 'I;16', 'I;16B', 'I;16L', 'F'):
            raise ValueError(f'{path}: only 8-bit images are supported (mode {im.mode})')
        if im.mode not in ('RGB', 'L'):
            im = im.convert('RGB')
        return to_rgb(np.asarray(im))


def load_images(source, color='RGB'):
    """Returns a list of (rgb_array, path_or_None)."""
    if isinstance(source, np.ndarray):
        return [(to_rgb(source, color), None)]
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
            if not files:
                raise FileNotFoundError(f'no images in {p}')
            return [(read_image(f), str(f)) for f in files]
        if not p.exists():
            raise FileNotFoundError(p)
        return [(read_image(p), str(p))]
    if isinstance(source, (list, tuple)):
        return [item for s in source for item in load_images(s, color)]
    raise TypeError(f'unsupported source type {type(source).__name__}')
