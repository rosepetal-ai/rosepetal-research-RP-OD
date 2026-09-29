"""YOLO datasets: read data.yaml, check every image and label once, cache the result.

data.yaml:
    path: /data/parts          # optional, defaults to the folder of the yaml
    train: images/train        # required: a folder, a .txt listing images, or a list of those
    val: images/val            # required
    test: images/test          # optional, never used for any training decision
    names: {0: scratch, 1: screw}   # or a list

Labels live in the parallel 'labels' folder (images/x/a.jpg -> labels/x/a.txt), one line per object:
`class cx cy w h`, normalized 0-1. A missing or empty label file means an image without objects.

Errors (the dataset cannot be used as it is) stop everything and list every offending file with the reason.
Warnings (worth knowing, not blocking) are reported and training goes on.
"""
import hashlib
import io
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .sources import IMAGE_EXTENSIONS

CACHE_VERSION = 1
CACHE_DIR = Path(os.environ.get('RPOD_CACHE', Path.home() / '.cache' / 'rpod')) / 'datasets'
BOX_TOLERANCE = 1e-3            # coordinates this far outside [0, 1] are clipped silently
EIGHT_BIT_MODES = {'1', 'L', 'LA', 'P', 'PA', 'RGB', 'RGBA', 'RGBX', 'CMYK', 'YCbCr'}


class DatasetError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ data.yaml
def load_data_yaml(path):
    """Returns {'root', 'names', 'splits': {'train': [...], 'val': [...], 'test': [...] or None}}."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f'data file not found: {path}')
    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    root = Path(spec.get('path') or path.parent)
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    names = parse_names(spec.get('names'), path)
    if 'nc' in spec and int(spec['nc']) != len(names):
        raise DatasetError(f"{path}: nc is {spec['nc']} but names has {len(names)} classes")
    splits = {}
    for split in ('train', 'val', 'test'):
        if not spec.get(split):
            if split != 'test':
                raise DatasetError(f"{path}: '{split}' is required (a folder of images or a .txt listing them)")
            splits[split] = None
            continue
        images = []
        for item in spec[split] if isinstance(spec[split], list) else [spec[split]]:
            images += _list_images(root, item, path, split)
        if not images:
            raise DatasetError(f"{path}: '{split}' contains no images")
        splits[split] = images
    return {'root': root, 'names': names, 'splits': splits}


def parse_names(names, where='data.yaml'):
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise DatasetError(f"{where}: 'names' is required, as a list or as {{id: name}}")
    try:
        names = {int(k): str(v) for k, v in names.items()}
    except (TypeError, ValueError):
        raise DatasetError(f"{where}: class ids in 'names' must be integers") from None
    if sorted(names) != list(range(len(names))):
        raise DatasetError(f"{where}: class ids must be 0..{len(names) - 1} without gaps, got {sorted(names)}")
    if len(set(names.values())) != len(names):
        raise DatasetError(f'{where}: class names must be unique')
    return names


def _list_images(root, item, yaml_path, split):
    p = Path(item)
    p = p if p.is_absolute() else root / p
    if p.is_dir():
        return sorted(str(f) for f in p.rglob('*') if f.suffix.lower() in IMAGE_EXTENSIONS)
    if p.suffix == '.txt' and p.exists():
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        return [str(Path(ln) if Path(ln).is_absolute() else (root / ln)) for ln in lines]
    raise DatasetError(f"{yaml_path}: '{split}' entry {item!r} is neither a folder nor a .txt file ({p})")


def label_path(image_path):
    """images/x/a.jpg -> labels/x/a.txt (last 'images' folder of the path); next to the image otherwise."""
    parts = list(Path(image_path).parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == 'images':
            parts[i] = 'labels'
            break
    return str(Path(*parts).with_suffix('.txt'))


# ------------------------------------------------------------------------------------------------ checking
def _check_one(image_path, num_classes):
    """Returns (record or None, problems) for one image and its label file."""
    problems = []
    try:
        data = Path(image_path).read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            mode, (w, h) = im.mode, im.size
    except FileNotFoundError:
        return None, [('error', image_path, 'image file not found')]
    except Exception as e:  # noqa: BLE001 - any decoding failure makes the image unusable
        return None, [('error', image_path, f'unreadable image ({type(e).__name__})')]
    if mode not in EIGHT_BIT_MODES:
        problems.append(('error', image_path, f'not an 8-bit image (mode {mode}); the platform only works with 8 bits'))
    cls, boxes = [], []
    lp = Path(label_path(image_path))
    if lp.exists():
        for n, line in enumerate(lp.read_text().splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            where = f'{lp}:{n}'
            if len(parts) != 5:
                reason = ('segmentation polygon: only detection boxes are supported (class cx cy w h)'
                          if len(parts) > 5 and len(parts) % 2 == 1 else f'expected 5 values (class cx cy w h), got {len(parts)}')
                problems.append(('error', where, reason))
                continue
            try:
                c, *xywh = (float(v) for v in parts)
            except ValueError:
                problems.append(('error', where, 'values are not numbers'))
                continue
            if c != int(c) or not 0 <= c < num_classes:
                problems.append(('error', where, f'class {parts[0]} is not one of 0..{num_classes - 1}'))
                continue
            x, y, bw, bh = xywh
            x0, y0, x1, y1 = x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2
            if bw <= 0 or bh <= 0:
                problems.append(('error', where, 'box width and height must be positive'))
                continue
            if min(x0, y0) < -BOX_TOLERANCE or max(x1, y1) > 1 + BOX_TOLERANCE:
                problems.append(('error', where, 'box goes outside the image (coordinates must be normalized 0-1)'))
                continue
            x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)
            cls.append(int(c))
            boxes.append([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])
    record = {'path': image_path, 'w': w, 'h': h, 'hash': hashlib.md5(data).hexdigest(),
              'cls': np.array(cls, np.int64), 'boxes': np.array(boxes, np.float32).reshape(-1, 4)}
    return record, problems


def _cache_key(images, num_classes):
    h = hashlib.sha1(f'{CACHE_VERSION}:{num_classes}'.encode())
    for p in images:
        h.update(p.encode())
        for f in (p, label_path(p)):
            try:
                st = os.stat(f)
                h.update(f'{st.st_size}:{st.st_mtime_ns}'.encode())
            except FileNotFoundError:
                h.update(b'-')
    return h.hexdigest()


def scan_split(images, num_classes, workers=8, use_cache=True):
    """Check every image and label of one split. Returns (records, problems). Cached until any file changes."""
    key = _cache_key(images, num_classes)
    cache = CACHE_DIR / f'{key}.pkl'
    if use_cache and cache.exists():
        with open(cache, 'rb') as f:
            return pickle.load(f)
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda p: _check_one(p, num_classes), images))
    records = [r for r, _ in results if r is not None]
    problems = [p for _, ps in results for p in ps]
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f'.tmp{os.getpid()}')
            with open(tmp, 'wb') as f:
                pickle.dump((records, problems), f)
            os.replace(tmp, cache)
        except OSError:
            pass  # a read-only home only costs a re-scan next time
    return records, problems


# ------------------------------------------------------------------------------------------------ report
def check_dataset(data_yaml, use_cache=True):
    """Check a whole dataset. Raises DatasetError listing every error; returns the dataset and the warnings."""
    ds = load_data_yaml(data_yaml)
    num_classes = len(ds['names'])
    records, problems = {}, []
    for split, images in ds['splits'].items():
        if images is None:
            continue
        records[split], p = scan_split(images, num_classes, use_cache=use_cache)
        problems += [(sev, split, where, why) for sev, where, why in p]
    if ds['splits']['test'] is None:
        problems.append(('warning', 'test', str(data_yaml), 'no test split: validation metrics are optimistic, '
                         'because validation is used to pick the best epoch and the threshold'))
    seen = {}
    for split, recs in records.items():
        for r in recs:
            seen.setdefault(r['hash'], []).append((split, r['path']))
    for copies in seen.values():
        splits = {s for s, _ in copies}
        if len(copies) > 1:
            kind = f"across {' and '.join(sorted(splits))}" if len(splits) > 1 else f'inside {copies[0][0]}'
            problems.append(('warning', copies[0][0], copies[0][1],
                             f"identical image {kind}: {', '.join(p for _, p in copies[1:])}"))
    errors = [p for p in problems if p[0] == 'error']
    warnings = [p for p in problems if p[0] == 'warning']
    if errors:
        raise DatasetError(format_problems(errors, f'{len(errors)} error(s) in {data_yaml}, fix them and run again'))
    stats = {s: {'images': len(r), 'objects': int(sum(len(x['cls']) for x in r)),
                 'backgrounds': sum(1 for x in r if not len(x['cls']))} for s, r in records.items()}
    stats['max_objects_per_image'] = max((len(x['cls']) for x in records['train']), default=0)
    return {**ds, 'records': records, 'warnings': warnings, 'stats': stats}


def format_problems(problems, title, limit=50):
    lines = [title + ':']
    for _, split, where, why in problems[:limit]:
        lines.append(f'  [{split}] {where}: {why}')
    if len(problems) > limit:
        lines.append(f'  ... and {len(problems) - limit} more')
    return '\n'.join(lines)
