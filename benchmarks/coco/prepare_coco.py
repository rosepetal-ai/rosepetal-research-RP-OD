# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Download COCO 2017 and lay it out the way RP-OD reads datasets.

    <root>/images/train2017, images/val2017     official jpgs
    <root>/labels/train2017, labels/val2017     one `class cx cy w h` line per object, normalized
    <root>/annotations/instances_val2017.json   official annotations, for the final pycocotools evaluation
    <root>/coco.yaml                            RP-OD data file (no test split: COCO test-dev is scored by a server)
    <root>/classes.json                         class index -> COCO category id

Crowd boxes are left out of the labels (as in every YOLO-format COCO); pycocotools still treats them as ignore regions
in the final evaluation. Downloads resume if interrupted; every step is skipped when its output already exists.

    python prepare_coco.py --root /work/coco
"""
import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

URL = 'http://images.cocodataset.org'
FILES = {'annotations_trainval2017.zip': f'{URL}/annotations/annotations_trainval2017.zip',
         'val2017.zip': f'{URL}/zips/val2017.zip',
         'train2017.zip': f'{URL}/zips/train2017.zip'}


def download(url, dst):
    done = dst.stat().st_size if dst.exists() else 0
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD')) as r:
        total = int(r.headers['Content-Length'])
    if done == total:
        return
    req = urllib.request.Request(url, headers={'Range': f'bytes={done}-'} if done else {})
    with urllib.request.urlopen(req) as r, open(dst, 'ab' if done else 'wb') as f:
        step = total // 20
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if done // step != (done - len(chunk)) // step:
                print(f'  {dst.name}: {100 * done / total:.0f}% of {total / 2 ** 30:.1f} GB', flush=True)


def convert(ann_file, split, root):
    """Official json -> YOLO label files (class index = position of the category sorted by COCO id)."""
    data = json.loads(ann_file.read_text())
    cats = sorted(data['categories'], key=lambda c: c['id'])
    index = {c['id']: i for i, c in enumerate(cats)}
    images = {im['id']: im for im in data['images']}
    lines = {i: [] for i in images}
    for a in data['annotations']:
        im = images[a['image_id']]
        x, y, w, h = a['bbox']
        x0, y0 = max(x, 0.0), max(y, 0.0)
        x1, y1 = min(x + w, im['width']), min(y + h, im['height'])
        if a.get('iscrowd') or x1 - x0 < 1 or y1 - y0 < 1:
            continue
        W, H = im['width'], im['height']
        lines[a['image_id']].append(f"{index[a['category_id']]} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                                    f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    out = root / 'labels' / split
    out.mkdir(parents=True, exist_ok=True)
    for i, im in images.items():          # images without objects get no label file: RP-OD reads them as background
        if lines[i]:
            (out / Path(im['file_name']).with_suffix('.txt').name).write_text('\n'.join(lines[i]) + '\n')
    print(f'  {split}: {len(images)} images, {sum(map(len, lines.values()))} boxes', flush=True)


def main(root, keep_zips=False):
    root = Path(root)
    (root / 'images').mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = root / ('annotations' if name.startswith('annotations') else f'images/{name[:-4]}')
        if target.exists():
            continue
        print(f'downloading {name}', flush=True)
        zip_path = root / name
        download(url, zip_path)
        print(f'extracting {name}', flush=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root if name.startswith('annotations') else root / 'images')
        if not keep_zips:
            zip_path.unlink()
    for split in ('val2017', 'train2017'):
        if not (root / 'labels' / split).exists():
            print(f'converting {split}', flush=True)
            convert(root / 'annotations' / f'instances_{split}.json', split, root)
    if not (root / 'coco.yaml').exists():
        cats = sorted(json.loads((root / 'annotations' / 'instances_val2017.json').read_text())['categories'],
                      key=lambda c: c['id'])
        names, cat_ids = [c['name'] for c in cats], [c['id'] for c in cats]
        (root / 'classes.json').write_text(json.dumps({'names': names, 'category_ids': cat_ids}))
        (root / 'coco.yaml').write_text('train: images/train2017\nval: images/val2017\nnames:\n'
                                        + ''.join(f'  {i}: {n}\n' for i, n in enumerate(names)))
    print('COCO ready:', root / 'coco.yaml', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='coco')
    p.add_argument('--keep-zips', action='store_true')
    main(**vars(p.parse_args()))
