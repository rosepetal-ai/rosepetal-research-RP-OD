"""COCO-pretrained weights of each size: release files of the RP-OD repository, downloaded the first time a model of
that size is trained and cached in ~/.cache/rpod (the RPOD_CACHE environment variable moves it).

A new model starts from them: everything loads except the class heads, which depend on the dataset.
"""
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

REPO = 'https://github.com/rosepetal-ai/rosepetal-research-RP-OD'
# size -> (release tag, file name, sha256); a size missing here has no COCO weights published yet
PRETRAINED = {}


def cache_dir():
    return Path(os.environ.get('RPOD_CACHE', Path.home() / '.cache' / 'rpod'))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pretrained_weights(size, log=print):
    """Path of the COCO weights of `size`, downloaded and checked the first time; None if none are published."""
    if size not in PRETRAINED:
        return None
    tag, name, sha256 = PRETRAINED[size]
    path = cache_dir() / tag / name
    if path.exists():
        return path
    url = f'{REPO}/releases/download/{tag}/{name}'
    log(f'downloading the COCO weights of size {size}: {url}')
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    with urllib.request.urlopen(url) as response, open(part, 'wb') as f:
        shutil.copyfileobj(response, f)
    if _sha256(part) != sha256:
        part.unlink()
        raise RuntimeError(f'{url} arrived corrupted (checksum mismatch); try again')
    part.rename(path)
    return path
