import numpy as np
import pytest
import torch
from PIL import Image

from rpod.data import (AugmentError, DatasetError, DetectionDataset, Sample, check_dataset, class_map, collate,
                       letterbox_geometry)
from rpod.data.check import label_path

NAMES = ['scratch', 'screw', 'label']


def make_dataset(root, splits=None, seed=0, mode='RGB'):
    """A YOLO dataset: images/<split>/*.png + labels/<split>/*.txt + data.yaml. Returns the yaml path."""
    rng = np.random.default_rng(seed)
    splits = splits or {'train': 6, 'val': 3, 'test': 2}
    for split, n in splits.items():
        (root / 'images' / split).mkdir(parents=True)
        (root / 'labels' / split).mkdir(parents=True)
        for i in range(n):
            img = rng.integers(0, 256, (120, 200, 3), dtype=np.uint8)
            Image.fromarray(img).convert(mode).save(root / 'images' / split / f'{split}{i}.png')
            lines = [f'{rng.integers(0, 3)} {rng.uniform(.3, .7):.4f} {rng.uniform(.3, .7):.4f} 0.2000 0.1000'
                     for _ in range(rng.integers(0, 4))]
            (root / 'labels' / split / f'{split}{i}.txt').write_text('\n'.join(lines))
    text = 'path: .\n' + ''.join(f'{s}: images/{s}\n' for s in splits) + f'names: {NAMES}\n'
    (root / 'data.yaml').write_text(text)
    return root / 'data.yaml'


def test_valid_dataset(tmp_path):
    ds = check_dataset(make_dataset(tmp_path), use_cache=False)
    assert ds['names'] == dict(enumerate(NAMES))
    assert ds['stats']['train']['images'] == 6 and ds['warnings'] == []


def test_relative_yaml_without_path_key(tmp_path, monkeypatch):
    y = make_dataset(tmp_path / 'd', {'train': 2, 'val': 1})
    y.write_text(y.read_text().replace('path: .\n', ''))
    monkeypatch.chdir(tmp_path)
    assert check_dataset('d/data.yaml', use_cache=False)['stats']['train']['images'] == 2


def test_label_path_convention():
    assert label_path('/d/images/train/a.jpg') == '/d/labels/train/a.txt'
    assert label_path('/d/pics/a.jpg') == '/d/pics/a.txt'


def test_val_is_mandatory_and_test_optional(tmp_path):
    y = make_dataset(tmp_path, {'train': 2, 'test': 1})
    with pytest.raises(DatasetError, match="'val' is required"):
        check_dataset(y, use_cache=False)
    ds = check_dataset(make_dataset(tmp_path / 'b', {'train': 2, 'val': 1}), use_cache=False)
    assert any('no test split' in w[3] for w in ds['warnings'])


def test_errors_list_every_file_and_reason(tmp_path):
    y = make_dataset(tmp_path)
    (tmp_path / 'labels/train/train0.txt').write_text('0 0.1 0.2 0.3 0.4 0.5 0.6\n')      # polygon
    (tmp_path / 'labels/train/train1.txt').write_text('7 0.5 0.5 0.1 0.1\n')             # class out of range
    (tmp_path / 'labels/val/val0.txt').write_text('0 0.95 0.5 0.3 0.1\n')                # outside the image
    (tmp_path / 'images/val/val1.png').write_bytes(b'not an image')
    Image.fromarray(np.zeros((8, 8), np.uint16)).save(tmp_path / 'images/test/test0.png')  # 16-bit
    with pytest.raises(DatasetError) as e:
        check_dataset(y, use_cache=False)
    msg = str(e.value)
    for text in ('segmentation polygon', 'class 7', 'outside the image', 'unreadable image', 'not an 8-bit'):
        assert text in msg, text


def test_duplicates_across_splits_are_reported(tmp_path):
    y = make_dataset(tmp_path)
    (tmp_path / 'images/test/test0.png').write_bytes((tmp_path / 'images/train/train3.png').read_bytes())
    ds = check_dataset(y, use_cache=False)
    assert any('identical image across test and train' in w[3] or 'identical image across train and test' in w[3]
               for w in ds['warnings'])


def test_cache_is_reused_and_invalidated(tmp_path, monkeypatch):
    monkeypatch.setattr('rpod.data.check.CACHE_DIR', tmp_path / 'cache')
    y = make_dataset(tmp_path / 'd')
    check_dataset(y)
    n = len(list((tmp_path / 'cache').glob('*.pkl')))
    check_dataset(y)
    assert len(list((tmp_path / 'cache').glob('*.pkl'))) == n            # nothing new: served from cache
    (tmp_path / 'd/labels/train/train0.txt').write_text('1 0.5 0.5 0.2 0.2\n')
    assert check_dataset(y)['records']['train'][0]['cls'].tolist() == [1]   # the change is seen


def test_targets_follow_the_letterbox(tmp_path):
    ds = check_dataset(make_dataset(tmp_path, {'train': 1, 'val': 1}), use_cache=False)
    rec = dict(ds['records']['train'][0], cls=np.array([0]), boxes=np.array([[0.5, 0.5, 0.2, 0.1]], np.float32))
    image, target, meta = DetectionDataset([rec], [128, 128], 3)[0]
    scale, rh, rw, top, left = letterbox_geometry(120, 200, 128, 128)
    assert image.shape == (3, 128, 128) and meta['orig_shape'] == (120, 200)
    cx, cy, w, h = target['boxes'][0].tolist()
    assert abs(cx - (100 * scale + left) / 128) < 1e-5 and abs(cy - (60 * scale + top) / 128) < 1e-5
    assert abs(w - 40 * scale / 128) < 1e-5 and abs(h - 12 * scale / 128) < 1e-5


def test_augment_hook_contract(tmp_path):
    ds = check_dataset(make_dataset(tmp_path), use_cache=False)
    seen = {}

    def augment(sample, ctx):
        seen.update(epoch=ctx.epoch, epochs=ctx.epochs, imgsz=ctx.imgsz, others=ctx.sample(3))
        sample.boxes = np.vstack([sample.boxes, [[-50, -50, 30, 30], [10, 10, 10.5, 60]]])   # one to clip, one to drop
        sample.labels = np.append(sample.labels, [1, 2])
        return sample

    train = DetectionDataset(ds['records']['train'], [96, 160], 3, augment, epochs=12)
    train.set_epoch(4)
    _, target, _ = train[0]
    assert seen['epoch'] == 4 and seen['epochs'] == 12 and seen['imgsz'] == [96, 160]
    assert len(seen['others']) == 3 and all(isinstance(s, Sample) for s in seen['others'])
    n_orig = len(ds['records']['train'][0]['cls'])
    assert len(target['labels']) == n_orig + 1                    # the 0.5 px wide box was dropped
    assert (target['boxes'][:, :2] >= 0).all()


def test_augment_output_is_validated(tmp_path):
    ds = check_dataset(make_dataset(tmp_path), use_cache=False)
    bad = DetectionDataset(ds['records']['train'], [96, 96], 3, lambda s, c: Sample(s.image[..., 0], s.boxes, s.labels))
    with pytest.raises(AugmentError, match='uint8 RGB'):
        bad[0]


def test_class_map_by_name_any_order():
    m = class_map({0: 'b', 1: 'a'}, {0: 'a', 1: 'b'})
    assert m.tolist() == [1, 0]
    with pytest.raises(DatasetError, match="missing in the dataset: \\['c'\\]"):
        class_map({0: 'a', 1: 'b'}, {0: 'a', 1: 'b', 2: 'c'})


def test_dataloader_workers_see_the_epoch(tmp_path):
    ds = check_dataset(make_dataset(tmp_path), use_cache=False)
    train = DetectionDataset(ds['records']['train'], [64, 64], 3, _tag)
    loader = torch.utils.data.DataLoader(train, batch_size=2, num_workers=2, collate_fn=collate, persistent_workers=True)
    for epoch in (1, 2):   # persistent workers must see each change
        train.set_epoch(epoch)
        for _, targets, _ in loader:
            assert all(int(t['labels'][-1]) == epoch % 3 for t in targets)


def _tag(sample, ctx):
    """Adds a 20x20 box whose class encodes the epoch the worker saw."""
    sample.boxes = np.vstack([sample.boxes, [[5, 5, 25, 25]]])
    sample.labels = np.append(sample.labels, ctx.epoch % 3)
    return sample
