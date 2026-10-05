import contextlib
import io

import numpy as np
import pytest

from rpod.utils.metrics import DEFAULT_THRESHOLD, evaluate

NAMES = {0: 'a', 1: 'b'}


def random_case(seed, n_img=40, n_cls=3):
    rng = np.random.default_rng(seed)
    preds, gts = [], []
    for _ in range(n_img):
        n = rng.integers(0, 6)
        xy = rng.uniform(0, 400, (n, 2))
        gb = np.concatenate([xy, xy + rng.uniform(8, 200, (n, 2))], 1)
        gc = rng.integers(0, n_cls, n)
        m = rng.integers(0, 10)
        fxy = rng.uniform(0, 500, (m, 2))
        pb = np.concatenate([gb + rng.normal(0, 6, gb.shape), np.concatenate([fxy, fxy + rng.uniform(5, 120, (m, 2))], 1)])
        pb[:, 2:] = np.maximum(pb[:, 2:], pb[:, :2] + 1)
        pc = np.concatenate([np.where(rng.random(n) < 0.15, rng.integers(0, n_cls, n), gc), rng.integers(0, n_cls, m)])
        preds.append((pb, rng.uniform(0.01, 1, len(pb)), pc))
        gts.append((gb, gc))
    return preds, gts


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_map_matches_pycocotools(seed):
    pytest.importorskip('pycocotools')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    preds, gts = random_case(seed)
    images, anns, dets = [], [], []
    for i, ((pb, ps, pc), (gb, gc)) in enumerate(zip(preds, gts)):
        images.append({'id': i, 'width': 640, 'height': 480})
        for b, c in zip(gb, gc):
            anns.append({'id': len(anns) + 1, 'image_id': i, 'category_id': int(c) + 1, 'iscrowd': 0,
                         'bbox': [b[0], b[1], b[2] - b[0], b[3] - b[1]], 'area': float((b[2] - b[0]) * (b[3] - b[1]))})
        dets += [{'image_id': i, 'category_id': int(c) + 1, 'score': float(s), 'bbox': [b[0], b[1], b[2] - b[0], b[3] - b[1]]}
                 for b, s, c in zip(pb, ps, pc)]
    coco = COCO()
    coco.dataset = {'images': images, 'annotations': anns, 'categories': [{'id': c + 1} for c in range(3)]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
        e = COCOeval(coco, coco.loadRes(dets), 'bbox')
        e.evaluate()
        e.accumulate()
        e.summarize()
    ours = evaluate(preds, gts, {0: '0', 1: '1', 2: '2'})
    assert abs(ours.map50_95 - e.stats[0]) < 1e-9 and abs(ours.map50 - e.stats[1]) < 1e-9
    for mine, theirs in ((ours.map_small, e.stats[3]), (ours.map_medium, e.stats[4]), (ours.map_large, e.stats[5])):
        assert abs(mine - theirs) < 1e-9                            # COCO small / medium / large, same rules


def test_sizes_are_measured_at_the_model_input():
    preds, gts = random_case(3)
    at_input = evaluate(preds, gts, {0: '0', 1: '1', 2: '2'}, scales=[0.5] * len(gts))
    scaled = evaluate([(pb * 0.5, ps, pc) for pb, ps, pc in preds], [(gb * 0.5, gc) for gb, gc in gts],
                      {0: '0', 1: '1', 2: '2'})
    assert (at_input.map_small, at_input.map_medium, at_input.map_large) == (scaled.map_small, scaled.map_medium,
                                                                            scaled.map_large)
    assert at_input.size_objects == scaled.size_objects and at_input.map50_95 == scaled.map50_95


def test_no_objects_of_a_size():
    gb = np.array([[0, 0, 200, 200]], float)                         # only a large object
    m = evaluate([(gb, np.array([0.9]), np.array([0]))], [(gb, np.array([0]))], {0: 'a'})
    assert m.map_large == 1 and m.map_small == m.map_medium == -1 and m.size_objects == {'small': 0, 'medium': 0, 'large': 1}


def test_perfect_predictions():
    gb = np.array([[10, 10, 50, 60], [100, 100, 150, 130]], float)
    m = evaluate([(gb, np.array([0.9, 0.8]), np.array([0, 1]))], [(gb, np.array([0, 1]))], NAMES)
    assert m.map50_95 == 1 and m.precision == m.recall == m.f1 == 1 and m.mean_iou == 1 and m.fp_per_image == 0
    assert m.fitness == 1 and m.confusion_matrix.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 0]]


def test_best_f1_threshold_and_fixed_threshold():
    gb = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    pb = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]], float)
    ps = np.array([0.9, 0.6, 0.3])          # two hits, one false alarm at 0.3
    m = evaluate([(pb, ps, np.array([0, 0, 0]))], [(gb, np.array([0, 0]))], {0: 'a'})
    assert 0.3 <= m.threshold < 0.6 and m.recall == 1 and m.precision == 1   # drops the false alarm, keeps both hits
    fixed = evaluate([(pb, ps, np.array([0, 0, 0]))], [(gb, np.array([0, 0]))], {0: 'a'}, threshold=0.1)
    assert fixed.threshold == 0.1 and fixed.fp_per_image == 1 and abs(fixed.precision - 2 / 3) < 1e-9


def test_confusion_matrix_counts_confusions_misses_and_false_alarms():
    gb = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    pb = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)   # first: right place, wrong class; second: false alarm
    m = evaluate([(pb, np.array([0.9, 0.9]), np.array([1, 0]))], [(gb, np.array([0, 0]))], NAMES, threshold=0.5)
    #          pred a  pred b  missed
    assert m.confusion_matrix.tolist() == [[0, 1, 1],    # true a
                                           [0, 0, 0],    # true b
                                           [1, 0, 0]]    # background (false alarms)


def test_nothing_detected():
    m = evaluate([(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))], [(np.array([[0, 0, 5, 5.]]), np.array([0]))], NAMES)
    assert m.threshold == DEFAULT_THRESHOLD and m.map50_95 == 0 and m.recall == 0
    assert '-' in m.table(NAMES).splitlines()[-1]   # class b has no objects: shown as '-'
