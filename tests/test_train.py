import csv

import numpy as np
import pytest
import torch
import yaml

from rpod import RPOD
from rpod.engine.trainer import RECIPE, Schedule, _param_groups
from rpod.loss import SetCriterion

from .conftest import make_net
from .test_dataset import make_dataset


def test_schedule_warmup_flat_final_and_patience():
    s = Schedule(epochs=10, steps_per_epoch=100, r=dict(RECIPE, warmup_steps=50))
    assert s.final_start == 8 and s.end == 10
    assert s.factor(0, 0) == pytest.approx(1 / 50) and s.factor(0, 49) == 1 and s.factor(5, 0) == 1
    assert s.factor(8, 0) == pytest.approx(1) and s.factor(9, 99) == pytest.approx(RECIPE['final_lr_ratio'], abs=1e-3)
    s.final_start = 4   # patience triggered after epoch 4
    assert s.end == 6 and s.phase(4) == 'final' and s.factor(5, 99) < 0.1


def test_param_groups_learning_rates_and_no_decay_on_1d():
    net = make_net()
    names = {id(p): n for n, p in net.named_parameters()}
    groups = _param_groups(net, RECIPE)
    assert sum(len(g['params']) for g in groups) == len(names)
    for g in groups:
        for p in g['params']:
            n = names[id(p)]
            assert g['base_lr'] == (RECIPE['backbone_lr'] if n.startswith('backbone.') else RECIPE['lr'])
            assert (g['weight_decay'] == 0) == (p.ndim == 1 or 'embed' in n)


def test_accumulation_equals_one_big_batch():
    """Two micro-batches of 1 accumulated, normalized by the objects of both, give the gradient of one batch of 2."""
    net = make_net()
    net.decoder.num_denoising = 0          # denoising noise is random: switch it off to compare exactly
    net.train()
    for m in net.modules():                # batch statistics differ by construction: use fixed ones
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    crit = SetCriterion(3)
    x = torch.rand(2, 3, 128, 128)
    t = [{'labels': torch.tensor([0, 1]), 'boxes': torch.tensor([[.3, .3, .2, .2], [.6, .6, .1, .3]])},
         {'labels': torch.tensor([2]), 'boxes': torch.tensor([[.5, .5, .4, .2]])}]
    crit(net(x, t), t)['loss'].backward()
    big = [p.grad.clone() for p in net.parameters() if p.grad is not None]
    net.zero_grad()
    for i in range(2):
        crit(net(x[i:i + 1], t[i:i + 1]), t[i:i + 1], num_boxes=3)['loss'].backward()
    small = [p.grad.clone() for p in net.parameters() if p.grad is not None]
    worst = max(float((a - b).abs().max() / (a.abs().max() + 1e-8)) for a, b in zip(big, small))
    assert worst < 1e-3


def test_train_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv('RPOD_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 4, 'val': 2, 'test': 2})
    m = RPOD(backbone='hgnetv2-b0')
    res = m.train(data, imgsz=128, epochs=3, patience=1, batch=2, device='cpu', name='t')
    run = res['dir']
    rows = list(csv.DictReader(open(run / 'results.csv')))
    assert 1 <= len(rows) <= 3 and {'map50_95', 'recall', 'fitness', 'threshold', 'phase'} <= rows[0].keys()
    args = yaml.safe_load(open(run / 'args.yaml'))
    assert args['micro_batch'] == 2 and args['accumulate'] == 1 and args['classes'] == {0: 'scratch', 1: 'screw', 2: 'label'}
    for f in ('weights/best.pt', 'weights/last.pt', 'val/pr_curves.png', 'val/confusion_matrix.png', 'val/predictions.jpg',
              'test/metrics.json'):
        assert (run / f).exists(), f
    assert m.names and m.imgsz == [128, 128] and m.ckpt_path == run / 'weights' / 'best.pt'
    assert len(m.predict(np.zeros((50, 70, 3), np.uint8), device='cpu')) == 1
    assert m.val(split='test', device='cpu').threshold == m.threshold
    reloaded = RPOD(run / 'weights' / 'best.pt')
    assert reloaded.data == str(data.resolve()) and reloaded.threshold == m.threshold
