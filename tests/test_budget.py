# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Training against a time budget: the projection, the rules that keep started phases in place, and a real run on a
fake clock."""
import re
from types import SimpleNamespace

import numpy as np
import pytest

import rpod.data.augment as A
from rpod import RPOD
from rpod.engine.trainer import RECIPE, Budget, Schedule, _clean_start, _fit_to_budget, train

from .test_dataset import make_dataset


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def budget(seconds, epochs=(), clean=(), reserve=None):
    """A Budget on a fake clock that has run `epochs` (and `clean` epochs) back to back."""
    clock = Clock()
    b = Budget(seconds, 0.0, reserve, clock=clock)
    for s in epochs:
        clock.now += s
        b.record(s, False, 0.0, 1, 0.0)
    for s in clean:
        clock.now += s
        b.record(s, True, 0.0, 1, 0.0)
    return b


def test_epoch_cost_leaves_out_the_first_epoch_and_averages_the_last_three():
    assert budget(100, [30]).epoch_s == 30                       # alone, the first one is all there is
    assert budget(100, [30, 4, 5, 6, 7]).epoch_s == pytest.approx(6)
    assert budget(100, [30, 4], clean=[2]).clean_s == 2
    assert budget(100, [30, 4]).clean_s == 4                     # no clean epoch yet: a normal one


def test_projection_reserve_and_cap():
    b = budget(100, [10])
    assert b.planned(1, 500) == 10                               # 90 s left, 10 s per epoch: 9 more
    assert b.planned(1, 5) == 5
    assert budget(100, [10], reserve=30).planned(1, 500) == 7
    stats = {}
    budget(100, [10], reserve=lambda s: stats.update(s) or 0).remaining()
    assert set(stats) == {'epoch_s', 'val_s', 'n_val', 'val_s_per_image'}
    assert budget(5, [10]).planned(1, 500) == 2                  # over budget: still at least one more


def schedule(epochs=100, spe=10):
    return Schedule(epochs, spe, dict(RECIPE, warmup_steps=2000))


def test_flat_phase_follows_the_projection_and_the_warmup_only_shrinks():
    s = schedule()
    warmup = s.warmup
    _fit_to_budget(budget(100, [10]), s, 1, 100, None)
    assert s.end == 10 and s.final_start == 8 and s.warmup <= warmup
    _fit_to_budget(budget(150, [10]), s, 1, 100, None)          # the projection grows: so does the end
    assert s.end == 15 and s.final_start == 12


def test_started_decay_never_gets_longer_only_shorter():
    s = schedule()
    s.final_start, s.final_epochs = 5, 3                         # decay from epoch 5, ends at 8
    _fit_to_budget(budget(1000, [10] * 6), s, 6, 100, None)
    assert (s.final_start, s.end) == (5, 8)
    _fit_to_budget(budget(85, [10] * 6), s, 6, 100, None)       # 25 s left: room for 2 epochs, the plan said 2 more
    assert (s.final_start, s.end) == (5, 8)
    s.final_start, s.final_epochs = 5, 5                         # plan: ends at 10, but only 7 fit
    _fit_to_budget(budget(75, [10] * 6), s, 6, 100, None)
    assert (s.final_start, s.end) == (5, 7)


def test_two_epochs_ahead_the_next_one_is_clean_and_last():
    s = schedule()
    _fit_to_budget(budget(65, [10] * 5), s, 5, 100, None)        # 15 s left: one epoch, not two
    assert s.end == 6 and _clean_start(s.end, RECIPE) <= 5 and s.final_start == 5
    s = schedule()
    _fit_to_budget(budget(55, [10] * 5), s, 5, 100, None)        # 5 s left: not even the clean one
    assert s.end == 5


def test_first_epoch_eating_the_budget_ends_the_run():
    s = schedule()
    _fit_to_budget(budget(8, [10]), s, 1, 100, None)
    assert s.end == 1


def test_clean_stretch_stops_as_soon_as_the_next_one_does_not_fit():
    s = schedule()
    s.final_start, s.final_epochs = 80, 20                       # clean stretch from 96
    _fit_to_budget(budget(1000, [1] * 96, clean=[1]), s, 97, 100, 96)
    assert s.end == 100                                          # fits: unchanged, never extended
    _fit_to_budget(budget(97.5, [1] * 96, clean=[1]), s, 97, 100, 96)
    assert s.end == 97


def test_frozen_mosaic_end_keeps_the_mosaic_off(monkeypatch):
    calls = []
    monkeypatch.setattr(A, 'P_MOSAIC', 1.0)
    monkeypatch.setattr(A, 'mosaic', lambda samples, imgsz, rng: calls.append(1) or samples[0])
    sample = A.Sample(np.zeros((40, 40, 3), np.uint8), np.zeros((0, 4), np.float32), np.zeros(0, np.int64), 'x')
    ctx = SimpleNamespace(epoch=2, epochs=100, imgsz=[32, 32], rng=np.random.default_rng(0), sample=lambda k: [sample] * k,
                          mosaic_end=None)
    A.default_augment(sample, ctx)
    assert calls == [1]                                          # epoch 2 of 100: mosaic time
    ctx.mosaic_end = 1                                           # a budget fixed its end at epoch 1
    A.default_augment(sample, ctx)
    assert calls == [1]


class SteppedBudget(Budget):
    """Every epoch takes `step` seconds of a fake clock, whatever it really took."""

    def __init__(self, seconds, step, reserve=None):
        self.fake = Clock()
        super().__init__(seconds, 0.0, reserve, clock=self.fake)
        self.step = step

    def record(self, seconds, clean, val_s, n_val, save_s):
        self.fake.now += self.step
        super().record(self.step, clean, val_s, n_val, save_s)


def test_training_to_a_budget_end_to_end(tmp_path, monkeypatch, capsys):
    """45 s at 10 s per epoch: 4 epochs, the last one clean from the best weights, the mosaic ended at epoch 1 and
    never back. The log lines the platform parses keep their format."""
    monkeypatch.setenv('RPOD_WORKERS', '0')
    data = make_dataset(tmp_path / 'data', {'train': 4, 'val': 2})
    seen = {}

    def record(sample, ctx):
        seen.setdefault(ctx.epoch, set()).add((ctx.epochs, ctx.mosaic_end))
        return sample

    m = RPOD('s')
    b = SteppedBudget(45, 10)
    res = train(m, str(data), [128, 128], 100, None, 2, 'cpu', record, 't', project=tmp_path / 'runs', budget=b,
                final_eval=False)
    assert res['planned_epochs'] == 4 and res['stop_reason'] == 'TIME_BUDGET_REACHED'
    assert res['budget_overrun_s'] == pytest.approx(-5)
    assert res['val'] is None and res['test'] is None
    assert m.threshold is not None and m.ckpt_path == res['dir'] / 'weights' / 'best.pt'
    assert seen[0] == {(100, None)} and all(mosaic == 1 for e in (1, 2, 3) for _, mosaic in seen[e])
    out = capsys.readouterr().out
    epochs = re.findall(r'^epoch (\d+)/(\d+) \[(flat|final)\]', out, re.M)
    assert [(int(e), int(n)) for e, n, _ in epochs] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert len(re.findall(r'^val  mAP50-95 \d\.\d{3}  mAP50 \d\.\d{3}  P \d\.\d{3}  R \d\.\d{3}', out, re.M)) == 4


def test_patience_none_or_a_function_of_the_planned_epochs(tmp_path, monkeypatch):
    monkeypatch.setenv('RPOD_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 2, 'val': 1})
    res = RPOD('s').train(data, imgsz=128, epochs=3, patience=None, batch=2, device='cpu', pretrained=False,
                          final_eval=False)
    assert res['planned_epochs'] == 3 and res['stop_reason'] == 'EPOCHS_COMPLETED'
    asked = []
    res = RPOD('s').train(data, imgsz=128, epochs=10, patience=lambda e: asked.append(e) or 0, batch=2, device='cpu',
                          pretrained=False, final_eval=False)
    assert asked[0] == 10 and res['stop_reason'] == 'EARLY_STOPPED' and res['planned_epochs'] == 3
