# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Training against a time budget: costs by kind of epoch, the phases as shares of the time, the rules that keep
finished phases finished, and a real run on a fake clock."""
import re
from types import SimpleNamespace

import numpy as np
import pytest

import rpod.data.augment as A
from rpod import RPOD
from rpod.engine.trainer import RECIPE, Budget, Schedule, _plan_epoch, train

from .test_dataset import make_dataset


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def budget(seconds, mosaic=(), plain=(), clean=(), reserve=None):
    """A Budget on a fake clock that has run these epochs back to back, in this order."""
    clock = Clock()
    b = Budget(seconds, 0.0, reserve, clock=clock)
    for kind, times in (('mosaic', mosaic), ('plain', plain), ('clean', clean)):
        for s in times:
            clock.now += s
            b.record(s, kind, 0.0, 1, 0.0)
    return b


def schedule(epochs=100, spe=10):
    return Schedule(epochs, spe, dict(RECIPE, warmup_steps=2000))


def plan(b, s, done, clean_from=None, mosaic=True, early=False, cap=100):
    return _plan_epoch(b, s, done, cap, clean_from, mosaic, early, RECIPE)


def test_each_kind_of_epoch_costs_its_own_and_borrows_from_the_nearest():
    b = budget(1000, mosaic=[9, 7, 7, 7, 7])
    assert b.cost('mosaic') == 7                                 # the last three
    assert b.cost('plain') == b.cost('clean') == 7               # nothing else measured yet: the mosaic ones
    b = budget(1000, mosaic=[9, 7], plain=[5, 6])
    assert b.cost('plain') == 5.5 and b.cost('clean') == 5.5 and b.cost('mosaic') == 8
    assert budget(1000, plain=[5], clean=[4]).cost('clean') == 4
    assert budget(10).cost('plain') == 0


def test_an_epoch_without_mosaic_is_estimated_from_the_steps_of_a_mosaic_one():
    """Where the GPU is the bottleneck, steps with more composite samples compute longer: the line through them at
    zero gives the step of a plain epoch. Measured beats estimated."""
    rng = np.random.default_rng(0)
    share = rng.binomial(4, 0.5, 100) / 4
    steps = [(0.0, 1.0 + 3.0 * x + rng.normal(0, 0.05), [(i < 4 * x, 0.01) for i in range(4)]) for x in share]
    b = budget(1000, mosaic=[260])
    b.estimate_from_steps(steps, 100, workers=8, overhead_s=10)
    assert b.cost('plain') == pytest.approx(110, rel=0.05) and b.cost('clean') == b.cost('plain')
    assert b.cost('mosaic') == 260
    b.record(150, 'plain', 0.0, 1, 0.0)
    assert b.cost('plain') == 150 and b.cost('clean') == 150
    b = budget(1000, mosaic=[260])
    b.estimate_from_steps(steps[:5], 100, workers=8, overhead_s=10)        # too few steps: nothing to go on
    assert b.cost('plain') == 260


def test_where_the_workers_are_the_bottleneck_a_plain_epoch_costs_its_samples():
    """test_rp_od_4 on a T4 VM: 2048 px images, 8 CPUs, the mosaic epoch 2.4x the plain one. A composite sample (four
    reads) takes 4 s in its worker, a plain one 1 s, the GPU 0.2 s a step: the steps wait for the data, and without
    mosaic the samples (and so the epoch) take 2.5x less."""
    rng = np.random.default_rng(0)
    steps = []
    for _ in range(100):
        samples = [(bool(c), 4.0 if c else 1.0) for c in rng.integers(0, 2, 4)]
        steps.append((sum(t for _, t in samples) / 8 - 0.2, 0.2, samples))
    b = budget(1000, mosaic=[200])
    b.estimate_from_steps(steps, 100, workers=8, overhead_s=10)
    assert b.cost('plain') == pytest.approx(4 * 100 * 1.0 / 8 + 10, rel=0.1)


def test_test_rp_od_6_trains_on_after_a_first_epoch_of_half_the_time():
    """900 s for epochs, the first (mosaic) took 452 s: 448 s left. Costed as a mosaic epoch, the clean one did not fit
    and the run ended after one epoch; with the estimate (190 s) two more fit, without mosaic, the last one clean."""
    b = budget(900, mosaic=[452])
    s = schedule()
    assert plan(b, s, 1) == (None, False) and s.end == 1
    b.guesses.update(plain=190, clean=190)
    s = schedule()
    clean_from, mosaic = plan(b, s, 1)
    assert s.end == 3 and not mosaic and clean_from is None and s.phase(1) == 'flat'
    b.clock.now += 190
    b.record(190, 'plain', 0.0, 1, 0.0)
    clean_from, mosaic = plan(b, s, 2, clean_from, mosaic)       # 258 s left: one clean epoch, the last
    assert s.end == 3 and clean_from == 2 and s.phase(2) == 'final'


def test_the_mosaic_ends_at_a_quarter_of_the_time_and_does_not_come_back():
    b = budget(1000, mosaic=[7] * 35)                            # 245 s: the next one's middle at 24.85 %
    assert plan(b, schedule(), 35)[1] is True
    b = budget(1000, mosaic=[7] * 35, plain=[2])                 # 247 s: 25.05 %
    assert plan(b, schedule(), 36)[1] is False
    assert plan(budget(1000, mosaic=[7]), schedule(), 1, mosaic=False)[1] is False


def test_the_end_follows_the_measured_costs_not_the_mosaic_ones():
    """test_rp_od_3: mosaic epochs 6.7 s, the rest 5.5 s. Averaging them together moved the projected end from 149 to
    167 when the mosaic ended; by kind it is right from the first plain epoch on."""
    s = schedule()
    plan(budget(1000, mosaic=[7] * 36, plain=[5.5]), s, 37, cap=500)
    assert s.end == 37 + 2 + int((1000 - 7 * 36 - 5.5 - 11) // 5.5)


def test_decay_and_clean_stretch_start_at_their_share_of_the_time():
    s = schedule()
    clean_from, _ = plan(budget(100, [10], [1] * 69), s, 70, mosaic=False)
    assert s.final_start > 70 and clean_from is None             # the next one's middle at 79.5 %: not yet
    s = schedule()
    clean_from, _ = plan(budget(100, [10], [1] * 70), s, 71, mosaic=False)
    assert s.final_start == 71 and s.end == 91 and clean_from is None    # 80.5 %: the decay starts
    s = schedule()
    clean_from, _ = plan(budget(100, [10], [1] * 86), s, 87, mosaic=False)
    assert clean_from == 87                                      # 96.5 %: clean from here to the end


def test_the_next_epoch_is_clean_and_last_when_two_do_not_fit():
    s = schedule()
    assert plan(budget(65, plain=[10] * 5), s, 5) == (5, False)  # 15 s left: one epoch, not two
    assert s.end == 6 and s.final_start == 5
    s = schedule()
    plan(budget(55, plain=[10] * 5), s, 5)                       # 5 s left: not even the clean one
    assert s.end == 5


def test_a_running_decay_follows_the_time_left_without_a_jump_in_the_learning_rate():
    s = schedule()
    s.start_decay(80, 10)
    before = s.factor(84, 9)
    plan(budget(1000, plain=[1] * 85), s, 85, mosaic=False)      # far more time than planned: it ends later
    assert s.end > 90 and s.factor(85, 0) == pytest.approx(before, abs=0.02) and s.factor(85, 0) <= before
    assert s.factor(s.end - 1, 9) == pytest.approx(RECIPE['final_lr_ratio'], abs=1e-3)
    end = s.end
    plan(budget(87, plain=[1] * 85), s, 85, mosaic=False)        # and earlier with less
    assert s.end == 87 < end


def test_after_patience_time_only_brings_the_end_forward():
    s = schedule()
    s.start_decay(30, 6)
    plan(budget(1000, plain=[1] * 31), s, 31, mosaic=False, early=True)
    assert s.end == 36
    plan(budget(34, plain=[1] * 31), s, 31, mosaic=False, early=True)
    assert s.end == 34


def test_the_warmup_shrinks_to_the_budget_from_where_it_is():
    """test_rp_od_6: a warmup of 2000 steps in a run of 126 trained at 6 % of the learning rate at most. Resized after
    20 steps to what the budget holds, it goes on from its current factor to 1, never in one jump."""
    s = schedule(500, 126)                                       # 5 % of 63000 steps: capped at 2000
    assert s.warmup == 2000
    s.resize_warmup(12, 20)
    assert s.warmup == 40
    assert s.factor(0, 19) == pytest.approx(20 / 2000)
    assert s.factor(0, 20) - s.factor(0, 19) < 0.06 and s.factor(0, 39) == 1 and s.factor(0, 40) == 1
    s = schedule(500, 126)
    s.resize_warmup(300, 20)
    assert s.warmup == 300 and s.factor(1, 9) < 1


def test_a_time_budget_sets_the_phase_of_the_augmentation(monkeypatch):
    calls = []
    monkeypatch.setattr(A, 'P_MOSAIC', 1.0)
    monkeypatch.setattr(A, 'mosaic', lambda samples, imgsz, rng: calls.append(1) or samples[0])
    sample = A.Sample(np.zeros((40, 40, 3), np.uint8), np.zeros((0, 4), np.float32), np.zeros(0, np.int64), 'x')
    ctx = SimpleNamespace(epoch=60, epochs=100, imgsz=[32, 32], rng=np.random.default_rng(0),
                          sample=lambda k: [sample] * k, mosaic=None, clean=None)
    A.default_augment(sample, ctx)
    assert calls == []                                           # epoch 60 of 100: past the mosaic share
    ctx.mosaic = True
    A.default_augment(sample, ctx)
    assert calls == [1]                                          # the budget says mosaic
    ctx.epoch, ctx.mosaic = 2, False
    A.default_augment(sample, ctx)
    assert calls == [1]
    ctx.clean = True
    assert A.default_augment(sample, ctx) is sample


class SteppedBudget(Budget):
    """Every epoch takes `step` seconds of a fake clock, whatever it really took."""

    def __init__(self, seconds, step, reserve=None):
        self.fake = Clock()
        super().__init__(seconds, 0.0, reserve, clock=self.fake)
        self.step = step

    def record(self, seconds, kind, val_s, n_val, save_s):
        self.fake.now += self.step
        super().record(self.step, kind, val_s, n_val, save_s)


def test_training_to_a_budget_end_to_end(tmp_path, monkeypatch, capsys):
    """45 s at 10 s per epoch: 4 epochs, mosaic only in the first (its middle is under a quarter of the time), the last
    one clean from the best weights. The log lines the platform parses keep their format."""
    monkeypatch.setenv('RPOD_WORKERS', '0')
    data = make_dataset(tmp_path / 'data', {'train': 4, 'val': 2})
    seen = {}

    def record(sample, ctx):
        seen.setdefault(ctx.epoch, set()).add((ctx.mosaic, ctx.clean))
        return sample

    m = RPOD('s')
    b = SteppedBudget(45, 10)
    res = train(m, str(data), [128, 128], 100, None, 2, 'cpu', record, 't', project=tmp_path / 'runs', budget=b,
                final_eval=False)
    assert res['planned_epochs'] == 4 and res['stop_reason'] == 'TIME_BUDGET_REACHED'
    assert res['budget_overrun_s'] == pytest.approx(-5)
    assert res['val'] is None and res['test'] is None
    assert m.threshold is not None and m.ckpt_path == res['dir'] / 'weights' / 'best.pt'
    assert seen[0] == {(True, False)} and seen[1] == seen[2] == {(False, False)} and seen[3] == {(False, True)}
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
