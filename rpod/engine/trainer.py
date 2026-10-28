# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Training with one fixed recipe (RT-DETR / D-FINE / DEIM practice). The user chooses data, size, epochs, patience,
batch, device and an optional augmentation function; everything else is decided here.

Schedule: warmup, then a flat learning rate, then a cosine decay over the last 20 % of the epochs. If validation does
not improve for `patience` epochs during the flat phase, the final decay starts right away, from the weights of the
best epoch so far, and training ends when it is done. The last 4 % of the epochs (at least one: the clean images of the
default augmentation, at imgsz with multi-scale) also start again from the best epoch's weights: they add a little to
the best epoch, while continuing from weights already past it can only partly recover (COCO, size m: best epoch
49.0 AP, 49.4 after one clean epoch from it; the run's own last epoch, 48.5).

With a time budget the same phases are shares of the time instead of the epochs (Budget, _plan_epoch): training
stops between epochs, and the warmup is sized to the steps the budget holds.

The best epoch is the one with the highest mAP50-95, the latest one on a tie; its EMA weights and threshold (the
best-F1 one) go to best.pt. The test split, if any, is evaluated once at the end with that threshold and
never used for a decision.
"""
import contextlib
import copy
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from ..cfg.schema import check_input_size, derive, from_dict, positions, queries_for, to_dict
from ..data import DetectionDataset, MultiScaleBatchSampler, check_dataset, collate
from ..data.augment import MOSAIC_FRACTION
from ..data.check import format_problems
from ..loss import HungarianMatcher, SetCriterion
from ..nn import Detector
from ..utils.checkpoint import load_weights, save_checkpoint
from .validator import validate

RECIPE = dict(lr=1e-4, backbone_lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999), warmup_steps=2000, warmup_max_fraction=0.05,
              final_fraction=0.2, final_lr_ratio=0.05, ema_decay=0.9999, ema_tau=2000, clip_grad=0.1,
              freeze_backbone_bn_below=8, clean_fraction=0.04, restore_best=True, seed=0)
PROGRESS_SECONDS = 120      # one progress line this often inside an epoch (none in epochs shorter than this)
CSV_FIELDS = ['epoch', 'phase', 'lr', 'loss', 'cls', 'bbox', 'giou', 'map50_95', 'map50', 'map_small', 'map_medium',
              'map_large', 'precision', 'recall', 'f1',
              'threshold', 'mean_iou', 'fp_per_image', 'fitness', 'best', 'seconds']


def log(msg=''):
    print(msg, flush=True)


class ModelEMA:
    """Exponential moving average of the weights; it is what gets validated and saved."""

    def __init__(self, model, decay, tau):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        src, own = model.state_dict(), self.ema.state_dict()
        floats = [k for k, v in own.items() if v.dtype.is_floating_point]
        torch._foreach_lerp_([own[k] for k in floats], [src[k].detach() for k in floats], 1 - d)   # one call for all
        for k, v in own.items():
            if not v.dtype.is_floating_point:
                v.copy_(src[k])


class Schedule:
    """Learning-rate factor per optimizer step: warmup, flat, final cosine decay (which patience can bring forward)."""

    def __init__(self, epochs, steps_per_epoch, r):
        self.spe, self.r = steps_per_epoch, r
        self.final_epochs = max(1, round(r['final_fraction'] * epochs))
        self.final_start = epochs - self.final_epochs
        self.warmup = self._warmup(epochs)
        self.low = r['final_lr_ratio']
        self.decay_from, self.decay_p0 = None, 0.0     # set when a running decay is re-planned (time budget)
        self.warmup_from, self.warmup_f0 = 0, 0.0         # set when a running warmup is resized (time budget)

    def _warmup(self, epochs):
        return max(1, min(self.r['warmup_steps'], round(self.r['warmup_max_fraction'] * epochs * self.spe)))

    @property
    def end(self):
        return self.final_start + self.final_epochs

    def resize_warmup(self, steps, done):
        """Time budget: the warmup now lasts `steps` (at most warmup_steps), counted from the start, after `done` steps
        of it. The factor goes on from where it is to 1 over what is left, at least `done` more steps: a budget that
        holds fewer steps than the warmup it started with must not jump to the full learning rate."""
        steps = max(1, min(self.r['warmup_steps'], steps))
        if done >= self.warmup:
            return
        f0 = done / self.warmup if self.warmup_from == 0 else self._warmup_factor(done - 1)
        self.warmup_from, self.warmup_f0, self.warmup = done, f0, max(steps, 2 * done)

    def _warmup_factor(self, g):
        if self.warmup_from == 0:
            return (g + 1) / self.warmup
        return self.warmup_f0 + (1 - self.warmup_f0) * (g + 1 - self.warmup_from) / (self.warmup - self.warmup_from)

    def start_decay(self, epoch, epochs):
        """The final decay starts at `epoch` and lasts `epochs` (patience, or the time budget reaching it)."""
        self.final_start, self.final_epochs = epoch, epochs
        self.decay_from, self.decay_p0 = None, 0.0

    def plan_decay(self, end, done):
        """Time budget, the decay running: training now ends at `end`. What is left of the decay is spread over the
        epochs from `done` to `end`, so the learning rate goes on from where it is, without a jump, and still reaches
        its floor at the end."""
        if end == self.end:
            return
        self.decay_p0, self.decay_from = self._progress(done, 0), done
        self.final_epochs = end - self.final_start

    def plan_flat(self, end, done):
        """Time budget, before the decay: training is projected to end at `end`; the decay keeps its share of it but
        never starts before the next epoch (the budget starts it, when the time comes)."""
        final = max(1, round(self.r['final_fraction'] * end))
        self.final_start = max(done + 1, end - final)
        self.final_epochs = max(1, end - self.final_start)

    def phase(self, epoch):
        return 'final' if epoch >= self.final_start else 'flat'

    def _progress(self, epoch, step):
        if self.decay_from is None:
            p = ((epoch - self.final_start) * self.spe + step) / (self.final_epochs * self.spe)
        else:
            p = self.decay_p0 + (1 - self.decay_p0) * (((epoch - self.decay_from) * self.spe + step)
                                                       / ((self.end - self.decay_from) * self.spe))
        return min(max(p, 0.0), 1.0)

    def factor(self, epoch, step):
        g = epoch * self.spe + step
        if g < self.warmup:
            return self._warmup_factor(g)
        if epoch < self.final_start:
            return 1.0
        return self.low + (1 - self.low) * 0.5 * (1 + math.cos(math.pi * self._progress(epoch, step)))


def _clean_start(end, r):
    """First epoch of the clean last stretch: the last clean_fraction of the epochs up to where training really ends (at
    least one), the same rule as rpod.data.augment, so patience moves both together."""
    return end - max(1, round(r['clean_fraction'] * end))


class Budget:
    """Training against a time budget (RPOD.train(time=...)): the phases are fractions of the time, not of the epochs.

    `seconds` counts from `started` (the call to train, so the setup is included) to the end of train. `reserve` is
    time the caller needs afterwards (seconds, or reserve(stats) -> seconds with the stats below), taken off the budget
    after every epoch. Epochs are measured by kind, since they cost differently: with mosaic (four images and four times
    the objects per sample), plain (augmented, no mosaic) and clean; a kind not measured yet costs what the nearest one
    did. An epoch costs the mean of the last WINDOW of its kind; the first epoch of the run is measured without its
    first steps, which carry the CUDA and worker start-up."""

    WINDOW = 3
    NEAREST = {'mosaic': ('mosaic', 'plain', 'clean'), 'plain': ('plain', 'clean', 'mosaic'),
               'clean': ('clean', 'plain', 'mosaic')}

    def __init__(self, seconds, started, reserve=None, clock=time.time):
        self.seconds, self.started, self.clock = float(seconds), started, clock
        self.reserve = reserve if callable(reserve) else (lambda stats, s=float(reserve or 0): s)
        self.costs = {kind: [] for kind in self.NEAREST}
        self.guesses = {}                         # kind -> seconds, from the first epoch's steps (estimate_from_steps)
        self.val_s = self.n_val = self.save_s = 0

    def record(self, seconds, kind, val_s, n_val, save_s):
        self.costs[kind].append(seconds)
        self.val_s, self.n_val, self.save_s = val_s, n_val, save_s

    def cost(self, kind):
        """Seconds an epoch of this kind costs: measured, else estimated from the first epoch's steps, else what the
        nearest kind measured (0 before the first epoch)."""
        for k in self.NEAREST[kind]:
            if k == 'mosaic' and kind in self.guesses:      # an estimate of its own kind beats a mosaic epoch
                return self.guesses[kind]
            if self.costs[k]:
                last = self.costs[k][-self.WINDOW:]
                return sum(last) / len(last)
        return 0.0

    def estimate_from_steps(self, steps, steps_per_epoch, workers, overhead_s):
        """Before any epoch without mosaic has run, what one would cost, from the steps of a mosaic epoch. A step waits
        for its data and then computes; the dataloader workers prepare samples in parallel, so an epoch takes about
        the longer of the two: steps x GPU time per step, or samples x seconds per sample / workers (their sum with no
        workers). A mosaic sample reads four images, so where the workers are the bottleneck (big images, few CPUs: on a
        T4 VM, 596 s with mosaic and 244 s without) the plain epoch is much cheaper; where the GPU is, a little.

        `steps`: [(seconds waiting for the data, GPU seconds, [(composite, seconds) per sample])] per step. The GPU time of a step without composite
        samples comes from a least-squares line against their share (their mean when the share does not vary), the
        sample time from the samples without any. The model is scaled to what the mosaic steps really took, and the
        epoch's validation and save (`overhead_s`) added. With fewer than 8 steps nothing is estimated."""
        if len(steps) < 8:
            return
        gpu = np.array([g for _, g, _ in steps], dtype=np.float64)
        share = np.array([np.mean([c for c, _ in samples]) if samples else 0.0 for _, _, samples in steps])
        every = [t for _, _, samples in steps for _, t in samples]
        alone = [t for _, _, samples in steps for c, t in samples if not c] or every
        gpu_plain = gpu.mean()
        if share.std() >= 0.05:
            slope, base = np.polyfit(share, gpu, 1)
            if slope > 0 and base > 0:
                gpu_plain = base

        def train_s(gpu_step, sample_s):
            load = sample_s * len(every)
            return gpu_step * len(steps) + load if workers == 0 else max(gpu_step * len(steps), load / workers)

        took = sum(w + g for w, g, _ in steps)
        model = train_s(gpu.mean(), float(np.mean(every)))
        scale = took / model if model > 0 and took > 0 else 1.0
        plain = scale * train_s(gpu_plain, float(np.mean(alone))) * steps_per_epoch / len(steps)
        self.guesses['plain'] = self.guesses['clean'] = plain + overhead_s

    @property
    def epoch_s(self):
        return self.cost('plain')

    @property
    def stats(self):
        return {'epoch_s': self.epoch_s, 'val_s': self.val_s, 'n_val': self.n_val,
                'val_s_per_image': self.val_s / max(self.n_val, 1)}

    def elapsed(self):
        return self.clock() - self.started

    def remaining(self):
        """Seconds left for epochs: the budget minus what has run, the caller's reserve and rpod's own close (reload
        the best weights, adopt them)."""
        return self.seconds - self.elapsed() - self.reserve(self.stats) - 2 * self.save_s

    def fraction(self, seconds=0.0):
        """Share of the time for epochs used once `seconds` more have run."""
        total = self.elapsed() + self.remaining()
        return (self.elapsed() + seconds) / total if total > 0 else 1.0

    def overrun(self):
        """Seconds past the budget the caller left to rpod (negative: time to spare)."""
        return self.elapsed() - (self.seconds - self.reserve(self.stats))

    def state(self):
        return {'seconds': self.seconds, 'elapsed': self.elapsed(), 'costs': self.costs, 'guesses': self.guesses,
                'val_s': self.val_s, 'n_val': self.n_val, 'save_s': self.save_s}

    def load(self, state):
        """Continue a resumed run with what was left of its budget."""
        self.started = self.clock() - state['elapsed']
        self.costs = {k: list(v) for k, v in state['costs'].items()}
        self.guesses = dict(state['guesses'])
        self.val_s, self.n_val, self.save_s = state['val_s'], state['n_val'], state['save_s']


def _plan_epoch(budget, schedule, done, cap, clean_from, mosaic, early, r):
    """After `done` epochs, in time mode: what the next epoch is, from the time left and what each kind of epoch costs.
    Returns (clean_from, mosaic) and moves the end of `schedule` (equal to `done`: training stops here).

    Training only stops between epochs, and the last epoch is always clean, so the next one is the last when it and a
    clean one do not both fit. The phases go by the share of the time at the middle of the next epoch: mosaic before
    MOSAIC_FRACTION, the final decay from 1 - final_fraction, clean from 1 - clean_fraction. A phase never comes back
    once over: the mosaic stays off, the decay and the clean stretch only end sooner or later with the time left (the
    decay without a jump in the learning rate). After patience (`early`) time can only bring the end forward."""
    left, clean_s = budget.remaining(), budget.cost('clean')
    in_clean = clean_from is not None and clean_from < done
    mosaic = mosaic and not in_clean and budget.fraction(budget.cost('mosaic') / 2) < MOSAIC_FRACTION
    next_s = clean_s if in_clean else budget.cost('mosaic' if mosaic else 'plain')
    if done >= cap or left < clean_s:
        n = 0
    elif in_clean:
        n = math.floor(left / clean_s) if clean_s > 0 else cap - done
    elif left < next_s + clean_s:
        n = 1
    else:
        plain_s = budget.cost('plain')
        n = 2 + (math.floor((left - next_s - clean_s) / plain_s) if plain_s > 0 else cap)
    end = min(cap, done + n)
    if early:
        end = min(end, schedule.end)
    if end <= done:
        start = min(schedule.final_start, done)
        schedule.start_decay(start, done - start)
        return clean_from, False
    clean = (in_clean or end == done + 1 or budget.fraction(clean_s / 2) >= 1 - r['clean_fraction']
             or early and done >= _clean_start(end, r))        # after patience the end is known in epochs
    if clean and clean_from is None:
        clean_from = done
    if schedule.final_start <= done:
        schedule.plan_decay(end, done)
    elif clean or budget.fraction(next_s / 2) >= 1 - r['final_fraction']:
        schedule.start_decay(done, end - done)
    else:
        schedule.plan_flat(end, done)
    return clean_from, mosaic and not clean


def _restore_best(path, net, ema):
    """The model and its EMA back to the best epoch's weights (the EMA saved in best.pt). The EMA restarts its warmup,
    so that it follows what gets trained from there instead of staying on the restored weights; the optimizer keeps its
    moments."""
    state_dict = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    net.load_state_dict(state_dict)
    ema.ema.load_state_dict(state_dict)
    ema.updates = 0


def _param_groups(net, r):
    """Backbone and the rest at their own learning rates (equal: a 10x slower backbone, the usual DETR choice, learned
    much worse on a small real dataset); no weight decay on biases, norms and embeddings."""
    groups = {}
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        part = 'backbone' if n.startswith('backbone.') else 'rest'
        decay = p.ndim > 1 and 'embed' not in n
        groups.setdefault((part, decay), []).append(p)
    return [{'params': ps, 'lr': r['backbone_lr'] if part == 'backbone' else r['lr'], 'weight_decay': r['weight_decay'] if decay else 0.0,
             'base_lr': r['backbone_lr'] if part == 'backbone' else r['lr']} for (part, decay), ps in groups.items()]


def _set_train_mode(net, freeze_backbone_bn):
    net.train()
    if freeze_backbone_bn:
        for m in net.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def _freeze_backbone_bn(net):
    for m in net.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                p.requires_grad_(False)


def _amp(device):
    if device.type != 'cuda':
        return None
    # without including_emulation=False a T4 (sm75) answers True and trains in
    # emulated bfloat16, several times slower than its float16 tensor cores
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def _autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype else contextlib.nullcontext()


def _to_float(out):
    """Losses are computed in float32 whatever the precision of the forward pass."""
    if torch.is_tensor(out):
        return out.float() if out.is_floating_point() else out
    if isinstance(out, dict):
        return {k: _to_float(v) for k, v in out.items()}
    if isinstance(out, list):
        return [_to_float(v) for v in out]
    return out


def _split(batch, size):
    """A collated batch (images, targets, metas) in pieces of at most `size` images."""
    images, targets, metas = batch
    return [(images[i:i + size], targets[i:i + size], metas[i:i + size]) for i in range(0, len(images), size)]


def _raise_open_file_limit(target=65536):
    """DataLoader workers hand every tensor to the main process as an open file: with many workers and many objects
    per batch, a limit of 1024 open files (common in containers) runs out ('received 0 items of ancdata'). Raise the
    soft limit towards the hard one; nothing happens where that is not possible."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ImportError, ValueError, OSError):
        pass


def _micro_batch(net, criterion, batch, imgsz, device, amp_dtype, max_objects, headroom=0.8):
    """Largest divisor of `batch` whose forward + backward peaks under `headroom` of the GPU memory: the margin is
    for the optimizer, the EMA and the fragmentation that changing image sizes cause. The weights are left untouched."""
    if device.type != 'cuda':
        return batch
    backup = {k: v.clone() for k, v in net.state_dict().items()}
    n = max(1, max_objects)
    limit = headroom * torch.cuda.get_device_properties(device).total_memory
    for micro in [d for d in range(batch, 0, -1) if batch % d == 0]:
        images = targets = out = None
        torch.cuda.reset_peak_memory_stats(device)
        try:
            images = torch.rand(micro, 3, *imgsz, device=device)
            targets = [{'labels': torch.zeros(n, dtype=torch.int64, device=device),
                        'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]], device=device).repeat(n, 1)} for _ in range(micro)]
            with _autocast(device, amp_dtype):
                out = net(images, targets)
            criterion(_to_float(out), targets)['loss'].backward()
            torch.cuda.synchronize(device)
            ok = torch.cuda.max_memory_allocated(device) <= limit or micro == 1
        except torch.cuda.OutOfMemoryError:
            ok = False
        net.zero_grad(set_to_none=True)
        images = targets = out = None
        torch.cuda.empty_cache()
        if ok:
            net.load_state_dict(backup)
            return micro
    raise RuntimeError(f'the model does not fit in GPU memory even with 1 image per step at {imgsz[0]}x{imgsz[1]}')


def _progress(epoch, epochs, step, steps, loss, images, seconds, width=20):
    done = step / steps
    bar = '█' * round(width * done) + '░' * (width - round(width * done))
    left = seconds / step * (steps - step) / 60
    return (f'  epoch {epoch + 1}/{epochs} [{bar}] {100 * done:3.0f}% · {step}/{steps} steps · loss {loss:.3f} · '
            f'{images / seconds:.0f} img/s · ~{left:.0f} min left')


def _run_dir(project, name):
    base = Path(project) / name
    path, k = base, 2
    while path.exists():
        path, k = Path(f'{base}{k}'), k + 1
    path.mkdir(parents=True)
    return path


def train(model, data, imgsz, epochs, patience, batch, device, augment, name, project='runs/train', workers=None,
          save_state=False, resume=None, multiscale=False, budget=None, final_eval=True):
    """Trains `model` (an RPOD) in place and returns {'dir', 'best_epoch', 'val', 'test', 'planned_epochs',
    'stop_reason', 'budget_overrun_s'}.

    patience: epochs without improvement before the final decay starts early; None never; a function
    patience(planned_epochs) -> int is evaluated again whenever the end of training moves.

    budget: a Budget, to train for a time instead of a number of epochs (`epochs` is then the most it may train). The
    phases are shares of the time: the mosaic, the decay and the clean stretch start at the epoch whose middle falls past
    their share, and after every epoch _plan_epoch decides what the next one is from what each kind of epoch cost so
    far. It stops between epochs, never inside one. Not with multiscale.

    final_eval: evaluate the best weights on val and test (with plots) at the end. False: only reload and adopt them
    ('val' and 'test' are None), for a caller that evaluates on its own.

    multiscale: every batch at a random size, height and width drawn separately (letterboxed from the original
    images): from 0.5x to 1.5x imgsz in the first quarter of the epochs, from 0.75x to 1.25x until the last sixth, then
    at imgsz (rpod.data.dataset.MULTISCALE); validation is always at imgsz. The batch is split into accumulation steps
    by its size: whole when it fits, in pieces at the largest sizes.

    save_state: also write weights/state.pt after every epoch (optimizer, EMA, schedule, best so far), so a long run
    that gets interrupted can continue with resume=<that file> and the same arguments. Not part of RPOD.train.
    """
    r = RECIPE
    if budget is not None and multiscale:
        raise ValueError('a time budget does not support multiscale yet: its phases are fixed when training starts')
    state = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if state and from_dict(state['args']['model']) != model.cfg:
        raise ValueError(f'{resume} is a run of another model configuration')
    torch.manual_seed(r['seed'])
    np.random.seed(r['seed'])
    random.seed(r['seed'])
    device = torch.device(device)
    workers = int(os.environ.get('RPOD_WORKERS', min(8, os.cpu_count() or 1))) if workers is None else workers
    if workers > 0:
        _raise_open_file_limit()

    # ---- data
    ds = check_dataset(data)
    names, stats = ds['names'], ds['stats']
    if ds['warnings']:
        log(format_problems(ds['warnings'], f"{len(ds['warnings'])} warning(s) about the dataset"))
    log(f"data: {len(names)} classes · train {stats['train']['images']} images ({stats['train']['objects']} objects, "
        f"{stats['train']['backgrounds']} without objects) · val {stats['val']['images']} images"
        + (f" · test {stats['test']['images']} images" if 'test' in stats else ''))

    # ---- model
    num_queries = queries_for(stats['max_objects_per_image'])
    d = derive(model.cfg, len(names), num_queries)
    check_input_size(d['levels'], imgsz, num_queries)
    net = Detector(d, pretrained_backbone=state is None)
    if state:
        net.load_state_dict(state['net'])
        log(f"resuming {state['args']['name']} at epoch {state['epoch'] + 1}")
    elif model.net is not None:
        report = load_weights(net, model.net.state_dict())
        log(f"starting from the current weights: {report['summary']}")
    net.to(device)
    amp_dtype = _amp(device)
    matcher = HungarianMatcher()
    criterion = SetCriterion(len(names), matcher)

    train_set = DetectionDataset(ds['records']['train'], imgsz, len(names), augment, epochs=epochs, seed=r['seed'])
    val_set = DetectionDataset(ds['records']['val'], imgsz, len(names))
    sampler = None
    if multiscale:            # whole batches; the smallest side still leaves the decoder enough candidate positions
        min_side = next(s for s in range(32, 100000, 32) if positions(d['levels'], [s, s]) >= num_queries)
        sampler = MultiScaleBatchSampler(len(train_set), batch, imgsz, epochs, seed=r['seed'],
                                         drop_last=len(train_set) >= batch, min_side=min_side)
    largest = list(sampler.largest()) if sampler else list(imgsz)
    micro = state['micro'] if state else _micro_batch(net, criterion, batch, largest, device, amp_dtype,
                                                      stats['max_objects_per_image'])   # at the largest size
    accumulate = math.ceil(batch / micro)
    freeze_bn = micro < r['freeze_backbone_bn_below']
    if freeze_bn:
        _freeze_backbone_bn(net)
    log(f'batch {batch} = {micro} images x {accumulate} accumulation step(s)' + (' at the largest size' if sampler else '')
        + (f' · backbone BatchNorm frozen ({micro} images per step < {r["freeze_backbone_bn_below"]})' if freeze_bn else '')
        + f" · precision {str(amp_dtype).replace('torch.', '') if amp_dtype else 'float32'} · device {device}")
    if sampler:
        start = 0
        for end, hs, ws in sampler.phases:
            if end > start:
                log(f'multi-scale: epochs {start + 1}-{end} at heights {hs[0]}-{hs[-1]} x widths {ws[0]}-{ws[-1]} '
                    f'(drawn separately for every batch)')
            start = max(start, end)
        log(f'multi-scale: epochs {start + 1}-{epochs} at {imgsz[0]}x{imgsz[1]}')
    if sampler:
        loader = DataLoader(train_set, batch_sampler=sampler, num_workers=workers, collate_fn=collate,
                            pin_memory=device.type == 'cuda', persistent_workers=workers > 0)
    else:
        loader = DataLoader(train_set, batch_size=micro, shuffle=True, num_workers=workers, collate_fn=collate,
                            drop_last=len(train_set) >= micro, pin_memory=device.type == 'cuda',
                            persistent_workers=workers > 0,
                            generator=torch.Generator().manual_seed(r['seed'] + (state['epoch'] if state else 0)))
    steps_per_epoch = max(1, len(loader) if sampler else math.ceil(len(loader) / accumulate))
    largest_px = largest[0] * largest[1]
    optimizer = torch.optim.AdamW(_param_groups(net, r), betas=r['betas'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
    ema = ModelEMA(net, r['ema_decay'], r['ema_tau'])
    schedule = Schedule(epochs, steps_per_epoch, r)
    params = [p for p in net.parameters() if p.requires_grad]

    best_fitness, best_epoch, stale, epoch = -1.0, -1, 0, 0
    if state:
        ema.ema.load_state_dict(state['ema'])
        ema.updates = state['ema_updates']
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        schedule.final_start = state['final_start']
        best_fitness, best_epoch, stale, epoch = state['best_fitness'], state['best_epoch'], state['stale'], state['epoch']
    clean_from, early = None, False           # clean_from: first clean epoch, fixed once it has started
    mosaic = True                             # time mode: the mosaic is on until the budget turns it off for good
    if budget is not None and epochs == 1:
        clean_from = 0                        # the only epoch is the last one, clean as in epoch mode
    if state and budget is not None:
        schedule.final_epochs, schedule.warmup = state['final_epochs'], state['warmup']
        schedule.decay_from, schedule.decay_p0 = state['decay_from'], state['decay_p0']
        schedule.warmup_from, schedule.warmup_f0 = state['warmup_from'], state['warmup_f0']
        clean_from, mosaic, early = state['clean_from'], state['mosaic'], state['early']
        budget.load(state['budget'])
    train_set.set_epochs(schedule.end)        # the augmentation's phases follow the real end of training
    first_epoch = epoch                       # its first steps carry the start-up: left out of its cost

    # ---- run folder
    run = Path(state['run']) if state else _run_dir(project, name)
    if not state:
        (run / 'weights').mkdir()
    args = {'data': str(Path(data).resolve()), 'imgsz': list(imgsz), 'epochs': epochs,
            'patience': getattr(patience, '__name__', 'function') if callable(patience) else patience,
            'time': budget.seconds if budget is not None else None, 'batch': batch,
            'device': str(device), 'augment': getattr(augment, '__name__', None) if augment else None, 'name': run.name,
            'model': to_dict(model.cfg), 'classes': names, 'num_queries': num_queries, 'micro_batch': micro,
            'accumulate': accumulate, 'backbone_bn_frozen': freeze_bn, 'multiscale': bool(multiscale), 'recipe': {k: list(v) if isinstance(v, tuple) else v
                                                                                   for k, v in r.items()}}
    if state:
        args['name'] = state['args']['name']
    else:
        (run / 'args.yaml').write_text(yaml.safe_dump(args, sort_keys=False, allow_unicode=True))
    csv_file = open(run / 'results.csv', 'a' if state else 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not state:
        writer.writeheader()
    csv_file.flush()
    extra = {'data': args['data']}
    log(f'results: {run}')

    try:
        while epoch < schedule.end:
            t0 = time.time()
            if budget is None and clean_from is None and epoch >= _clean_start(schedule.end, r):
                clean_from = epoch                   # time mode: _plan_epoch decides it
            clean = clean_from is not None and epoch >= clean_from
            if r['restore_best'] and epoch == clean_from and 0 <= best_epoch < epoch - 1:
                _restore_best(run / 'weights' / 'best.pt', net, ema)
                log(f'\nepochs {clean_from + 1}-{schedule.end}: back to the weights of the best epoch ({best_epoch + 1})')
            if budget is not None:
                train_set.set_phase(mosaic=mosaic and not clean, clean=clean)
            train_set.set_epoch(epoch)
            if sampler:
                sampler.epoch, sampler.fixed = epoch, clean_from is not None and epoch >= clean_from
            _set_train_mode(net, freeze_bn)
            sums, done, seen, next_report = {}, 0, 0, t0 + PROGRESS_SECONDS
            # the first epoch of the run is timed from step `warm`, and in time mode its steps `warm`-2*`warm` size the
            # warmup to the steps the budget holds, not to `epochs` (only a cap there)
            warm = min(10, steps_per_epoch // 4) if epoch == first_epoch else 0
            t_warm = None
            # time mode, a mosaic epoch before any other: every step timed (synchronised) with its share of composite
            # samples, to estimate an epoch without mosaic before one has run (Budget.estimate_from_steps)
            probing = budget is not None and epoch == first_epoch and mosaic and not clean and not budget.costs['plain']
            probe = [] if probing else None
            it = iter(loader)
            for step in range(steps_per_epoch):
                t_step = time.time()
                factor = schedule.factor(epoch, step)
                for g in optimizer.param_groups:
                    g['lr'] = g['base_lr'] * factor
                group = []
                if sampler:          # a whole batch at one size: in pieces of what fits at that size
                    whole = next(it, None)
                    if whole is not None:
                        h, w = whole[0].shape[-2:]
                        group = _split(whole, max(1, micro * largest_px // (h * w)))
                else:
                    for _ in range(accumulate):
                        try:
                            group.append(next(it))
                        except StopIteration:
                            break
                if not group:
                    break
                t_data = time.time()
                total_boxes = sum(len(t['labels']) for _, targets, _ in group for t in targets)
                total_images = sum(len(images) for images, _, _ in group)
                for images, targets, _ in group:
                    images = images.to(device, non_blocking=True)
                    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                    with _autocast(device, amp_dtype):
                        out = net(images, targets)
                    losses = criterion(_to_float(out), targets, num_boxes=total_boxes, num_images=total_images)
                    scaler.scale(losses['loss']).backward()
                    for k in ('loss', 'cls', 'bbox', 'giou'):   # on the device: read only for the log
                        sums[k] = sums.get(k, 0.0) + losses[k].detach()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, r['clip_grad'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(net)
                done += 1
                seen += total_images
                if probe is not None and done > warm:
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    probe.append((t_data - t_step, time.time() - t_data,
                                  [(m['drawn'] > 0, m['seconds']) for _, _, metas in group for m in metas]))
                if warm and done in (warm, 2 * warm):
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    if done == warm:
                        t_warm = time.time()
                    elif budget is not None and not state:
                        step_s = (time.time() - t_warm) / warm
                        if step_s > 0:
                            steps = (budget.elapsed() + budget.remaining()) / step_s
                            schedule.resize_warmup(round(r['warmup_max_fraction'] * steps), done)
                if time.time() >= next_report:
                    next_report = time.time() + PROGRESS_SECONDS
                    log(_progress(epoch, schedule.end, step + 1, steps_per_epoch, float(sums['loss']) / done,
                                  seen, time.time() - t0))
            means = {k: float(v) / max(done, 1) for k, v in sums.items()}   # per optimizer step (effective batch)

            t_val = time.time()
            metrics = validate(ema.ema, val_set, names, device, micro, workers=workers, amp_dtype=amp_dtype)
            val_s = time.time() - t_val
            phase = schedule.phase(epoch)
            # a tie takes over as the best but is no improvement for patience. A small validation set saturates
            # (fitness 1.0 from epoch 8 on, 3 val images): keeping the first epoch to get there shipped an undertrained
            # model with duplicate boxes, while resetting patience on ties trained to the end of the budget
            improved = metrics.fitness >= best_fitness
            if improved:
                if metrics.fitness > best_fitness:
                    stale = 0
                elif phase == 'flat':
                    stale += 1
                best_fitness, best_epoch = metrics.fitness, epoch
                save_checkpoint(run / 'weights' / 'best.pt', ema.ema, model.cfg, names, imgsz, num_queries,
                                metrics.threshold, {**metrics.to_dict(), 'epoch': epoch + 1}, extra)
            elif phase == 'flat':
                stale += 1
            t_save = time.time()
            save_checkpoint(run / 'weights' / 'last.pt', ema.ema, model.cfg, names, imgsz, num_queries,
                            metrics.threshold, {**metrics.to_dict(), 'epoch': epoch + 1}, extra)
            save_s = time.time() - t_save
            seconds = time.time() - t0
            wait = patience(schedule.end) if callable(patience) else patience
            stopped = phase == 'flat' and wait is not None and stale >= wait and epoch + 1 < schedule.final_start
            back = False
            if stopped:
                schedule.start_decay(epoch + 1, schedule.final_epochs)
                early = True
                back = r['restore_best'] and best_epoch < epoch
                if back:
                    _restore_best(run / 'weights' / 'best.pt', net, ema)
            if budget is not None:     # plan before the log, so that 'epoch N/E' shows the current projection
                cost = seconds
                if t_warm is not None and done > warm:       # the steps before `warm`, at the pace of the rest
                    cost = (t_val - t_warm) * done / (done - warm) + time.time() - t_val
                budget.record(cost, 'clean' if clean else 'mosaic' if mosaic else 'plain', val_s, len(val_set), save_s)
                if probe is not None:
                    budget.estimate_from_steps(probe, steps_per_epoch, workers, time.time() - t_val)
                before, had_mosaic, had_clean = schedule.end, mosaic, clean_from
                clean_from, mosaic = _plan_epoch(budget, schedule, epoch + 1, epochs, clean_from, mosaic, early, r)
            train_set.set_epochs(schedule.end)
            writer.writerow({'epoch': epoch + 1, 'phase': phase, 'lr': float(f"{factor * r['lr']:.4g}"),
                             **{k: round(v, 5) for k, v in means.items()},
                             **{k: round(v, 5) for k, v in metrics.to_dict().items()}, 'best': int(improved),
                             'seconds': round(seconds, 1)})
            csv_file.flush()
            log(f"\nepoch {epoch + 1}/{schedule.end} [{phase}]  lr {factor * r['lr']:.2e}  loss {means.get('loss', 0):.3f} "
                f"(cls {means.get('cls', 0):.3f} l1 {means.get('bbox', 0):.3f} giou {means.get('giou', 0):.3f})  {seconds:.0f}s")
            log(f"val  mAP50-95 {metrics.map50_95:.3f}  mAP50 {metrics.map50:.3f}  P {metrics.precision:.3f}  "
                f"R {metrics.recall:.3f}  F1 {metrics.f1:.3f} @ {metrics.threshold:.3f}  IoU {metrics.mean_iou:.3f}  "
                f"FP/img {metrics.fp_per_image:.2f}  fitness {metrics.fitness:.4f}" + ('  * best' if improved else '')
                + '\n     mAP50-95 by size  ' + '  '.join(f'{k} ' + (f'{v:.3f}' if v >= 0 else '-') for k, v in
                                                     (('small', metrics.map_small), ('medium', metrics.map_medium),
                                                      ('large', metrics.map_large))))
            log(metrics.table(names))
            if stopped:
                log(f'no improvement for {wait} epochs: '
                    + (f'back to the weights of the best epoch ({best_epoch + 1}), ' if back else '')
                    + f'starting the final learning-rate decay ({schedule.final_epochs} epochs, ends at epoch {schedule.end})')
            if budget is not None:
                log(f'time: {budget.remaining() / 60:.1f} min left for epochs ({100 * budget.fraction():.0f} % used), '
                    f'~{budget.epoch_s:.0f}s each · training ends at epoch {schedule.end}'
                    + (f' (was {before})' if schedule.end != before else ''))
                if had_mosaic and not mosaic and epoch + 1 < schedule.end:
                    log(f'time: no more mosaic from epoch {epoch + 2}')
                if had_clean is None and clean_from is not None and epoch + 1 < schedule.end:
                    log(f'time: clean images from epoch {clean_from + 1}')
                if epoch + 1 == 1 and schedule.end == 1:
                    log('warning: the first epoch used up the time budget; training ends without a clean epoch')
            if save_state:
                tmp = run / 'weights' / 'state.pt.tmp'
                torch.save({'run': str(run.resolve()), 'epoch': epoch + 1, 'net': net.state_dict(), 'ema': ema.ema.state_dict(),
                            'ema_updates': ema.updates, 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                            'final_start': schedule.final_start, 'best_fitness': best_fitness, 'best_epoch': best_epoch,
                            'stale': stale, 'micro': micro, 'args': args, 'final_epochs': schedule.final_epochs,
                            'warmup': schedule.warmup, 'decay_from': schedule.decay_from, 'decay_p0': schedule.decay_p0,
                            'warmup_from': schedule.warmup_from, 'warmup_f0': schedule.warmup_f0,
                            'clean_from': clean_from, 'mosaic': mosaic, 'early': early,
                            'budget': budget.state() if budget is not None else None}, tmp)
                os.replace(tmp, run / 'weights' / 'state.pt')
            epoch += 1
    finally:
        csv_file.close()

    # ---- final: best weights, plots, test
    best = torch.load(run / 'weights' / 'best.pt', map_location='cpu', weights_only=True)
    net = Detector(derive(model.cfg, len(names), num_queries), pretrained_backbone=False)
    net.load_state_dict(best['state_dict'])
    net.to(device).eval()
    threshold = best['threshold']
    log(f'\nbest epoch {best_epoch + 1} (fitness {best_fitness:.4f}), recommended threshold {threshold:.3f}')
    model._adopt(net.eval(), names, imgsz, threshold, num_queries, run / 'weights' / 'best.pt', args['data'])
    result = {'dir': run, 'best_epoch': best_epoch + 1, 'val': None, 'test': None, 'planned_epochs': schedule.end,
              'stop_reason': ('EARLY_STOPPED' if early else 'TIME_BUDGET_REACHED' if budget is not None and schedule.end < epochs
                              else 'EPOCHS_COMPLETED'),
              'budget_overrun_s': None}
    if not final_eval:
        if budget is not None:
            result['budget_overrun_s'] = budget.overrun()
        return result
    val_metrics = validate(net, val_set, names, device, micro, threshold=threshold, plots_dir=run / 'val', workers=workers,
                           amp_dtype=amp_dtype)
    test_metrics = None
    if ds['records'].get('test'):
        test_set = DetectionDataset(ds['records']['test'], imgsz, len(names))
        test_metrics = validate(net, test_set, names, device, micro, threshold=threshold, plots_dir=run / 'test',
                                workers=workers, amp_dtype=amp_dtype)
        log(f'\ntest (never used for any decision), threshold {threshold:.3f}')
        log(test_metrics.table(names))
    result.update(val=val_metrics, test=test_metrics)
    if budget is not None:
        result['budget_overrun_s'] = budget.overrun()
    return result
