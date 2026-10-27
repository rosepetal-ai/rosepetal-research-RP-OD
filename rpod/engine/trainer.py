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

The best epoch is the one with the highest 0.5 * mAP50-95 + 0.5 * recall (IoU >= 0.5, at the best-F1 threshold); its
EMA weights and threshold go to best.pt. The test split, if any, is evaluated once at the end with that threshold and
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

    def _warmup(self, epochs):
        return max(1, min(self.r['warmup_steps'], round(self.r['warmup_max_fraction'] * epochs * self.spe)))

    @property
    def end(self):
        return self.final_start + self.final_epochs

    def replan(self, epochs, done):
        """Time budget, before the final decay: training now ends after `epochs`. The decay keeps its share of them but
        never starts before `done` (epochs already trained). The warmup only gets shorter: if the step already passed
        the new one, it ends there."""
        self.final_epochs = max(1, round(self.r['final_fraction'] * epochs))
        self.final_start = max(done, epochs - self.final_epochs)
        self.final_epochs = epochs - self.final_start
        self.warmup = min(self.warmup, self._warmup(epochs))

    def cut(self, end, done):
        """Time budget: training ends at `end`, earlier than planned. A decay already running keeps its start and only
        gets shorter; one that has not started begins at the last epoch."""
        if self.final_start >= done:
            self.final_start = min(self.final_start, max(end - 1, done))
        self.final_epochs = end - self.final_start

    def phase(self, epoch):
        return 'final' if epoch >= self.final_start else 'flat'

    def factor(self, epoch, step):
        g = epoch * self.spe + step
        if g < self.warmup:
            return (g + 1) / self.warmup
        if epoch < self.final_start:
            return 1.0
        progress = ((epoch - self.final_start) * self.spe + step) / (self.final_epochs * self.spe)
        return self.low + (1 - self.low) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _clean_start(end, r):
    """First epoch of the clean last stretch: the last clean_fraction of the epochs up to where training really ends (at
    least one), the same rule as rpod.data.augment, so patience moves both together."""
    return end - max(1, round(r['clean_fraction'] * end))


class Budget:
    """Training against a time budget (RPOD.train(time=...)): how many epochs fit, measured as the run goes.

    `seconds` counts from `started` (the call to train, so the setup is included) to the end of train. `reserve` is
    time the caller needs afterwards (seconds, or reserve(stats) -> seconds with the stats below), taken off the budget
    after every epoch. An epoch costs the mean of the last WINDOW epochs (the first one too, until there are others: it
    carries the CUDA and worker start-up); clean epochs are measured apart, without augmentation they are cheaper."""

    WINDOW = 3

    def __init__(self, seconds, started, reserve=None, clock=time.time):
        self.seconds, self.started, self.clock = float(seconds), started, clock
        self.reserve = reserve if callable(reserve) else (lambda stats, s=float(reserve or 0): s)
        self.normal, self.clean = [], []
        self.val_s = self.n_val = self.save_s = 0

    def record(self, seconds, clean, val_s, n_val, save_s):
        (self.clean if clean else self.normal).append(seconds)
        self.val_s, self.n_val, self.save_s = val_s, n_val, save_s

    @property
    def epoch_s(self):
        times = self.normal[1:] if len(self.normal) > 1 else self.normal or self.clean
        return sum(times[-self.WINDOW:]) / len(times[-self.WINDOW:])

    @property
    def clean_s(self):
        return sum(self.clean) / len(self.clean) if self.clean else self.epoch_s

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

    def fits(self, normal, clean):
        return normal * self.epoch_s + clean * self.clean_s <= self.remaining()

    def planned(self, done, cap):
        """Epochs the whole run can have, from `done` already trained: at least one more, at most `cap`."""
        more = math.floor(self.remaining() / self.epoch_s) if self.epoch_s > 0 else cap
        return max(done + 1, min(cap, done + more))

    def overrun(self):
        """Seconds past the budget the caller left to rpod (negative: time to spare)."""
        return self.elapsed() - (self.seconds - self.reserve(self.stats))

    def state(self):
        return {'seconds': self.seconds, 'elapsed': self.elapsed(), 'normal': self.normal, 'clean': self.clean,
                'val_s': self.val_s, 'n_val': self.n_val, 'save_s': self.save_s}

    def load(self, state):
        """Continue a resumed run with what was left of its budget."""
        self.started = self.clock() - state['elapsed']
        self.normal, self.clean = list(state['normal']), list(state['clean'])
        self.val_s, self.n_val, self.save_s = state['val_s'], state['n_val'], state['save_s']


def _fit_to_budget(budget, schedule, done, cap, clean_from):
    """After an epoch, in time mode: move the end of training to what fits. Stops only at an epoch boundary, and only
    after an epoch that leaves room for one clean epoch; started phases never move (the decay can only get shorter, the
    clean stretch can only end sooner)."""
    if done >= schedule.end:
        return
    if clean_from is not None and done > clean_from:            # in the clean stretch: one more if it fits
        if not budget.fits(0, 1):
            schedule.cut(done, done)
        return
    if not budget.fits(1, 1):          # no room for another epoch and the clean one after it: the next is the last
        schedule.cut(done + 1 if budget.fits(0, 1) else done, done)
        return
    planned = budget.planned(done, cap)
    if done > schedule.final_start:                             # the decay has started: it can only get shorter
        if planned < schedule.end:
            schedule.cut(planned, done)
    else:
        schedule.replan(planned, done)


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

    budget: a Budget, to train for a time instead of a number of epochs (`epochs` is then the most it may train). After
    every epoch the run is replanned to the epochs that fit: the decay, the clean stretch and the end of the mosaic
    follow, but once a phase has started it does not move back. Not with multiscale.

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
    if state and budget is not None:
        schedule.final_epochs, schedule.warmup = state['final_epochs'], state['warmup']
        clean_from = state['clean_from']
        if state['mosaic_end'] is not None:
            train_set.freeze_mosaic_end(state['mosaic_end'])
        budget.load(state['budget'])
    train_set.set_epochs(schedule.end)        # the augmentation's phases follow the real end of training

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
            if clean_from is None and epoch >= _clean_start(schedule.end, r):
                clean_from = epoch
            if r['restore_best'] and epoch == clean_from and 0 <= best_epoch < epoch - 1:
                _restore_best(run / 'weights' / 'best.pt', net, ema)
                log(f'\nepochs {clean_from + 1}-{schedule.end}: back to the weights of the best epoch ({best_epoch + 1})')
            if budget is not None and train_set.mosaic_end is None and epoch >= round(MOSAIC_FRACTION * schedule.end):
                train_set.freeze_mosaic_end(epoch)   # the end of the mosaic stays put if the projection grows later
            train_set.set_epoch(epoch)
            if sampler:
                sampler.epoch, sampler.fixed = epoch, clean_from is not None and epoch >= clean_from
            _set_train_mode(net, freeze_bn)
            sums, done, seen, next_report = {}, 0, 0, t0 + PROGRESS_SECONDS
            it = iter(loader)
            for step in range(steps_per_epoch):
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
                if time.time() >= next_report:
                    next_report = time.time() + PROGRESS_SECONDS
                    log(_progress(epoch, schedule.end, step + 1, steps_per_epoch, float(sums['loss']) / done,
                                  seen, time.time() - t0))
            means = {k: float(v) / max(done, 1) for k, v in sums.items()}   # per optimizer step (effective batch)

            t_val = time.time()
            metrics = validate(ema.ema, val_set, names, device, micro, workers=workers, amp_dtype=amp_dtype)
            val_s = time.time() - t_val
            phase = schedule.phase(epoch)
            # a tie counts as an improvement: a small validation set saturates (fitness 1.0 from epoch 8 on, 3 val
            # images), and keeping the first epoch to get there shipped an undertrained model with duplicate boxes
            improved = metrics.fitness >= best_fitness
            if improved:
                best_fitness, best_epoch, stale = metrics.fitness, epoch, 0
                save_checkpoint(run / 'weights' / 'best.pt', ema.ema, model.cfg, names, imgsz, num_queries,
                                metrics.threshold, {**metrics.to_dict(), 'epoch': epoch + 1}, extra)
            elif phase == 'flat':
                stale += 1
            t_save = time.time()
            save_checkpoint(run / 'weights' / 'last.pt', ema.ema, model.cfg, names, imgsz, num_queries,
                            metrics.threshold, {**metrics.to_dict(), 'epoch': epoch + 1}, extra)
            save_s = time.time() - t_save
            seconds = time.time() - t0
            if budget is not None:     # replan before the log, so that 'epoch N/E' shows the current projection
                budget.record(seconds, clean_from is not None and epoch >= clean_from, val_s, len(val_set), save_s)
                before = schedule.end
                _fit_to_budget(budget, schedule, epoch + 1, epochs, clean_from)
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
            wait = patience(schedule.end) if callable(patience) else patience
            if phase == 'flat' and wait is not None and stale >= wait and epoch + 1 < schedule.final_start:
                schedule.final_start, early = epoch + 1, True
                train_set.set_epochs(schedule.end)
                back = r['restore_best'] and best_epoch < epoch
                if back:
                    _restore_best(run / 'weights' / 'best.pt', net, ema)
                log(f'no improvement for {wait} epochs: '
                    + (f'back to the weights of the best epoch ({best_epoch + 1}), ' if back else '')
                    + f'starting the final learning-rate decay ({schedule.final_epochs} epochs, ends at epoch {schedule.end})')
            if budget is not None:
                log(f'time: {budget.remaining() / 60:.1f} min left for epochs, ~{budget.epoch_s:.0f}s each · '
                    f'training ends at epoch {schedule.end}' + (f' (was {before})' if schedule.end != before else ''))
                if epoch + 1 == 1 and schedule.end == 1:
                    log('warning: the first epoch used up the time budget; training ends without a clean epoch')
            if save_state:
                tmp = run / 'weights' / 'state.pt.tmp'
                torch.save({'run': str(run.resolve()), 'epoch': epoch + 1, 'net': net.state_dict(), 'ema': ema.ema.state_dict(),
                            'ema_updates': ema.updates, 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                            'final_start': schedule.final_start, 'best_fitness': best_fitness, 'best_epoch': best_epoch,
                            'stale': stale, 'micro': micro, 'args': args, 'final_epochs': schedule.final_epochs,
                            'warmup': schedule.warmup, 'clean_from': clean_from, 'mosaic_end': train_set.mosaic_end,
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
