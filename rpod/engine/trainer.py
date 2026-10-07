"""Training with one fixed recipe (RT-DETR / D-FINE / DEIM practice). The user chooses data, size, epochs, patience,
batch, device and an optional augmentation function; everything else is decided here.

Schedule: warmup, then a flat learning rate, then a cosine decay over the last 20 % of the epochs. If validation does
not improve for `patience` epochs during the flat phase, the final decay starts right away and training ends when it
is done. The best epoch is the one with the highest 0.5 * mAP50-95 + 0.5 * recall (IoU >= 0.5, at the best-F1
threshold); its EMA weights and threshold go to best.pt. The test split, if any, is evaluated once at the end with that
threshold and never used for a decision.
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

from ..cfg.schema import check_input_size, derive, queries_for, to_dict
from ..data import DetectionDataset, MultiScaleBatchSampler, check_dataset, collate, multiscale_sizes
from ..data.check import format_problems
from ..loss import HungarianMatcher, SetCriterion
from ..nn import Detector
from ..utils.checkpoint import load_weights, save_checkpoint
from .validator import validate

RECIPE = dict(lr=1e-4, backbone_lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999), warmup_steps=2000, warmup_max_fraction=0.05,
              final_fraction=0.2, final_lr_ratio=0.05, ema_decay=0.9999, ema_tau=2000, clip_grad=0.1,
              freeze_backbone_bn_below=8, clean_fraction=0.04, seed=0)
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
        self.spe = steps_per_epoch
        self.final_epochs = max(1, round(r['final_fraction'] * epochs))
        self.final_start = epochs - self.final_epochs
        self.warmup = max(1, min(r['warmup_steps'], round(r['warmup_max_fraction'] * epochs * steps_per_epoch)))
        self.low = r['final_lr_ratio']

    @property
    def end(self):
        return self.final_start + self.final_epochs

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
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


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


def _micro_batch(net, criterion, batch, imgsz, device, amp_dtype, max_objects):
    """Largest divisor of `batch` whose forward + backward fits in GPU memory. The weights are left untouched."""
    if device.type != 'cuda':
        return batch
    backup = {k: v.clone() for k, v in net.state_dict().items()}
    n = max(1, max_objects)
    for micro in [d for d in range(batch, 0, -1) if batch % d == 0]:
        images = targets = out = None
        try:
            images = torch.rand(micro, 3, *imgsz, device=device)
            targets = [{'labels': torch.zeros(n, dtype=torch.int64, device=device),
                        'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]], device=device).repeat(n, 1)} for _ in range(micro)]
            with _autocast(device, amp_dtype):
                out = net(images, targets)
            criterion(_to_float(out), targets)['loss'].backward()
            torch.cuda.synchronize(device)
            ok = True
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
          save_state=False, resume=None, multiscale=False):
    """Trains `model` (an RPOD) in place and returns {'dir', 'best_epoch', 'val', 'test'}.

    multiscale: every batch at a random size from 0.75x to 1.25x imgsz (letterboxed from the original images), and the
    last clean_fraction of the epochs (at least one) at imgsz; validation is always at imgsz.

    save_state: also write weights/state.pt after every epoch (optimizer, EMA, schedule, best so far), so a long run
    that gets interrupted can continue with resume=<that file> and the same arguments. Not part of RPOD.train.
    """
    r = RECIPE
    state = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if state and state['args']['model'] != to_dict(model.cfg):
        raise ValueError(f'{resume} is a run of another model configuration')
    torch.manual_seed(r['seed'])
    np.random.seed(r['seed'])
    random.seed(r['seed'])
    device = torch.device(device)
    workers = int(os.environ.get('RPOD_WORKERS', min(8, os.cpu_count() or 1))) if workers is None else workers

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
    sizes = multiscale_sizes(imgsz) if multiscale else [list(imgsz)]
    check_input_size(d['levels'], sizes[0], num_queries)
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

    micro = state['micro'] if state else _micro_batch(net, criterion, batch, sizes[-1], device, amp_dtype,
                                                      stats['max_objects_per_image'])   # at the largest size
    accumulate = batch // micro
    freeze_bn = micro < r['freeze_backbone_bn_below']
    if freeze_bn:
        _freeze_backbone_bn(net)
    log(f'batch {batch} = {micro} images x {accumulate} accumulation step(s)'
        + (f' · backbone BatchNorm frozen ({micro} images per step < {r["freeze_backbone_bn_below"]})' if freeze_bn else '')
        + f" · precision {str(amp_dtype).replace('torch.', '') if amp_dtype else 'float32'} · device {device}")
    clean_epochs = max(1, round(r['clean_fraction'] * epochs))
    if multiscale:
        log(f'multi-scale: {len(sizes)} sizes from {sizes[0][0]}x{sizes[0][1]} to {sizes[-1][0]}x{sizes[-1][1]}, '
            f'the last {clean_epochs} epoch(s) at {imgsz[0]}x{imgsz[1]}')

    train_set = DetectionDataset(ds['records']['train'], imgsz, len(names), augment, epochs=epochs, seed=r['seed'])
    val_set = DetectionDataset(ds['records']['val'], imgsz, len(names))
    sampler = MultiScaleBatchSampler(len(train_set), micro, sizes, imgsz, r['seed'], len(train_set) >= micro) \
        if multiscale else None
    if sampler:
        loader = DataLoader(train_set, batch_sampler=sampler, num_workers=workers, collate_fn=collate,
                            pin_memory=device.type == 'cuda', persistent_workers=workers > 0)
    else:
        loader = DataLoader(train_set, batch_size=micro, shuffle=True, num_workers=workers, collate_fn=collate,
                            drop_last=len(train_set) >= micro, pin_memory=device.type == 'cuda',
                            persistent_workers=workers > 0,
                            generator=torch.Generator().manual_seed(r['seed'] + (state['epoch'] if state else 0)))
    steps_per_epoch = max(1, math.ceil(len(loader) / accumulate))
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

    # ---- run folder
    run = Path(state['run']) if state else _run_dir(project, name)
    if not state:
        (run / 'weights').mkdir()
    args = {'data': str(Path(data).resolve()), 'imgsz': list(imgsz), 'epochs': epochs, 'patience': patience, 'batch': batch,
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
            train_set.set_epoch(epoch)
            if sampler:
                sampler.epoch, sampler.fixed = epoch, epoch >= schedule.end - clean_epochs
            _set_train_mode(net, freeze_bn)
            sums, count, seen, next_report = {}, 0, 0, t0 + PROGRESS_SECONDS
            it = iter(loader)
            for step in range(steps_per_epoch):
                factor = schedule.factor(epoch, step)
                for g in optimizer.param_groups:
                    g['lr'] = g['base_lr'] * factor
                group = []
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
                    count += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, r['clip_grad'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(net)
                seen += total_images
                if time.time() >= next_report:
                    next_report = time.time() + PROGRESS_SECONDS
                    log(_progress(epoch, schedule.end, step + 1, steps_per_epoch, float(sums['loss']) * accumulate / count,
                                  seen, time.time() - t0))
            means = {k: float(v) * accumulate / max(count, 1) for k, v in sums.items()}   # per effective batch

            metrics = validate(ema.ema, val_set, names, device, micro, workers=workers, amp_dtype=amp_dtype)
            phase = schedule.phase(epoch)
            improved = metrics.fitness > best_fitness
            if improved:
                best_fitness, best_epoch, stale = metrics.fitness, epoch, 0
                save_checkpoint(run / 'weights' / 'best.pt', ema.ema, model.cfg, names, imgsz, num_queries,
                                metrics.threshold, {**metrics.to_dict(), 'epoch': epoch + 1}, extra)
            elif phase == 'flat':
                stale += 1
            save_checkpoint(run / 'weights' / 'last.pt', ema.ema, model.cfg, names, imgsz, num_queries,
                            metrics.threshold, {**metrics.to_dict(), 'epoch': epoch + 1}, extra)
            seconds = time.time() - t0
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
            if phase == 'flat' and stale >= patience and epoch + 1 < schedule.final_start:
                schedule.final_start = epoch + 1
                log(f'no improvement for {patience} epochs: starting the final learning-rate decay '
                    f'({schedule.final_epochs} epochs, ends at epoch {schedule.end})')
            if save_state:
                tmp = run / 'weights' / 'state.pt.tmp'
                torch.save({'run': str(run.resolve()), 'epoch': epoch + 1, 'net': net.state_dict(), 'ema': ema.ema.state_dict(),
                            'ema_updates': ema.updates, 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                            'final_start': schedule.final_start, 'best_fitness': best_fitness, 'best_epoch': best_epoch,
                            'stale': stale, 'micro': micro, 'args': args}, tmp)
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
    val_metrics = validate(net, val_set, names, device, micro, threshold=threshold, plots_dir=run / 'val', workers=workers,
                           amp_dtype=amp_dtype)
    test_metrics = None
    if ds['records'].get('test'):
        test_set = DetectionDataset(ds['records']['test'], imgsz, len(names))
        test_metrics = validate(net, test_set, names, device, micro, threshold=threshold, plots_dir=run / 'test',
                                workers=workers, amp_dtype=amp_dtype)
        log(f'\ntest (never used for any decision), threshold {threshold:.3f}')
        log(test_metrics.table(names))
    model._adopt(net.eval(), names, imgsz, threshold, num_queries, run / 'weights' / 'best.pt', args['data'])
    return {'dir': run, 'best_epoch': best_epoch + 1, 'val': val_metrics, 'test': test_metrics}
