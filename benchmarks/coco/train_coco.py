# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Train RP-OD on COCO 2017 with the protocol of the papers we compare with. Start it detached; it writes
runs/train/<name>/ (results.csv after every epoch, weights/best.pt and last.pt) under the current directory.

Protocol: validation at 640x640, 300 queries (the library would take 5x the busiest image, 465 on COCO), the library's
default augmentation (rpod.data.augment: RT-DETR recipe, mosaic in the first quarter of the epochs, off for the last
4 %), multi-scale training (height and
width drawn separately per batch: 320-960 in the first quarter, 480-800 until the last sixth, then 640), the full
schedule (patience never brings the final decay forward), the learning
rate decaying over the second half (--final-fraction 0.5: flat for 80 % of a 72-epoch run left the model on a plateau
from epoch ~40 with nothing gained in the decay), batch 32 (split into accumulation steps automatically if it does not
fit). Everything else is the library recipe.

    python train_coco.py --name rpod_s --size s                          # COCO pretraining of the size s
    python train_coco.py --name pilot --size s --epochs 12               # a 12-epoch pilot
    python train_coco.py --resume runs/train/rpod_s/weights/state.pt     # continue an interrupted run

Every run saves its whole training state after each epoch (weights/state.pt), so it can be resumed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the RP-OD repo, so it runs without installing

import rpod.data.augment as augment_module  # noqa: E402
import rpod.engine.trainer as trainer  # noqa: E402
from rpod import RPOD  # noqa: E402
from rpod.cfg.schema import SIZES, from_dict, size_of  # noqa: E402

QUERIES = 300
SIZE_LR = {'s': (2e-4, 1e-4), 'm': (2e-4, 2e-5), 'l': (2e-4, 1e-5)}   # head / backbone: D-FINE's 0.5x, 0.1x, 0.05x

p = argparse.ArgumentParser()
p.add_argument('--name')
p.add_argument('--data', default='coco/coco.yaml')
p.add_argument('--epochs', type=int, default=72)
p.add_argument('--batch', type=int, default=32)
p.add_argument('--size', choices=list(SIZES), default='s')
p.add_argument('--lr', type=float, default=None, help='default: 2e-4')
p.add_argument('--backbone-lr', type=float, default=None, help='default: 1e-4 (s), 2e-5 (m), 1e-5 (l)')
p.add_argument('--final-fraction', type=float, default=0.5, help='share of the epochs with the learning rate decaying')
p.add_argument('--multiscale', type=int, default=1,
               help='height and width drawn separately per batch: 320-960 in the first quarter of the epochs, 480-800 '
                    'until the last sixth, then 640 (0: always 640)')
p.add_argument('--mosaic', type=int, default=1,
               help='mosaic augmentation with probability 0.5 in the first quarter of the epochs (0: off)')
p.add_argument('--init', default=None, help='checkpoint to start from, any size (loads what fits)')
p.add_argument('--resume', default=None, help='weights/state.pt of an interrupted run: everything else comes from it')
a = p.parse_args()

if a.resume:      # same configuration, recipe, data, epochs and batch as the interrupted run
    import torch
    run_args = torch.load(a.resume, map_location='cpu', weights_only=True)['args']
    a.size = size_of(from_dict(run_args['model']))
    if a.size is None:
        p.error(f'{a.resume} is a run of a model that is not a standard size: resume it with the RP-OD it started with')
    a.name, a.data, a.epochs, a.batch = run_args['name'], run_args['data'], run_args['epochs'], run_args['batch']
    a.lr, a.backbone_lr, a.init = run_args['recipe']['lr'], run_args['recipe']['backbone_lr'], None
    a.multiscale = int(run_args.get('multiscale', False))
    a.mosaic = int(run_args['recipe'].get('mosaic', 0))
    a.final_fraction = run_args['recipe']['final_fraction']
elif not a.name:
    p.error('--name is required (unless --resume)')

trainer.queries_for = lambda max_objects: QUERIES
trainer.RECIPE['lr'] = a.lr if a.lr is not None else SIZE_LR[a.size][0]
trainer.RECIPE['backbone_lr'] = a.backbone_lr if a.backbone_lr is not None else SIZE_LR[a.size][1]
trainer.RECIPE['final_fraction'] = a.final_fraction
augment_module.P_MOSAIC = 0.5 if a.mosaic else 0.0
trainer.RECIPE['mosaic'] = a.mosaic                 # recorded in args.yaml and the resumable state
model = RPOD(a.size)
if a.init:
    model.net = RPOD(a.init).net      # the trainer loads what fits (fewer levels, blocks or layers) and logs a report
trainer.train(model, a.data, [640, 640], a.epochs, a.epochs, a.batch, 'cuda', augment_module.default_augment, a.name,
              save_state=True, resume=a.resume, multiscale=bool(a.multiscale))
