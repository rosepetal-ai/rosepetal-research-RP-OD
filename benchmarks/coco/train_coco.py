"""Train RP-OD on COCO 2017 with the protocol of the papers we compare with. Start it detached; it writes
runs/train/<name>/ (results.csv after every epoch, weights/best.pt and last.pt) under the current directory.

Protocol: 640x640, 300 queries (the library would take 5x the busiest image, 465 on COCO), augment.py (RT-DETR recipe,
off for the last 4 % of the epochs), the full schedule (patience never brings the final decay forward), the learning
rate decaying over the second half (--final-fraction 0.5: flat for 80 % of a 72-epoch run left the model on a plateau
from epoch ~40 with nothing gained in the decay), batch 32 (split into accumulation steps automatically if it does not
fit). Everything else is the library recipe.

    python train_coco.py --name max_b2 --epochs 72                       # the maximum model: b2, neck l, 6 layers
    python train_coco.py --name b2_m3 --neck-size m --decoder-layers 3 --epochs 12 \
                         --init runs/train/max_b2/weights/best.pt         # a reduced model, starting from the maximum
    python train_coco.py --resume runs/train/max_b2/weights/state.pt     # continue an interrupted run

Every run saves its whole training state after each epoch (weights/state.pt), so it can be resumed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the RP-OD repo, so it runs without installing
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rpod.engine.trainer as trainer  # noqa: E402
import augment as augment_module  # noqa: E402
from augment import augment  # noqa: E402
from rpod import RPOD  # noqa: E402

QUERIES = 300

p = argparse.ArgumentParser()
p.add_argument('--name')
p.add_argument('--data', default='coco/coco.yaml')
p.add_argument('--epochs', type=int, default=72)
p.add_argument('--batch', type=int, default=32)
p.add_argument('--lr', type=float, default=None, help='default: the library recipe')
p.add_argument('--backbone-lr', type=float, default=None, help='default: the library recipe')
p.add_argument('--final-fraction', type=float, default=0.5, help='share of the epochs with the learning rate decaying')
p.add_argument('--backbone', default='hgnetv2-b2')
p.add_argument('--neck-size', default='l')
p.add_argument('--decoder-layers', type=int, default=6)
p.add_argument('--multiscale', type=int, default=0, help='1: train at random sizes from 480 to 800 (last epochs at 640)')
p.add_argument('--mosaic', type=int, default=1, help='mosaic augmentation with probability 0.5 (0: off)')
p.add_argument('--init', default=None, help='checkpoint to start from, any configuration (loads what fits)')
p.add_argument('--resume', default=None, help='weights/state.pt of an interrupted run: everything else comes from it')
a = p.parse_args()

if a.resume:      # same configuration, recipe, data, epochs and batch as the interrupted run
    import torch
    run_args = torch.load(a.resume, map_location='cpu', weights_only=True)['args']
    cfg = run_args['model']
    a.name, a.data, a.epochs, a.batch = run_args['name'], run_args['data'], run_args['epochs'], run_args['batch']
    a.backbone, a.neck_size = cfg['backbone'], cfg['neck']['size']
    a.decoder_layers = cfg['decoder']['layers']
    a.lr, a.backbone_lr, a.init = run_args['recipe']['lr'], run_args['recipe']['backbone_lr'], None
    a.multiscale = int(run_args.get('multiscale', False))
    a.mosaic = int(run_args['recipe'].get('mosaic', 0))
    a.final_fraction = run_args['recipe']['final_fraction']
elif not a.name:
    p.error('--name is required (unless --resume)')

trainer.queries_for = lambda max_objects: QUERIES
if a.lr is not None:
    trainer.RECIPE['lr'] = a.lr
if a.backbone_lr is not None:
    trainer.RECIPE['backbone_lr'] = a.backbone_lr
trainer.RECIPE['final_fraction'] = a.final_fraction
augment_module.P_MOSAIC = 0.5 if a.mosaic else 0.0
trainer.RECIPE['mosaic'] = a.mosaic                 # recorded in args.yaml and the resumable state
model = RPOD(backbone=a.backbone, neck_size=a.neck_size, decoder_layers=a.decoder_layers)
if a.init:
    model.net = RPOD(a.init).net      # the trainer loads what fits (fewer levels, blocks or layers) and logs a report
trainer.train(model, a.data, [640, 640], a.epochs, a.epochs, a.batch, 'cuda', augment, a.name, save_state=True,
              resume=a.resume, multiscale=bool(a.multiscale))
