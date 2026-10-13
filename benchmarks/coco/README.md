# COCO 2017: pretraining and benchmark

Trains with the library's default augmentation (`rpod.data.augment`, the RT-DETR / D-FINE recipe plus mosaic), which
is also what `RPOD.train` uses on any dataset.

| File | |
|---|---|
| `benchmark.ipynb` | Drives everything: data, a look at the augmentations, pilot, full run, curves, official numbers |
| `prepare_coco.py` | Downloads COCO 2017 and writes it in the layout RP-OD reads (`coco/coco.yaml`) |
| `train_coco.py` | The COCO pretraining of a size (`--size s`, `m` or `l`), optionally starting from a checkpoint (`--init`) |
| `eval_coco.py` | AP with pycocotools on the official annotations |
| `dashboard.py` | Self-refreshing page with the state of a run, or of runs made one after another (`--run a,b,c`): progress, ETA, curves, official numbers; served by Jupyter at `/files/dashboard/index.html` |

Protocol, as the papers we compare with: train2017 → val2017, evaluated at 640×640, 300 queries, up to 100
detections, AP by pycocotools. Differences: letterbox instead of stretching to a square. Learning rate flat for the
first half and decaying over the second (`--final-fraction 0.5`).

| Epochs (of 120) | Input size of each batch (height and width drawn separately) | Mosaic | Other augmentations |
|---|---|---|---|
| 1 – 30 | 320 – 960 | 50 % of the images | yes |
| 31 – 100 | 480 – 800 | no | yes |
| 101 – 115 | 640 × 640 | no | yes |
| 116 – 120 | 640 × 640 | no | no (clean images) |

Each size (`s`, `m`, `l`) is pretrained once on COCO, from its ImageNet backbone (`train_coco.py --size s`). The
result is the starting point for training that size on our datasets: everything loads except the class heads.

## Architecture pilots (October 2026)

HGNetv2-B0 · neck `l` · 6 layers, 12 epochs from ImageNet, lr 2e-4 / backbone 1e-4, batch 32, mosaic; official AP of
`last.pt` on val2017. One run each, so differences of ~0.2 are noise.

| | AP | APs | ARs | AP, first 3 layers |
|---|---|---|---|---|
| RT-DETR head and losses (VFL, L1 + GIoU, top-K proposals) | 41.2 | 22.3 | 36.6 | 39.6 |
| A: boxes as edge distributions, FGL + DDF (D-FINE) | 41.4 | 21.8 | 36.3 | 39.0 |
| A + C: MAL instead of VFL (DEIM) | 41.7 | 22.9 | 36.6 | 39.4 |
| A + C + B: dense head supervised on every position | 41.6 | 23.2 | 38.1 | 39.5 |

A + C + B is what RP-OD does now: +0.4 AP, +0.9 APs and +1.5 small-object recall over the RT-DETR version. The
self-distillation did not make the decoder cut to 3 layers better at 12 epochs.
