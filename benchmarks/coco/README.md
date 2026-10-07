# COCO 2017: pretraining and benchmark

Outside the `rpod` package on purpose: the library has no augmentations; this folder brings the standard COCO ones
through the augment hook.

| File | |
|---|---|
| `benchmark.ipynb` | Drives everything: data, a look at the augmentations, pilot, full run, curves, official numbers |
| `prepare_coco.py` | Downloads COCO 2017 and writes it in the layout RP-OD reads (`coco/coco.yaml`) |
| `augment.py` | RT-DETR / D-FINE augmentations as an `augment(sample, ctx)` function |
| `train_coco.py` | One training, any configuration, optionally starting from another checkpoint (`--init`) |
| `eval_coco.py` | AP with pycocotools on the official annotations; `--decoder-layers N` keeps the first N layers |
| `dashboard.py` | Self-refreshing page with the state of a run, or of runs made one after another (`--run a,b,c`): progress, ETA, curves, official numbers; served by Jupyter at `/files/dashboard/index.html` |

Protocol, as the papers we compare with: train2017 → val2017, 640×640, 300 queries, up to 100 detections, AP by
pycocotools. Differences: letterbox instead of stretching to a square; multi-scale input sizes only with
`--multiscale 1`. Learning rate flat for the first half and decaying over the second (`--final-fraction 0.5`),
augmentations and multi-scale off for the last 4 % of the epochs.

The maximum model of a backbone (neck `l` · 6 decoder layers) is trained once; every smaller configuration with
that backbone loads all its weights from it (`tests/test_checkpoint_predict_export.py`), and keeping the first N
decoder layers is exact.

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
