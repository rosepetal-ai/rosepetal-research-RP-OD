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
| `dashboard.py` | Self-refreshing page with the state of a run (progress, ETA, curves, official numbers), served by Jupyter at `/files/dashboard/index.html` |

Protocol, as the papers we compare with: train2017 → val2017, 640×640, 300 queries, up to 100 detections, AP by
pycocotools. Differences: letterbox instead of stretching to a square; multi-scale input sizes only with
`--multiscale 1`. Learning rate flat for the first half and decaying over the second (`--final-fraction 0.5`),
augmentations and multi-scale off for the last 4 % of the epochs.

The maximum model (HGNetv2-B2 · neck `l` · P2 · 6 decoder layers) is trained once; every smaller configuration loads
all its weights from it (`tests/test_checkpoint_predict_export.py`), and keeping the first N decoder layers is exact.
