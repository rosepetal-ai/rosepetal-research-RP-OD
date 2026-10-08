# RP-OD

Real-time object detection for industrial vision. A dense head proposes candidates and a transformer decoder refines
their boxes and removes duplicates, so there is no NMS. Architecture: `docs/architecture.svg`. Each box edge is a
probability distribution that every decoder layer sharpens, and the last layer teaches the earlier ones
(`rpod/nn/decoder.py`, `rpod/loss/criterion.py`).

Everything is used from Python through one object, `RPOD`.

## Install

```bash
pip install -e .            # add [dev] for tests, ONNX checks and the pycocotools comparison
```

## Create a model

Three sizes and nothing else to choose: classes come from the dataset and the input size is chosen when training.

```python
from rpod import RPOD

model = RPOD('s')                                # 's', 'm' (default) or 'l'
model.info(imgsz=640)                            # parameters per block and GFLOPs
```

| Size | Backbone | Neck | Decoder | Parameters | GFLOPs (640) |
|---|---|---|---|---|---|
| `s` | HGNetv2-B0 | 1 fusion block | 3 layers | 9.8 M | 26 |
| `m` | HGNetv2-B2 | 2 fusion blocks | 4 layers | 16.0 M | 45 |
| `l` | HGNetv2-B4 | 3 fusion blocks, twice as wide | 6 layers | 33.1 M | 110 |

Parameters with 80 classes (about 0.1 M less with a few). The backbone grows with the size; the neck and the decoder
are 256 channels wide in all three.

## Train

```python
model = RPOD('m')
res = model.train('parts.yaml', imgsz=640, epochs=150, patience=20, batch=4, augment=None)
res['val'].map50_95, res['val'].recall, res['val'].threshold      # final metrics of the best epoch
model.predict(image)                                                # the object is now the best model
```

Only data, input size, epochs, patience, batch, device and the augmentation function are chosen; the recipe is
fixed (adjusted with experiments on a real dataset):

| | |
|---|---|
| Optimizer | AdamW, lr 1e-4 for the whole network including the backbone, weight decay 1e-4 (none on norms, biases, embeddings) |
| Schedule | warmup, flat, cosine decay over the last 20 % of the epochs; `patience` epochs without improvement start that decay early |
| Stability | EMA of the weights (what is validated and saved), gradient clipping 0.1, mixed precision on GPU, no dropout |
| Losses | classification: matchability-aware (MAL); boxes: L1 + GIoU, and on the edge distributions FGL + self-distillation from the last layer (DDF), on the union of the matches of all layers; on every decoder layer, the denoising queries and the dense head (every position, not only the selected ones) |
| Batch | `batch` is the effective batch, 4 by default: if it does not fit in the GPU it is split into accumulation steps, shown in the log; backbone BatchNorm is frozen below 8 images per step |
| Best epoch | highest 0.5 · mAP50-95 + 0.5 · recall (IoU ≥ 0.5, at the best-F1 threshold) |

Why these values: on a real dataset (77 training images), a backbone learning rate of 1e-4 instead of the usual
1e-5 gave +0.12 mAP50-95 on validation, and batch 4 instead of 8 another +0.06 (confirmed with three seeds).
Removing the flat phase, gradient clipping 10, weight decay 1e-3 and 300 epochs did not help.

Optional multi-scale training, off by default:

```python
model.train('parts.yaml', imgsz=640, multiscale=True)    # every batch at a random size, height and width apart
```

| Epochs | Height and width of each batch, drawn separately (at imgsz 640) |
|---|---|
| first 25 % | 0.5x – 1.5x imgsz (320 – 960): wide range, aspect ratios up to 3:1 |
| up to 5/6 | 0.75x – 1.25x imgsz (480 – 800) |
| last sixth | imgsz |

Each image is letterboxed from the original at the batch's size; validation stays at `imgsz`. A batch that does not
fit in the GPU at its size is split into accumulation steps, so the large sizes cost memory only when they come up.

The input needs at least as many candidate positions as the decoder keeps (300): 128×128 or larger.

`runs/train/<name>/` (the name gets a number if it exists):

```
weights/best.pt   weights/last.pt   results.csv (one row per epoch, written as it goes)   args.yaml
val/   test/      metrics.json, pr_curves.png, confusion_matrix.png, predictions.jpg (up to 16 images)
```

`best.pt` stores the recommended threshold (best mean F1 on val), used by default by `predict`, `val` and `export`.
Checkpoints from RP-OD up to `ba7e126`, before the box head became edge distributions, need that version.
The test split, if there is one, is evaluated once at the end with that threshold and never used for a decision.

## Validate

```python
model = RPOD('runs/train/exp/weights/best.pt')
m = model.val()                                   # val split of the training dataset -> runs/val/exp
m = model.val(data='other.yaml', split='test')    # another dataset: same class names, any order
m.map50_95, m.map50, m.precision, m.recall, m.f1, m.threshold, m.mean_iou, m.fp_per_image, m.fitness
m.per_class, m.confusion_matrix                   # confusion rows: true class + background, columns: predicted + background
```

mAP follows the COCO definition (checked against pycocotools); precision, recall, F1, mean IoU of the hits and false
positives per image are at the threshold with IoU ≥ 0.5. `m.map_small`, `m.map_medium`, `m.map_large`: mAP50-95 by
object size (small < 32×32, medium < 96×96), measured in pixels at the model input, with the COCO rules (also in
`results.csv` and in the training log).

## Predict and export (trained model)

```python
model = RPOD('runs/train/exp/weights/best.pt')
results = model.predict(image, color='BGR')     # numpy uint8 array + color tag, a file, a folder or a list
r = results[0]
r.boxes.xyxy, r.boxes.xyxyn, r.boxes.conf, r.boxes.cls, r.names
r.save('out.jpg')

model.export()                                   # model.onnx + config.json for the vision platform (Triton)
```

`predict` letterboxes exactly like the platform's GPU pre-processing, and uses by default the confidence threshold
recommended during validation. `export` fuses the neck for inference and checks the ONNX against PyTorch.

## Status

| Part | State |
|---|---|
| Model, configuration, `info`, `predict`, `export` | done |
| Data: YOLO format, dataset check with cache, augmentation hook | done |
| Training, validation metrics and plots | done |
| `rpod` branch in the vision platform (Triton) | next |

## Datasets

YOLO format: a `data.yaml` with `train` and `val` (required), `test` (optional, only ever used for a final
evaluation), and `names`. Every image and label is checked once (cached until a file changes): unreadable
images, images that are not 8-bit, segmentation polygons, classes out of range and boxes outside the image are
errors, listed with file and reason; identical images across splits and a missing `test` are warnings.

```python
from rpod.data import check_dataset
ds = check_dataset('parts.yaml')      # raises with the full list of errors; returns stats and warnings
```

There is no augmentation inside RP-OD. An external library plugs in with one function, applied to training
samples before the letterbox (see `rpod/data/dataset.py` for the full contract):

```python
def augment(sample, ctx):
    # sample.image uint8 RGB, sample.boxes xyxy pixels, sample.labels, sample.path
    # ctx.sample(k), ctx.epoch, ctx.epochs, ctx.imgsz, ctx.rng
    return sample
```

## COCO benchmark and pretraining

`benchmarks/coco/` trains and evaluates RP-OD on COCO 2017 with the protocol of the papers (AP with pycocotools,
640×640), using the standard augmentations through the hook. The model it trains also serves as pretrained weights for
any smaller configuration.

## Tests

```bash
python3 -m pytest tests -q
```
