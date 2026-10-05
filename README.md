# RP-OD

Real-time object detection for industrial vision. A dense, YOLO-like head proposes candidates and a DETR-style
decoder refines their boxes and removes duplicates, so there is no NMS. Architecture: `docs/architecture.svg`.

Everything is used from Python through one object, `RPOD`.

## Install

```bash
pip install -e .            # add [dev] for tests, ONNX checks and the pycocotools comparison
```

## Create a model

The model YAML only describes the architecture: four knobs, each explained in `rpod/cfg/models/rpod.yaml` and
`docs/MODEL_CONFIG.md`. Classes come from the dataset and the input size is chosen when training.
Arguments win over the YAML.

```python
from rpod import RPOD

model = RPOD()                                   # defaults: hgnetv2-b2, neck m, 3 decoder layers
model = RPOD('rpod.yaml', backbone='hgnetv2-b4', neck_size='l', small_objects=True, decoder_layers=4)
model.info(imgsz=640)                            # parameters per block and GFLOPs
```

| Knob | Values | Default |
|---|---|---|
| `backbone` | `hgnetv2-b0` · `hgnetv2-b2` · `hgnetv2-b4` · `hgnetv2-b5` · `resnet18` | `hgnetv2-b2` |
| `neck_size` | `s` · `m` · `l` | `m` |
| `small_objects` | `True` · `False` | `False` |
| `decoder_layers` | 3 – 6 | 3 |

## Train

```python
model = RPOD(backbone='hgnetv2-b2')
res = model.train('parts.yaml', imgsz=640, epochs=150, patience=20, batch=4, augment=None)
res['val'].map50_95, res['val'].recall, res['val'].threshold      # final metrics of the best epoch
model.predict(image)                                                # the object is now the best model
```

Only data, input size, epochs, patience, batch, device and the augmentation function are chosen; the recipe is
fixed (RT-DETR / D-FINE / DEIM practice, adjusted with experiments on a real dataset):

| | |
|---|---|
| Optimizer | AdamW, lr 1e-4 for the whole network including the backbone, weight decay 1e-4 (none on norms, biases, embeddings) |
| Schedule | warmup, flat, cosine decay over the last 20 % of the epochs; `patience` epochs without improvement start that decay early |
| Stability | EMA of the weights (what is validated and saved), gradient clipping 0.1, mixed precision on GPU, no dropout |
| Batch | `batch` is the effective batch, 4 by default: if it does not fit in the GPU it is split into accumulation steps, shown in the log; backbone BatchNorm is frozen below 8 images per step |
| Best epoch | highest 0.5 · mAP50-95 + 0.5 · recall (IoU ≥ 0.5, at the best-F1 threshold) |

Why these values: on a real dataset (77 training images), a backbone learning rate of 1e-4 instead of the usual
1e-5 gave +0.12 mAP50-95 on validation, and batch 4 instead of 8 another +0.06 (confirmed with three seeds).
Removing the flat phase, gradient clipping 10, weight decay 1e-3 and 300 epochs did not help.

The input needs at least as many candidate positions as the decoder keeps (300): 128×128 or larger.

`runs/train/<name>/` (the name gets a number if it exists):

```
weights/best.pt   weights/last.pt   results.csv (one row per epoch, written as it goes)   args.yaml
val/   test/      metrics.json, pr_curves.png, confusion_matrix.png, predictions.jpg (up to 16 images)
```

`best.pt` stores the recommended threshold (best mean F1 on val), used by default by `predict`, `val` and `export`.
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
