# RP-OD

Real-time object detection for industrial vision. A dense, YOLO-like head proposes candidates and a DETR-style
decoder refines their boxes and removes duplicates, so there is no NMS. Architecture: `docs/architecture.svg`.

Everything is used from Python through one object, `RPOD`.

## Install

```bash
pip install -e .            # add [dev] for tests and ONNX checks
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
| Training, validation metrics and plots | next |

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

## Tests

```bash
python3 -m pytest tests -q
```
