"""RPOD: the one object to create, inspect, predict with and export a detector.

    from rpod import RPOD
    model = RPOD('rpod.yaml', backbone='hgnetv2-b4')   # new model: YAML (optional) + arguments, arguments win
    model = RPOD('runs/train/exp/weights/best.pt')     # trained model: everything comes from the checkpoint
    model.info()
    results = model.predict(image, color='BGR')
    model.export()
"""
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from .cfg.schema import check_input_size, derive, load_config, positions, queries_for
from .data import DetectionDataset, check_dataset, class_map
from .engine.exporter import export_onnx
from .engine.predictor import predict as _predict
from .engine.trainer import _run_dir, train as _train
from .engine.validator import validate
from .nn import Detector
from .utils.checkpoint import build_from_checkpoint, load_checkpoint


def _default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class RPOD:
    def __init__(self, model=None, **args):
        self.ckpt_path = None
        self.net = None            # built when training starts, or loaded from a checkpoint
        self.names = None
        self.imgsz = None
        self.threshold = None
        self.data = None           # dataset the model was trained on
        self.num_queries = queries_for(0)
        self.fine_query_fraction = 0.0   # candidates reserved for the finest level (set by training with focus_small)
        if model is not None and str(model).endswith('.pt'):
            if args:
                raise ValueError('architecture arguments only apply to new models, not to a checkpoint')
            ckpt = load_checkpoint(model)
            self.ckpt_path = Path(model)
            self.cfg = ckpt['config']
            self.names, self.imgsz = ckpt['names'], ckpt['imgsz']
            self.threshold, self.num_queries = ckpt['threshold'], ckpt['num_queries']
            self.fine_query_fraction = ckpt.get('fine_query_fraction', 0.0)
            self.data = ckpt.get('data')
            self.net = build_from_checkpoint(ckpt).eval()
        else:
            self.cfg = load_config(model, **args)

    def __repr__(self):
        c = self.cfg
        state = f"trained, {len(self.names)} classes, {self.imgsz[0]}x{self.imgsz[1]}" if self.names else 'not trained'
        return (f'RPOD(backbone={c.backbone}, neck={c.neck.size}{", small_objects" if c.neck.small_objects else ""}, '
                f'decoder_layers={c.decoder.layers}; {state})')

    # -------------------------------------------------------------------------------------------- inspect
    def info(self, imgsz=None, verbose=True):
        """Parameters per block and GFLOPs at the given (or trained) input size."""
        imgsz = _size(imgsz or self.imgsz or 640)
        num_classes = len(self.names) if self.names else 1
        net = self.net if self.net is not None else Detector(derive(self.cfg, num_classes, self.num_queries),
                                                             pretrained_backbone=False)
        net = net.eval()
        d = net.decoder
        count = lambda m: sum(p.numel() for p in m.parameters())
        qsel = count(d.enc_output) + count(d.enc_score_head) + count(d.enc_bbox_head)
        blocks = {'backbone': count(net.backbone), 'neck': count(net.neck), 'query selection': qsel,
                  'decoder': count(d) - qsel - count(d.denoising_class_embed)}   # denoising is training-only
        levels = net.backbone.levels
        check_input_size(levels, imgsz, d.num_queries)
        device = next(net.parameters()).device
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            net(torch.zeros(1, 3, *imgsz, device=device))
        tokens = positions(levels, imgsz)
        summary = {**blocks, 'total': sum(blocks.values()), 'gflops': fc.get_total_flops() / 1e9, 'imgsz': imgsz,
                   'tokens': tokens, 'num_queries': d.num_queries}
        if verbose:
            print(repr(self))
            print(f"input {imgsz[0]}x{imgsz[1]} · levels {', '.join(f'P{l}' for l in levels)} · {tokens:,} positions "
                  f"· top-{d.num_queries} refined by the decoder")
            for name, n in blocks.items():
                print(f'  {name:16}{n / 1e6:8.2f} M')
            print(f"  {'total':16}{summary['total'] / 1e6:8.2f} M   {summary['gflops']:.1f} GFLOPs")
        return summary

    # -------------------------------------------------------------------------------------------- train / validate
    def train(self, data, imgsz=640, epochs=150, patience=20, batch=4, device=None, augment=None, name='exp',
              multiscale=False, focus_small=False):
        """Train on a YOLO dataset (data.yaml). Afterwards this object is the best model (runs/train/<name>/weights/best.pt).

        imgsz: int or [height, width], rounded up to multiples of 32; it is the size used in production too.
        epochs / patience: after `patience` epochs without improvement the final learning-rate decay starts early.
        batch: effective batch; if it does not fit in the GPU it is split into accumulation steps (shown in the log).
            4 by default: more optimizer steps per epoch trained clearly better on small datasets.
        augment: optional function(sample, ctx) -> sample from your augmentation library (training images only).
        multiscale: train every batch at a random size from 0.75x to 1.25x imgsz (the last epochs at imgsz), so the
            model learns objects at more scales; validation and production stay at imgsz.
        focus_small: give priority to small objects (< 32x32 pixels at the model input): they weigh more in training,
            some decoder candidates are reserved for the finest level, and the best epoch and the recommended
            threshold are chosen for them.
        Returns {'dir', 'best_epoch', 'val', 'test'} with the final metrics.
        """
        return _train(self, data, _size(imgsz), int(epochs), int(patience), int(batch), device or _default_device(),
                      augment, name, multiscale=multiscale, focus_small=focus_small)

    def val(self, data=None, split='val', device=None, batch=16, name='exp'):
        """Metrics and plots (runs/val/<name>) on a split of the training dataset or of another dataset with the same
        classes (any order). Uses the checkpoint's threshold, like production."""
        self._require_trained('validate')
        data = data or self.data
        if data is None:
            raise ValueError('no dataset: pass data="...yaml"')
        ds = check_dataset(data)
        if split not in ds['records']:
            raise ValueError(f"the dataset has no '{split}' split")
        id_map = class_map(ds['names'], self.names)
        dataset = DetectionDataset(ds['records'][split], self.imgsz, len(self.names), id_map=id_map)
        device = device or _default_device()
        out = _run_dir('runs/val', name)
        metrics = validate(self.net.to(device), dataset, self.names, device, batch,
                           threshold=0.5 if self.threshold is None else self.threshold, plots_dir=out)
        print(f"{split} of {data}, threshold {metrics.threshold:.3f} · plots in {out}")
        print(metrics.table(self.names))
        return metrics

    def _adopt(self, net, names, imgsz, threshold, num_queries, ckpt_path, data, fine_query_fraction=0.0):
        self.net, self.names, self.imgsz, self.threshold = net, names, list(imgsz), threshold
        self.num_queries, self.ckpt_path, self.data = num_queries, Path(ckpt_path), data
        self.fine_query_fraction = fine_query_fraction

    # -------------------------------------------------------------------------------------------- use
    def predict(self, source, color='RGB', conf=None, max_det=100, device=None):
        """Detections for an image array (with its color tag: RGB, BGR or GRAY), a file, a folder or a list.

        conf defaults to the threshold recommended during validation (stored in the checkpoint).
        Returns one Result per image, boxes in the pixels of the original image.
        """
        self._require_trained('predict')
        conf = self.threshold if conf is None else conf
        conf = 0.5 if conf is None else conf
        return _predict(self.net, self.names, self.imgsz, source, color, conf, max_det, device or _default_device())

    def export(self, output_dir=None):
        """model.onnx + config.json for the vision platform. Returns the folder."""
        self._require_trained('export')
        if output_dir is None:
            output_dir = self.ckpt_path.with_name(f'{self.ckpt_path.stem}_export') if self.ckpt_path else 'rpod_export'
        threshold = 0.5 if self.threshold is None else self.threshold
        return export_onnx(self.net, self.cfg.backbone, self.names, self.imgsz, threshold, output_dir)

    def _require_trained(self, what):
        if self.net is None or not self.names:
            raise RuntimeError(f'cannot {what}: this model is not trained yet. Train it, or load a checkpoint (.pt).')


STRIDE = 32


def _size(imgsz):
    """[height, width], each rounded up to a multiple of 32 (the image is letterboxed into it, never deformed)."""
    if isinstance(imgsz, int):
        imgsz = [imgsz, imgsz]
    if len(imgsz) != 2 or min(imgsz) < 1:
        raise ValueError(f'imgsz must be an int or [height, width], got {imgsz!r}')
    return [-(-int(s) // STRIDE) * STRIDE for s in imgsz]
