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

from .cfg.schema import derive, load_config, queries_for
from .engine.exporter import export_onnx
from .engine.predictor import predict as _predict
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
        self.num_queries = queries_for(0)
        if model is not None and str(model).endswith('.pt'):
            if args:
                raise ValueError('architecture arguments only apply to new models, not to a checkpoint')
            ckpt = load_checkpoint(model)
            self.ckpt_path = Path(model)
            self.cfg = ckpt['config']
            self.names, self.imgsz = ckpt['names'], ckpt['imgsz']
            self.threshold, self.num_queries = ckpt['threshold'], ckpt['num_queries']
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
        device = next(net.parameters()).device
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            net(torch.zeros(1, 3, *imgsz, device=device))
        levels = net.backbone.levels
        tokens = sum((imgsz[0] // 2 ** l) * (imgsz[1] // 2 ** l) for l in levels)
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
