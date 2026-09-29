"""Export for the vision platform (Triton path): model.onnx + config.json in one folder.

ONNX contract (what the platform's rp_pipeline and DETR decoder expect):
  input   images  float32 [1, 3, H, W], values 0-1 (letterboxed; the model normalizes inside)
  outputs dets    float32 [1, K, 4]  boxes cxcywh, normalized 0-1
          labels  float32 [1, K, C]  class logits (the platform applies the sigmoid)
The TensorRT engine is built by the platform on each machine.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OPSET = 17   # grid_sample (deformable attention) needs >= 16


class _Exportable(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images):
        out = self.net(images)
        return out['pred_boxes'], out['pred_logits']


def platform_config(backbone, names, imgsz, threshold):
    return {
        'common': {'model_name': f'rpod-{backbone}', 'task': 'detection',
                   'image_shape': {'height': int(imgsz[0]), 'width': int(imgsz[1]), 'channels': 3}, 'max_batch': 1},
        'task_specific': {'classes': {str(i): n for i, n in sorted(names.items())},
                          'confidence': round(float(threshold), 4), 'iou': 0.7},
        'model_specific': {},
    }


def export_onnx(net, backbone, names, imgsz, threshold, output_dir, check=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / 'model.onnx'
    reference = copy.deepcopy(net).cpu().eval()
    fused = copy.deepcopy(reference).fuse()
    x = torch.rand(1, 3, int(imgsz[0]), int(imgsz[1]))
    torch.onnx.export(_Exportable(fused), (x,), str(onnx_path), input_names=['images'], output_names=['dets', 'labels'],
                      opset_version=OPSET, dynamo=False)
    (output_dir / 'config.json').write_text(json.dumps(platform_config(backbone, names, imgsz, threshold), indent=2))
    if check:
        _check(onnx_path, reference, x)
    return output_dir


def _check(onnx_path, reference, x, top=100):
    """ONNX Runtime must reproduce PyTorch. Candidates come out sorted by score, and a 1e-6 difference can swap two
    nearly tied ones, so the comparison ignores order: every confident detection must have its twin on the other side."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    dets, labels = sess.run(None, {'images': x.numpy()})
    with torch.no_grad():
        ref = reference(x)
    a = np.concatenate([dets[0], labels[0]], 1)
    b = np.concatenate([ref['pred_boxes'][0].numpy(), ref['pred_logits'][0].numpy()], 1)
    worst = 0.0
    for p, q in ((a, b), (b, a)):
        best = p[np.argsort(-p[:, 4:].max(1))[:top]]
        dist = np.abs(best[:, None, :] - q[None, :, :]).max(-1).min(1)
        worst = max(worst, float(dist.max()))
    if worst > 1e-3:
        raise RuntimeError(f'exported ONNX differs from PyTorch (worst detection mismatch {worst:.2e})')
