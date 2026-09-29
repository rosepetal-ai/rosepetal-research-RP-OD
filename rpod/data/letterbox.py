"""Letterbox exactly as the vision platform's GPU pre-processing (rp_preprocess_cuda.cu, letterbox + antialias).

Scale = the largest that fits the whole image (upscaling allowed), computed in float32; resized size rounded down;
image centered (odd pixel to the bottom/right); padding 114; bilinear resampling with half-pixel centers and
antialias, in float32 without rounding back to uint8; then x / 255. The model applies its own normalization inside.
"""
import numpy as np
import torch
import torch.nn.functional as F

PAD_VALUE = 114
_F = np.float32


def letterbox_geometry(in_h, in_w, out_h, out_w):
    """(scale, resized_h, resized_w, pad_top, pad_left), with the kernel's float32 arithmetic."""
    scale = min(_F(out_h) / _F(in_h), _F(out_w) / _F(in_w))
    rh = max(1, int(np.floor(_F(in_h) * scale)))
    rw = max(1, int(np.floor(_F(in_w) * scale)))
    return float(scale), rh, rw, (out_h - rh) // 2, (out_w - rw) // 2


def letterbox(rgb, out_h, out_w, device='cpu', antialias=True):
    """rgb: uint8 H x W x 3 array. Returns (tensor [3, out_h, out_w] in [0, 1], geometry)."""
    h, w = rgb.shape[:2]
    geom = letterbox_geometry(h, w, out_h, out_w)
    _, rh, rw, top, left = geom
    x = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).permute(2, 0, 1)[None].float()   # one copy at most
    x = F.interpolate(x, size=(rh, rw), mode='bilinear', align_corners=False, antialias=antialias)
    out = torch.full((1, 3, out_h, out_w), float(PAD_VALUE), device=device)
    out[..., top:top + rh, left:left + rw] = x
    return out[0] * float(_F(1.0) / _F(255.0)), geom
