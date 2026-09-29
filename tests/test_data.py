import numpy as np
import pytest
import torch
from PIL import Image

from rpod.data import letterbox, letterbox_geometry, load_images, to_rgb

f = np.float32


def reference_kernel(img, out_h, out_w, pad=114):
    """The platform kernel's formulas (rp_preprocess_cuda.cu, letterbox, no antialias), written in numpy."""
    ih, iw = img.shape[:2]
    s = min(f(out_h) / f(ih), f(out_w) / f(iw))
    rh, rw = max(1, int(np.floor(f(ih) * s))), max(1, int(np.floor(f(iw) * s)))
    top, left = (out_h - rh) // 2, (out_w - rw) // 2
    sy = np.clip((np.arange(rh, dtype=f) + f(.5)) * (f(ih) / f(rh)) - f(.5), 0, ih - 1).astype(f)
    sx = np.clip((np.arange(rw, dtype=f) + f(.5)) * (f(iw) / f(rw)) - f(.5), 0, iw - 1).astype(f)
    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
    y1, x1 = np.minimum(y0 + 1, ih - 1), np.minimum(x0 + 1, iw - 1)
    wy, wx = (sy - y0).astype(f)[:, None, None], (sx - x0).astype(f)[None, :, None]
    src = img.astype(f)
    raw = (src[y0][:, x0] * ((1 - wy) * (1 - wx)) + src[y0][:, x1] * ((1 - wy) * wx)
           + src[y1][:, x0] * (wy * (1 - wx)) + src[y1][:, x1] * (wy * wx))
    out = np.full((3, out_h, out_w), f(pad) * (f(1) / f(255)), f)
    out[:, top:top + rh, left:left + rw] = raw.transpose(2, 0, 1) * (f(1) / f(255))
    return out


@pytest.mark.parametrize('in_hw,out_hw', [((1080, 1920), (640, 640)), ((301, 177), (256, 320)), ((100, 100), (640, 480))])
def test_letterbox_matches_platform_kernel(in_hw, out_hw):
    img = np.random.default_rng(0).integers(0, 256, (*in_hw, 3), dtype=np.uint8)
    ours, _ = letterbox(img, *out_hw, antialias=False)
    assert np.abs(ours.numpy() - reference_kernel(img, *out_hw)).max() < 1e-5


def test_geometry_centered_and_rounded_down():
    scale, rh, rw, top, left = letterbox_geometry(1080, 1920, 640, 640)
    assert (rh, rw) == (360, 640) and (top, left) == (140, 0) and abs(scale - 1 / 3) < 1e-6


def test_antialias_path_runs_and_pads():
    img = np.full((1080, 1920, 3), 200, np.uint8)
    t, (_, rh, rw, top, left) = letterbox(img, 640, 640)
    assert torch.allclose(t[:, top + 5, 5], torch.full((3,), 200 / 255), atol=1e-5)   # content
    assert torch.allclose(t[:, 0, 0], torch.full((3,), 114 / 255), atol=1e-6)        # padding


def test_color_handling_never_touches_the_input():
    bgr = np.zeros((4, 5, 3), np.uint8)
    bgr[..., 0] = 255                                  # blue in BGR
    bgr.setflags(write=False)
    rgb = to_rgb(bgr, 'BGR')
    assert rgb[0, 0].tolist() == [0, 0, 255] and bgr[0, 0].tolist() == [255, 0, 0]
    assert to_rgb(np.full((4, 5), 7, np.uint8), 'GRAY').shape == (4, 5, 3)
    assert to_rgb(np.zeros((4, 5, 4), np.uint8), 'RGB').shape == (4, 5, 3)
    with pytest.raises(ValueError, match='8-bit'):
        to_rgb(np.zeros((4, 5, 3), np.uint16))


def test_load_images_from_file_folder_and_list(tmp_path):
    for i in range(2):
        Image.fromarray(np.zeros((10, 12, 3), np.uint8)).save(tmp_path / f'{i}.png')
    assert len(load_images(tmp_path)) == 2
    assert len(load_images([str(tmp_path / '0.png'), np.zeros((3, 3, 3), np.uint8)])) == 2
