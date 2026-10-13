# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
import pytest
import torch

from rpod.cfg.schema import SIZES
from rpod.nn.deform_attn import ms_deform_attn

from .conftest import make_net


def test_deform_attn_reads_exact_pixel():
    shapes = [(8, 8), (4, 4)]
    value = torch.randn(1, 64 + 16, 2, 5)
    loc = torch.zeros(1, 1, 2, 2, 1, 2)
    loc[..., 0], loc[..., 1] = (3 + 0.5) / 4, (1 + 0.5) / 4
    w = torch.zeros(1, 1, 2, 2, 1)
    w[:, :, :, 1] = 1.0
    out = ms_deform_attn(value, shapes, loc, w)
    assert torch.allclose(out[0, 0], value[0, 64 + 1 * 4 + 3].flatten(), atol=1e-5)


@pytest.mark.parametrize('size', list(SIZES))
def test_every_size_builds_and_runs(size):
    net = make_net(size=size)
    with torch.no_grad():
        out = net(torch.rand(1, 3, 256, 320))
    assert out['pred_logits'].shape == (1, 300, 3) and out['pred_boxes'].shape == (1, 300, 4)
    assert (out['pred_boxes'] >= 0).all() and (out['pred_boxes'] <= 1).all()


@pytest.mark.parametrize('size', list(SIZES))
def test_fuse_keeps_outputs(size):
    """Fusing changes only the neck. With random weights many candidate scores are almost tied, so two queries can
    swap places: the detections are compared as a set."""
    net = make_net(size=size)
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        ref_feats, ref = net.neck(net.backbone(x)), net(x)
        net.fuse()
        feats, fused = net.neck(net.backbone(x)), net(x)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(ref_feats, feats))
    rows = lambda o: torch.cat([o['pred_boxes'][0], o['pred_logits'][0].sigmoid()], 1)
    dist = torch.cdist(rows(ref), rows(fused), p=float('inf'))
    assert dist.min(1).values.max() < 1e-4 and dist.min(0).values.max() < 1e-4
