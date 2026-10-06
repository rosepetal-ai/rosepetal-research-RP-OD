import pytest
import torch

from rpod.cfg.schema import BACKBONES
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


@pytest.mark.parametrize('backbone', list(BACKBONES))
def test_every_backbone_builds_and_runs(backbone):
    net = make_net(backbone=backbone)
    with torch.no_grad():
        out = net(torch.rand(1, 3, 256, 320))
    assert out['pred_logits'].shape == (1, 300, 3) and out['pred_boxes'].shape == (1, 300, 4)
    assert (out['pred_boxes'] >= 0).all() and (out['pred_boxes'] <= 1).all()


def test_smaller_necks_are_subsets_by_name():
    s, m, l = (set(make_net(neck_size=n).neck.state_dict()) for n in ('s', 'm', 'l'))
    assert s < m < l   # a smaller neck has the first fusion blocks of a bigger one: weights transfer by name


def test_fuse_keeps_outputs():
    net = make_net()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        ref = net(x)
        fused = net.fuse()(x)
    assert torch.allclose(ref['pred_boxes'], fused['pred_boxes'], atol=1e-4)
    assert torch.allclose(ref['pred_logits'], fused['pred_logits'], atol=1e-3)
