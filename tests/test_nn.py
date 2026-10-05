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
@pytest.mark.parametrize('small', [False, True])
def test_every_backbone_builds_and_runs(backbone, small):
    net = make_net(backbone=backbone, small_objects=small)
    with torch.no_grad():
        out = net(torch.rand(1, 3, 256, 320))
    assert out['pred_logits'].shape == (1, 300, 3) and out['pred_boxes'].shape == (1, 300, 4)
    assert (out['pred_boxes'] >= 0).all() and (out['pred_boxes'] <= 1).all()


def test_neck_names_are_per_level():
    without = set(make_net().neck.state_dict())
    with_p2 = set(make_net(small_objects=True).neck.state_dict())
    assert without < with_p2   # the model without level 2 is a strict subset: weights transfer by name


def test_fuse_keeps_outputs():
    net = make_net()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        ref = net(x)
        fused = net.fuse()(x)
    assert torch.allclose(ref['pred_boxes'], fused['pred_boxes'], atol=1e-4)
    assert torch.allclose(ref['pred_logits'], fused['pred_logits'], atol=1e-3)


def test_fine_level_quota_in_query_selection():
    from rpod.nn.decoder import Decoder
    dec = Decoder(3, num_queries=10, n_levels=2, num_layers=1, fine_query_fraction=0.3)
    scores = torch.cat([torch.zeros(1, 16), torch.ones(1, 4)], 1)        # finest level (4x4) scores lower
    picked = dec._select(scores, [(4, 4), (2, 2)])[0].tolist()
    assert len(picked) == len(set(picked)) == 10 and sum(i < 16 for i in picked) >= 3
    dec.fine_query_fraction = 0.0
    assert set(dec._select(scores, [(4, 4), (2, 2)])[0].tolist()) >= {16, 17, 18, 19}
