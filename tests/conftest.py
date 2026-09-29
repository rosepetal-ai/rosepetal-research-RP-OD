import pytest
import torch

from rpod.cfg.schema import derive, load_config, queries_for
from rpod.nn import Detector
from rpod.utils.checkpoint import save_checkpoint

NAMES = {0: 'scratch', 1: 'screw', 2: 'label'}


def make_net(num_classes=3, **args):
    torch.manual_seed(0)
    return Detector(derive(load_config(**args), num_classes), pretrained_backbone=False).eval()


@pytest.fixture
def trained_ckpt(tmp_path):
    """A checkpoint with random weights: enough to exercise predict and export."""
    cfg = load_config(backbone='hgnetv2-b0')
    net = make_net(backbone='hgnetv2-b0')
    path = tmp_path / 'best.pt'
    save_checkpoint(path, net, cfg, NAMES, [256, 320], queries_for(0), threshold=0.01)
    return path
