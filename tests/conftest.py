import pytest
import torch

from rpod.cfg.schema import SIZES, derive, queries_for
from rpod.nn import Detector
from rpod.utils.checkpoint import save_checkpoint

NAMES = {0: 'scratch', 1: 'screw', 2: 'label'}


def make_net(num_classes=3, size='s', arch=None):
    """A network of a size (or of any architecture, for the internals), random weights."""
    torch.manual_seed(0)
    return Detector(derive(arch or SIZES[size], num_classes), pretrained_backbone=False).eval()


@pytest.fixture
def trained_ckpt(tmp_path):
    """A checkpoint with random weights: enough to exercise predict and export."""
    cfg = SIZES['s']
    net = make_net()
    path = tmp_path / 'best.pt'
    save_checkpoint(path, net, cfg, NAMES, [256, 320], queries_for(0), threshold=0.01)
    return path
