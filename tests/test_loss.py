import torch

from rpod.loss import HungarianMatcher, SetCriterion

from .conftest import make_net


def test_matcher_picks_the_matching_query():
    tgt = [{'boxes': torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.1, 0.3]]), 'labels': torch.tensor([1, 0])}]
    boxes = torch.rand(1, 10, 4) * 0.5 + 0.25
    boxes[0, 7], boxes[0, 2] = tgt[0]['boxes'][0], tgt[0]['boxes'][1]
    logits = torch.full((1, 10, 2), -5.0)
    logits[0, 7, 1] = logits[0, 2, 0] = 5.0
    (src, dst), = HungarianMatcher()({'pred_logits': logits, 'pred_boxes': boxes}, tgt)
    assert dict(zip(dst.tolist(), src.tolist())) == {0: 7, 1: 2}


def test_train_step_reaches_every_parameter():
    net = make_net().train()
    targets = [{'boxes': torch.tensor([[0.42, 0.35, 0.58, 0.12], [0.30, 0.70, 0.14, 0.10]]), 'labels': torch.tensor([0, 1])},
               {'boxes': torch.tensor([[0.72, 0.72, 0.04, 0.04]]), 'labels': torch.tensor([2])}]
    out = net(torch.rand(2, 3, 256, 256), targets)
    losses = SetCriterion(3)(out, targets)
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
