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


def test_object_weights_scale_only_that_objects_terms():
    torch.manual_seed(0)
    targets = [{'boxes': torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.05, 0.05]]), 'labels': torch.tensor([1, 0])}]
    out = {'pred_logits': torch.randn(1, 10, 2), 'pred_boxes': torch.rand(1, 10, 4) * 0.5 + 0.25}
    crit = SetCriterion(2)
    run = lambda w: crit(out, [dict(targets[0], weights=torch.tensor(w))])
    base, ones = crit(out, targets), run([1.0, 1.0])
    assert all(torch.allclose(base[k], ones[k]) for k in base)                       # weights 1: nothing changes
    w13, w10, w01, w00 = run([1.0, 3.0]), run([1.0, 0.0]), run([0.0, 1.0]), run([0.0, 0.0])
    for k in ('bbox', 'giou'):                                                       # linear in each object's weight
        assert torch.allclose(w13[k], w10[k] + 3 * w01[k])
    assert torch.allclose(w13['vfl'], w10['vfl'] + 3 * (w01['vfl'] - w00['vfl']))  # negatives are not weighted


def test_small_object_weights_average_one():
    from rpod.engine.trainer import _small_object_weights
    images = torch.zeros(1, 3, 64, 64)
    targets = [{'boxes': torch.tensor([[0.5, 0.5, 0.6, 0.6], [0.2, 0.2, 0.25, 0.25], [0.8, 0.8, 0.05, 0.05]]),
                'labels': torch.tensor([0, 0, 0])}]
    _small_object_weights([(images, targets, None)], cap=3.0)
    w = targets[0]['weights']                   # sides 38 px (large), 16 px, 3.2 px (capped)
    assert torch.allclose(w / w[0], torch.tensor([1.0, 2.0, 3.0])) and torch.isclose(w.mean(), torch.tensor(1.0))
