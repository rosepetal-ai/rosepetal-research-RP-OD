# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
import torch

from rpod.loss import HungarianMatcher, SetCriterion
from rpod.nn.decoder import REG_MAX, bin_positions, box_to_distance, distance_to_box
from rpod.utils.boxes import cxcywh_to_xyxy

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


def test_edge_bins_are_symmetric_and_finer_near_the_initial_edge():
    w = bin_positions()
    assert len(w) == REG_MAX + 1 and w[REG_MAX // 2] == 0 and torch.allclose(w, -w.flip(0))
    steps = w.diff()
    assert (steps > 0).all() and steps[REG_MAX // 2] < steps[1] < steps[0]


def test_edge_distances_round_trip():
    ref = torch.tensor([[0.5, 0.4, 0.2, 0.1], [0.3, 0.6, 0.05, 0.3]])
    xyxy = torch.tensor([[0.38, 0.36, 0.62, 0.44], [0.27, 0.40, 0.35, 0.80]])
    assert torch.allclose(cxcywh_to_xyxy(distance_to_box(ref, box_to_distance(ref, xyxy))), xyxy, atol=1e-6)
    assert torch.allclose(distance_to_box(ref, torch.zeros(2, 4)), ref)      # distance 0 = the initial box


def test_fgl_target_bins_interpolate_the_true_edge():
    ref = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    targets = [{'labels': torch.tensor([0]), 'boxes': torch.tensor([[0.52, 0.49, 0.26, 0.17]])}]
    gt = (targets[0]['labels'], targets[0]['boxes'])
    left, w_left, w_right = SetCriterion(3)._edge_targets(ref, gt, (torch.tensor([0]),) * 3)
    w = bin_positions()
    dist = box_to_distance(ref[0], cxcywh_to_xyxy(targets[0]['boxes']))[0]
    assert torch.allclose(w_left * w[left] + w_right * w[left + 1], dist, atol=1e-6)


def test_go_union_keeps_the_object_most_layers_agree_on():
    a = [(torch.tensor([3, 5]), torch.tensor([0, 1]))]
    b = [(torch.tensor([3, 7]), torch.tensor([1, 1]))]      # query 3 matched to object 1 by one layer,
    c = [(torch.tensor([3]), torch.tensor([0]))]           # to object 0 by two
    (q, o), = SetCriterion._union([a, b, c])
    assert dict(zip(q.tolist(), o.tolist())) == {3: 0, 5: 1, 7: 1}


def test_distillation_vanishes_only_when_a_layer_agrees_with_the_last():
    torch.manual_seed(0)
    teacher = torch.randn(1, 5, 4 * (REG_MAX + 1))
    out = {'pred_corners': teacher.clone(), 'teacher_corners': teacher, 'teacher_logits': torch.randn(1, 5, 3)}
    idx = (torch.tensor([0]), torch.tensor([2]))
    assert SetCriterion._ddf(out, idx, torch.tensor([0.8]), 1.0, 1.0, 1.0).abs() < 1e-6
    out['pred_corners'] = teacher + torch.randn_like(teacher)
    assert SetCriterion._ddf(out, idx, torch.tensor([0.8]), 1.0, 1.0, 1.0) > 0.01


def test_matching_of_every_token_image_by_image_equals_the_batched_one_and_skips_invalid_tokens():
    torch.manual_seed(0)
    targets = [{'boxes': torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.1, 0.3]]), 'labels': torch.tensor([1, 0])},
               {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.int64)},
               {'boxes': torch.tensor([[0.5, 0.5, 0.4, 0.2]]), 'labels': torch.tensor([2])}]
    out = {'pred_logits': torch.randn(3, 50, 3), 'pred_boxes': torch.rand(3, 50, 4) * 0.5 + 0.25}
    every = {**out, 'valid': torch.ones(1, 50, 1, dtype=torch.bool)}
    m = HungarianMatcher()
    batched, per_image = m.match_all([out, every], targets)
    for (a, b), (c, d) in zip(batched, per_image):
        assert torch.equal(a, c) and torch.equal(b, d)
    best = batched[0][0]
    every['valid'][0, best] = False
    assert not set(m.match_all([every], targets)[0][0][0].tolist()) & set(best.tolist())


def test_dense_head_is_supervised_on_every_token():
    net = make_net().train()
    targets = [{'boxes': torch.tensor([[0.42, 0.35, 0.58, 0.12]]), 'labels': torch.tensor([0])}]
    out = net(torch.rand(1, 3, 128, 128), targets)
    enc = out['enc_aux_outputs'][0]
    assert enc['pred_logits'].shape[1] == (16 * 16 + 8 * 8 + 4 * 4) > net.decoder.num_queries
