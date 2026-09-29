"""Multi-scale deformable attention, pure PyTorch (bilinear sampling with F.grid_sample).

No custom CUDA op: runs on CPU/GPU and exports to ONNX (opset >= 16 for grid_sample).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ms_deform_attn(value, spatial_shapes, sampling_locations, attention_weights):
    """
    value:              [B, S, n_heads, head_dim]   (S = sum of h*w over levels, levels concatenated)
    spatial_shapes:     list of (h, w) per level
    sampling_locations: [B, Q, n_heads, n_levels, n_points, 2]  (x, y) in [0, 1]
    attention_weights:  [B, Q, n_heads, n_levels, n_points]     (softmax over levels * points)
    returns:            [B, Q, n_heads * head_dim]
    """
    b, _, nh, hd = value.shape
    _, q, _, nl, npt, _ = sampling_locations.shape
    values = value.split([h * w for h, w in spatial_shapes], dim=1)
    grids = 2 * sampling_locations - 1  # grid_sample works in [-1, 1]
    sampled = []
    for lvl, (h, w) in enumerate(spatial_shapes):
        v = values[lvl].flatten(2).transpose(1, 2).reshape(b * nh, hd, h, w)
        g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)  # [B*nh, Q, n_points, 2]
        sampled.append(F.grid_sample(v, g, mode='bilinear', padding_mode='zeros', align_corners=False))
    aw = attention_weights.transpose(1, 2).reshape(b * nh, 1, q, nl * npt)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * aw).sum(-1)  # [B*nh, hd, Q]
    return out.view(b, nh * hd, q).transpose(1, 2)


class MSDeformableAttention(nn.Module):
    def __init__(self, d=256, n_heads=8, n_levels=3, n_points=4):
        super().__init__()
        self.d, self.n_heads, self.n_levels, self.n_points = d, n_heads, n_levels, n_points
        self.head_dim = d // n_heads
        self.sampling_offsets = nn.Linear(d, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d, d)
        self.output_proj = nn.Linear(d, d)
        self._reset_parameters()

    def _reset_parameters(self):
        # start with each head looking in a different direction, points at increasing distance
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True).values).view(self.n_heads, 1, 1, 2)
        grid = grid.repeat(1, self.n_levels, self.n_points, 1)
        grid = grid * torch.arange(1, self.n_points + 1, dtype=torch.float32).view(1, 1, -1, 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, query, ref_boxes, memory, spatial_shapes):
        """query [B, Q, d]; ref_boxes [B, Q, 4] cxcywh in [0, 1]; memory [B, S, d]."""
        b, q, _ = query.shape
        value = self.value_proj(memory).view(b, memory.shape[1], self.n_heads, self.head_dim)
        offsets = self.sampling_offsets(query).view(b, q, self.n_heads, self.n_levels, self.n_points, 2)
        weights = self.attention_weights(query).view(b, q, self.n_heads, self.n_levels * self.n_points)
        weights = weights.softmax(-1).view(b, q, self.n_heads, self.n_levels, self.n_points)
        # points live relative to the box: center + offset * (w, h) * 0.5 / n_points
        ref = ref_boxes[:, :, None, None, None, :]
        locations = ref[..., :2] + offsets / self.n_points * ref[..., 2:] * 0.5
        return self.output_proj(ms_deform_attn(value, spatial_shapes, locations, weights))
