"""Query selection (the dense 'YOLO' head) + deformable decoder (the 'DETR' refiner).

In:  the neck maps, finest first (e.g. [P3, P4, P5]).
Out (eval):  pred_logits [B, K, C], pred_boxes [B, K, 4] (cxcywh, normalized).
Out (train): the same plus aux_outputs (every decoder layer), enc_aux_outputs (top-K proposals)
             and dn_aux_outputs / dn_meta (denoising queries).
"""
import math

import torch
import torch.nn as nn

from ..utils.boxes import cxcywh_to_xyxy, inverse_sigmoid, xyxy_to_cxcywh
from .deform_attn import MSDeformableAttention
from .ops import contiguous_grad


def bias_init_with_prob(p=0.01):
    return -math.log((1 - p) / p)


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, num_layers):
        super().__init__()
        dims = [d_in] + [d_hidden] * (num_layers - 1) + [d_out]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d=256, n_heads=8, dim_ff=1024, n_levels=3, n_points=4, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.cross_attn = MSDeformableAttention(d, n_heads, n_levels, n_points)
        self.norm2 = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d)
        self.norm3 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt, ref_boxes, memory, spatial_shapes, attn_mask, query_pos):
        # 1. queries look at each other (this is what lets duplicates suppress themselves)
        q = k = tgt + query_pos
        tgt = self.norm1(tgt + self.drop(self.self_attn(q, k, value=tgt, attn_mask=attn_mask, need_weights=False)[0]))
        # 2. each query samples a few points around its own box, on the three feature levels
        tgt = self.norm2(tgt + self.drop(self.cross_attn(tgt + query_pos, ref_boxes, memory, spatial_shapes)))
        # 3. FFN
        return self.norm3(tgt + self.drop(self.linear2(self.drop(torch.relu(self.linear1(tgt))))))


def build_denoising_group(targets, num_classes, num_queries, class_embed, num_denoising=100,
                          label_noise_ratio=0.5, box_noise_scale=1.0):
    """Training only: noisy copies of the ground truth become extra queries the decoder must fix.

    Each group holds a positive copy (small noise, must be recovered) and a negative copy
    (large noise, must be classified as background) of every GT box. Groups cannot see each other,
    and the normal queries cannot see any denoising query.
    """
    num_gts = [len(t['labels']) for t in targets]
    max_gt = max(num_gts) if num_gts else 0
    if max_gt == 0:
        return None, None, None, None
    num_group = max(1, num_denoising // max_gt)
    bs = len(num_gts)
    device = class_embed.weight.device
    cls = torch.full([bs, max_gt], num_classes, dtype=torch.int64, device=device)
    box = torch.zeros([bs, max_gt, 4], device=device)
    pad = torch.zeros([bs, max_gt], dtype=torch.bool, device=device)
    for i, n in enumerate(num_gts):
        if n:
            cls[i, :n] = targets[i]['labels']
            box[i, :n] = targets[i]['boxes']
            pad[i, :n] = True
    cls, box, pad = cls.tile(1, 2 * num_group), box.tile(1, 2 * num_group, 1), pad.tile(1, 2 * num_group)
    negative = torch.zeros([bs, 2 * max_gt, 1], device=device)
    negative[:, max_gt:] = 1
    negative = negative.tile(1, num_group, 1)
    positive = (1 - negative.squeeze(-1)) * pad
    dn_positive_idx = torch.nonzero(positive)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    num_dn = int(max_gt * 2 * num_group)

    if label_noise_ratio > 0:
        flip = torch.rand(cls.shape, device=device) < (label_noise_ratio * 0.5)
        cls = torch.where(flip & pad, torch.randint_like(cls, 0, num_classes), cls)
    if box_noise_scale > 0:
        xyxy = cxcywh_to_xyxy(box)
        diff = box[..., 2:].tile(1, 1, 2) * 0.5 * box_noise_scale
        sign = torch.randint_like(box, 0, 2) * 2.0 - 1.0
        part = torch.rand_like(box)
        part = (part + 1.0) * negative + part * (1 - negative)
        xyxy = (xyxy + part * sign * diff).clip(0.0, 1.0)
        box = xyxy_to_cxcywh(xyxy)
    dn_box_unact = inverse_sigmoid(box)
    dn_embed = class_embed(cls)

    size = num_dn + num_queries
    attn_mask = torch.zeros([size, size], dtype=torch.bool, device=device)  # True = blocked
    attn_mask[num_dn:, :num_dn] = True
    for i in range(num_group):
        s, e = max_gt * 2 * i, max_gt * 2 * (i + 1)
        attn_mask[s:e, e:num_dn] = True
        attn_mask[s:e, :s] = True
    dn_meta = {'dn_positive_idx': dn_positive_idx, 'dn_num_group': num_group, 'dn_num_split': [num_dn, num_queries]}
    return dn_embed, dn_box_unact, attn_mask, dn_meta


class Decoder(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_queries=300, n_heads=8, n_levels=3, n_points=4,
                 num_layers=6, dim_ff=1024, dropout=0.0, num_denoising=100, label_noise_ratio=0.5,
                 box_noise_scale=1.0, anchor_size=0.05):
        super().__init__()
        d = hidden_dim
        self.num_classes, self.num_queries, self.num_layers = num_classes, num_queries, num_layers
        self.num_denoising, self.label_noise_ratio, self.box_noise_scale = num_denoising, label_noise_ratio, box_noise_scale
        self.anchor_size = anchor_size
        # query selection = the dense head, applied to all tokens
        self.enc_output = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.enc_score_head = nn.Linear(d, num_classes)
        self.enc_bbox_head = MLP(d, d, 4, 3)
        # decoder
        self.query_pos_head = MLP(4, 2 * d, d, 2)
        self.layers = nn.ModuleList(DecoderLayer(d, n_heads, dim_ff, n_levels, n_points, dropout) for _ in range(num_layers))
        self.dec_score_head = nn.ModuleList(nn.Linear(d, num_classes) for _ in range(num_layers))
        self.dec_bbox_head = nn.ModuleList(MLP(d, d, 4, 3) for _ in range(num_layers))
        # denoising label embedding (index num_classes = padding)
        self.denoising_class_embed = nn.Embedding(num_classes + 1, d, padding_idx=num_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        b = bias_init_with_prob(0.01)
        nn.init.constant_(self.enc_score_head.bias, b)
        for head in self.dec_score_head:
            nn.init.constant_(head.bias, b)
        for mlp in [self.enc_bbox_head, *self.dec_bbox_head]:
            nn.init.zeros_(mlp.layers[-1].weight)
            nn.init.zeros_(mlp.layers[-1].bias)
        nn.init.xavier_uniform_(self.enc_output[0].weight)
        for layer in self.query_pos_head.layers:
            nn.init.xavier_uniform_(layer.weight)

    def _anchors(self, spatial_shapes, device, dtype, eps=0.01):
        """One anchor box per token: the cell center, size anchor_size * 2**level (in logit space)."""
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            gy, gx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing='ij')
            xy = torch.stack([(gx + 0.5) / w, (gy + 0.5) / h], -1).reshape(-1, 2)
            wh = torch.full_like(xy, self.anchor_size * 2.0 ** lvl)
            anchors.append(torch.cat([xy, wh], -1))
        anchors = torch.cat(anchors, 0)[None]
        valid = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid, anchors, torch.full_like(anchors, float('inf')))
        return anchors, valid

    def forward(self, feats, targets=None):
        spatial_shapes = [(f.shape[2], f.shape[3]) for f in feats]
        memory = torch.cat([contiguous_grad(f).flatten(2).transpose(1, 2) for f in feats], 1)  # [B, 8400, d] at 640
        bs = memory.shape[0]

        dn_embed = dn_box_unact = attn_mask = dn_meta = None
        if self.training and self.num_denoising > 0 and targets is not None:
            dn_embed, dn_box_unact, attn_mask, dn_meta = build_denoising_group(
                targets, self.num_classes, self.num_queries, self.denoising_class_embed,
                self.num_denoising, self.label_noise_ratio, self.box_noise_scale)

        # ---- query selection: one guess per token, keep the top-K (no NMS)
        anchors, valid = self._anchors(spatial_shapes, memory.device, memory.dtype)
        out_memory = self.enc_output(valid.to(memory.dtype) * memory)
        enc_logits = self.enc_score_head(out_memory)
        enc_box_unact = self.enc_bbox_head(out_memory) + anchors
        topk = enc_logits.max(-1).values.topk(self.num_queries, dim=1).indices  # [B, K]
        ref_unact = enc_box_unact.gather(1, topk[..., None].expand(-1, -1, 4))
        enc_topk_logits = enc_logits.gather(1, topk[..., None].expand(-1, -1, self.num_classes))
        enc_topk_boxes = ref_unact.sigmoid()
        tgt = out_memory.gather(1, topk[..., None].expand(-1, -1, out_memory.shape[-1])).detach()
        ref_unact = ref_unact.detach()
        if dn_embed is not None:
            tgt = torch.cat([dn_embed, tgt], 1)
            ref_unact = torch.cat([dn_box_unact, ref_unact], 1)

        # ---- decoder: refine the boxes layer by layer
        ref = ref_unact.sigmoid()
        ref_detach = ref
        out_boxes, out_logits = [], []
        n_layers = self.num_layers
        for i in range(n_layers):
            query_pos = self.query_pos_head(ref_detach)
            tgt = self.layers[i](tgt, ref_detach, memory, spatial_shapes, attn_mask, query_pos)
            delta = self.dec_bbox_head[i](tgt)
            new_ref = (delta + inverse_sigmoid(ref_detach)).sigmoid()
            if self.training:
                # the box loss of layer i also reaches layer i-1 through the (undetached) previous box
                boxes_i = new_ref if i == 0 else (delta + inverse_sigmoid(ref)).sigmoid()
                out_boxes.append(boxes_i)
                out_logits.append(self.dec_score_head[i](tgt))
            elif i == n_layers - 1:
                out_boxes.append(new_ref)
                out_logits.append(self.dec_score_head[i](tgt))
            ref = new_ref
            ref_detach = new_ref.detach()

        out_boxes, out_logits = torch.stack(out_boxes), torch.stack(out_logits)  # [L, B, n_dn + K, ...]
        if not self.training:
            return {'pred_logits': out_logits[-1], 'pred_boxes': out_boxes[-1]}

        if dn_meta is not None:
            dn_boxes, out_boxes = out_boxes.split(dn_meta['dn_num_split'], dim=2)
            dn_logits, out_logits = out_logits.split(dn_meta['dn_num_split'], dim=2)
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_boxes[-1],
               'aux_outputs': [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(out_logits[:-1], out_boxes[:-1])],
               'enc_aux_outputs': [{'pred_logits': enc_topk_logits, 'pred_boxes': enc_topk_boxes}]}
        if dn_meta is not None:
            out['dn_aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(dn_logits, dn_boxes)]
            out['dn_meta'] = dn_meta
        return out
