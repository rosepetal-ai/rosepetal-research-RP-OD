"""Neck (hybrid encoder): input projection -> global attention on the coarsest level -> FPN + PAN fusion.

Modules are keyed by feature level ('p3', 'p4', ...), not by position, so the same names mean the same thing with or
without level 2: weights of a model built with small objects load into one built without, and the other way round.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import contiguous_grad


@torch.no_grad()
def fuse_conv_bn(conv, bn):
    """Fold a BatchNorm into the preceding convolution (same output in eval mode)."""
    std = (bn.running_var + bn.eps).sqrt()
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True).to(conv.weight.device)
    fused.weight.data = conv.weight.data * (bn.weight / std).view(-1, 1, 1, 1)
    bias = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias.data = bn.bias + (bias - bn.running_mean) * bn.weight / std
    return fused


class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse(self):
        if isinstance(self.bn, nn.BatchNorm2d):
            self.conv, self.bn = fuse_conv_bn(self.conv, self.bn), nn.Identity()


class RepVggBlock(nn.Module):
    """3x3 and 1x1 branches, summed. For export both merge into one 3x3 convolution."""

    def __init__(self, c):
        super().__init__()
        self.conv1 = ConvBNAct(c, c, 3, act=False)
        self.conv2 = ConvBNAct(c, c, 1, act=False)
        self.act = nn.SiLU(inplace=True)
        self.fused = None

    def forward(self, x):
        if self.fused is not None:
            return self.act(self.fused(x))
        return self.act(self.conv1(x) + self.conv2(x))

    @torch.no_grad()
    def fuse(self):
        if self.fused is not None:
            return
        k3 = fuse_conv_bn(self.conv1.conv, self.conv1.bn)
        k1 = fuse_conv_bn(self.conv2.conv, self.conv2.bn)
        k3.weight.data += F.pad(k1.weight.data, [1, 1, 1, 1])
        k3.bias.data += k1.bias.data
        self.fused = k3
        del self.conv1, self.conv2


class FuseBlock(nn.Module):
    """Two 1x1 branches, one through n RepVgg blocks, summed (CSP-Rep layer)."""

    def __init__(self, c_in, c_out, n, expansion):
        super().__init__()
        h = int(c_out * expansion)
        self.conv1 = ConvBNAct(c_in, h, 1)
        self.conv2 = ConvBNAct(c_in, h, 1)
        self.blocks = nn.Sequential(*[RepVggBlock(h) for _ in range(n)])
        self.conv3 = ConvBNAct(h, c_out, 1) if h != c_out else nn.Identity()

    def forward(self, x):
        return self.conv3(self.blocks(self.conv1(x)) + self.conv2(x))


def sincos_pos_2d(h, w, dim, temperature=10000.0, device=None):
    """Fixed 2D sin-cos position encoding, row-major like feature.flatten(2): [1, h*w, dim]."""
    gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device),
                            torch.arange(w, dtype=torch.float32, device=device), indexing='ij')
    q = dim // 4
    omega = 1.0 / temperature ** (torch.arange(q, dtype=torch.float32, device=device) / q)
    ox = gx.flatten()[:, None] * omega[None]
    oy = gy.flatten()[:, None] * omega[None]
    return torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], dim=1)[None]


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, ffn_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.linear1 = nn.Linear(d, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, src, pos):
        q = k = src + pos
        src = self.norm1(src + self.self_attn(q, k, value=src, need_weights=False)[0])
        return self.norm2(src + self.linear2(F.gelu(self.linear1(src))))


class Neck(nn.Module):
    def __init__(self, in_channels, hidden_dim, attn_layers, attn_heads, attn_ffn, fusion_blocks, fusion_expansion):
        """in_channels: {level: channels} for consecutive levels, e.g. {3: 256, 4: 512, 5: 1024}."""
        super().__init__()
        self.levels = sorted(in_channels)
        h, finest, coarsest = hidden_dim, self.levels[0], self.levels[-1]
        self.input_proj = nn.ModuleDict({
            f'p{l}': nn.Sequential(nn.Conv2d(c, h, 1, bias=False), nn.BatchNorm2d(h)) for l, c in in_channels.items()})
        self.attention = nn.ModuleList(EncoderLayer(h, attn_heads, attn_ffn) for _ in range(attn_layers))
        # top-down (FPN): lateral 1x1 on level l, upsample, fuse into level l-1
        self.lateral = nn.ModuleDict({f'p{l}': ConvBNAct(h, h, 1) for l in self.levels if l != finest})
        self.fuse_td = nn.ModuleDict({f'p{l}': FuseBlock(2 * h, h, fusion_blocks, fusion_expansion)
                                      for l in self.levels if l != coarsest})
        # bottom-up (PAN): stride-2 conv from level l, fuse into level l+1
        self.down = nn.ModuleDict({f'p{l}': ConvBNAct(h, h, 3, 2) for l in self.levels if l != coarsest})
        self.fuse_bu = nn.ModuleDict({f'p{l}': FuseBlock(2 * h, h, fusion_blocks, fusion_expansion)
                                      for l in self.levels if l != finest})

    def forward(self, feats):
        """feats: {level: map}. Returns the fused maps as a list, finest first."""
        proj = {l: self.input_proj[f'p{l}'](feats[l]) for l in self.levels}
        top = proj[self.levels[-1]]
        if len(self.attention):
            b, c, h, w = top.shape
            tokens = contiguous_grad(top).flatten(2).transpose(1, 2)
            pos = sincos_pos_2d(h, w, c, device=top.device).to(tokens.dtype)
            for layer in self.attention:
                tokens = layer(tokens, pos)
            top = tokens.transpose(1, 2).reshape(b, c, h, w)
        lateral, x = {}, top
        for l in reversed(self.levels[1:]):
            lateral[l] = self.lateral[f'p{l}'](x)
            x = self.fuse_td[f'p{l - 1}'](torch.cat([F.interpolate(lateral[l], scale_factor=2.0, mode='nearest'), proj[l - 1]], 1))
        outs = [x]
        for l in self.levels[:-1]:
            x = self.fuse_bu[f'p{l + 1}'](torch.cat([self.down[f'p{l}'](x), lateral[l + 1]], 1))
            outs.append(x)
        return outs

    def fuse(self):
        """Merge BatchNorms and RepVgg branches for export. Same outputs, fewer operations."""
        for key, seq in self.input_proj.items():
            self.input_proj[key] = fuse_conv_bn(seq[0], seq[1])
        for m in [m for m in self.modules() if isinstance(m, RepVggBlock)]:   # first: they need their own BatchNorms
            m.fuse()
        for m in [m for m in self.modules() if isinstance(m, ConvBNAct)]:
            m.fuse()
