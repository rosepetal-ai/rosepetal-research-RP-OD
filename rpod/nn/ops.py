# RP-OD - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Small autograd helpers."""
import torch


class _ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.contiguous()


def contiguous_grad(x):
    """Identity whose backward hands a contiguous gradient to the layers before it.

    Feature maps are flattened and transposed into tokens (attention, decoder memory), so their gradient comes back
    transposed. With a single image per step, PyTorch (seen on 2.8, CPU) mistakes that layout for channels_last and
    BatchNorm's backward returns a wrong input gradient. Making it contiguous avoids it. No effect without autograd.
    """
    return _ContiguousGrad.apply(x) if torch.is_grad_enabled() and x.requires_grad else x
