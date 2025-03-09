# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import HuggingfaceTokenizer

__all__ = [
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
]

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

def fp16_clamp(x):
    """Clamp values to avoid overflow in fp16 precision."""
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


def init_weights(m):
    """Initialize weights for different layers."""
    if isinstance(m, T5LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5Model):
        nn.init.normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn)**-0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn)**-0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(
            m.embedding.weight, std=(2 * m.num_buckets * m.num_heads)**-0.5)


class GELU(nn.Module):
    """GELU activation function."""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    """Layer normalization for T5 model."""
    def __init__(self, dim, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    """Multi-head attention mechanism for T5."""
    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, n,