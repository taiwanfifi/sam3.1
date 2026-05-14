# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
# Modified: SDPA fallback for GPUs without Flash Attention 3 (e.g. RTX 5090 Blackwell)

import torch
import torch.nn.functional as F


def flash_attn_func(q, k, v):
    """
    Drop-in replacement for FA3 using PyTorch scaled_dot_product_attention.

    Input:  q, k, v — (batch, seq_len, num_heads, head_dim)
    Output: (batch, seq_len, num_heads, head_dim)
    """
    orig_dtype = q.dtype
    # SDPA expects (batch, num_heads, seq_len, head_dim)
    q = q.transpose(1, 2).contiguous().to(torch.bfloat16)
    k = k.transpose(1, 2).contiguous().to(torch.bfloat16)
    v = v.transpose(1, 2).contiguous().to(torch.bfloat16)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).contiguous().to(orig_dtype)
