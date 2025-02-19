#!/usr/bin/env python3

from typing import Optional
import torch
from torch import Tensor

from config import get_config, DEVICE
from transformer import Transformer


def precompute_theta_pos_frequencies(
        head_dim: int,
        seq_len:int,
        device: torch.device,
        theta: float = 10000.0,
) -> Tensor:
    assert head_dim % 2 == 0, "Dimension must be divisible by 2!"
    # theta_i = 10000.0 ^ (-2(i-1)/dim) for i in [1..dim/2]
    # [H_dim / 2]
    theta_numerator = torch.arange(
        start=0, end=head_dim, step=2, dtype=torch.float32
    )
    # [H_dim / 2]
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device=device)
    # [S]
    m = torch.arange(seq_len, device=device)
    # [S] outer* [H_dim / 2] = [S, H_dim / 2]
    freqs = torch.outer(input=m, vec2=theta).float()
    # c = R * e ^ (i * m * theta), where R = 1 as follows:
    freqs_complex = torch.polar(abs=torch.ones_like(input=freqs), angle=freqs)

    return freqs_complex


def apply_rotary_embeddings(x: Tensor, freqs_complex: Tensor, device: str):
    # [B, S, H, H_dim] -> [B, S, H, H_dim / 2]
    x_complex = torch.view_as_complex(
        input=x.float().reshape(*x.shape[:-1], -1, 2)
    )
    # [S, H_dim / 2] -> [1, S, 1, H_dim / 2]
    freqs_complex = freqs_complex.unsqueeze(dim=0).unsqueeze(dim=2)
    # [B, S, H, H_dim / 2] * [B, S, H, H_dim / 2]
    x_rotated = x_complex * freqs_complex
    # [B, S, H, H_dim / 2] -> [B, S, H, H_dim / 2, 2]
    x_out = torch.view_as_real(input=x_rotated)
    # [B, S, H, H_dim / 2] -> [B, S, H, H_dim]
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device=device)

import torch.nn as nn



def main() -> None:
    dim = 16
    n_heads = 4
    max_seq_len = 3
    rope_theta = 10000.0
    # =====================================================
    torch.manual_seed(0)
    freqs_complex = precompute_theta_pos_frequencies(
        head_dim=dim // n_heads,
        seq_len=max_seq_len,
        theta=rope_theta,
        device="cpu",
    )
    # print(freqs_complex)
    x = torch.randn((2, 3, n_heads, dim // n_heads))
    print(x.shape)
    x_ = apply_rotary_embeddings(x, freqs_complex, "cpu")
    print(x_)
    # =====================================================
    rope = RotaryEmbedding(dim // n_heads, max_seq_len, int(rope_theta))
    print(x.shape)
    _x = rope.forward(x, input_pos=None)
    print(_x)
    return
    # =====================================================

    cfg = get_config(args=None)

    math_enc = Transformer(
        vocab_size=cfg.MODEL.TX.VOCAB_SIZE,
        dim=cfg.MODEL.TX.DIM,
        n_layers=cfg.MODEL.TX.N_LAYERS,
        n_heads=cfg.MODEL.TX.N_HEADS,
        n_kv_heads=cfg.MODEL.TX.N_KV_HEADS,
        multiple_of=cfg.MODEL.TX.MULTIPLE_OF,
        ffn_dim_multiplier=cfg.MODEL.TX.FFN_DIM_MULTIPLIER,
        norm_eps=cfg.MODEL.TX.NORM_EPS,
        max_seq_len=cfg.MODEL.TX.MAX_SEQ_LEN,
        rope_theta=cfg.MODEL.TX.ROPE_THETA,
        device=DEVICE,
    )

    input = torch.randn((1, 64))

    output = math_enc(input, 0)
    print(output.size())

    return


if __name__ == '__main__':
    main()