from torch import Tensor
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: torch.device = None


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


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.size()
    if n_rep == 1:
        return x
    else:
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: int) -> None:
        super().__init__()
        # [H_D/2]
        theta = 1.0 / (
            base ** (torch.arange(
                        start=0, end=head_dim, step=2
                    )[:(head_dim//2)].float() / head_dim)
        )
        self.register_buffer(name="theta", tensor=theta, persistent=False)
        self.build_rope_cache(max_seq_len=max_seq_len)

        return

    def build_rope_cache(self, max_seq_len: int) -> None:
        # [S_max]
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        # [S_max, H_D/2, 2]
        cache = torch.stack(
            tensors=[torch.cos(input=idx_theta), torch.sin(input=idx_theta)],
            dim=-1,
        )
        self.register_buffer(name="cache", tensor=cache, persistent=False)

        return

    def forward(self, x: Tensor, input_pos: Optional[Tensor]) -> Tensor:
        seq_len = x.size(dim=1)
        # [S, H_D/2, 2]
        rope_cache = self.cache[:seq_len] if input_pos is None \
            else self.cache[input_pos]
        # [B, S, H, H_D] -> [B, S, H, H_D/2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        # [S, H_D/2, 2] -> [1, S, 1, H_D/2, 2]
        rope_cache = rope_cache.view(
            -1, xshaped.size(dim=1), 1, xshaped.size(dim=3), 2
        )
        # [B, S, H, H_D/2, 2]
        x_out = torch.stack(
            tensors=[
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        # [B, S, H, H_D/2, 2] -> [B, S, H, H_D]
        x_out = x_out.flatten(start_dim=3)

        return x_out.type_as(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            n_kv_heads: Optional[int],
    ) -> None:
        super().__init__()

        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_q_heads = n_heads
        self.n_rep = self.n_q_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(
            in_features=dim,
            out_features=n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            in_features=dim,
            out_features=self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            in_features=dim,
            out_features=self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            in_features=n_heads * self.head_dim,
            out_features=dim,
            bias=False,
        )

        # self.cache_k = torch.zeros(
        #     size=(
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_kv_heads,
        #         self.head_dim,
        #     ),
        #     device=self.device,
        # )
        # self.cache_v = torch.zeros(
        #     size=(
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_kv_heads,
        #         self.head_dim,
        #     ),
        #     device=self.device,
        # )

        return

    def forward(
            self,
            x: Tensor,
            freqs_complex: Tensor,
            mask: Optional[Tensor],
    ) -> Tensor:
        # [B, S, D]
        batch_size, seq_len, _ = x.shape
        # [B, S, D] -> [B, S,  H_Q * D_H]
        q = self.wq(x)
        # [B, S, D] -> [B, S, H_KV * D_H]
        k = self.wk(x)
        # [B, S, D] -> [B, S, H_KV * D_H]
        v = self.wv(x)

        # [B, S,  H_Q * D_H] -> [B, S,  H_Q, D_H]
        q = q.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        # [B, S, H_KV * D_H] -> [B, S, H_KV, D_H]
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # [B, S, H_KV * D_H] -> [B, S, H_KV, D_H]
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rotary_embeddings(
            x=q, freqs_complex=freqs_complex, device=x.device
        )
        k = apply_rotary_embeddings(
            x=k, freqs_complex=freqs_complex, device=x.device
        )

        # [B, S, H_Q, D_H] -> [B, H_Q, S, D_H]
        q = q.transpose(dim0=1, dim1=2)
        # [B, S_KV, H_Q, D_H] -> [B, H_Q, S_KV, D_H]
        k = k.transpose(dim0=1, dim1=2)
        # [B, S_KV, H_Q, D_H] -> [B, H_Q, S_KV, D_H]
        v = v.transpose(dim0=1, dim1=2)

        # [B, H_Q, S, D_H] @ [B, H_Q, D_H, S_KV] -> [B, H_Q, S, S_KV]
        scores = q @ k.transpose(dim0=-2, dim1=-1) / math.sqrt(self.H_dim)
        if mask is not None:
            # TODO: mask should be a bool tensor
            scores.masked_fill_(mask=mask, value=float('-inf'))
        scores = F.softmax(input=scores.float(), dim=-1).type_as(q)

        # [B, H_Q, S, S_KV] @ [B, H_Q, S_KV, D_H] -> [B, H_Q, S, D_H]
        output = scores @ v

        # [B, H_Q, S, D_H] -> [B, S, H_Q, D_H] -> [B, S, D]
        output = output.transpose(dim0=1, dim1=2).contiguous()\
            .view(batch_size, seq_len, -1)

        # [B, S, D] -> [B, S, D]
        output = self.wo(output)

        return output


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of \
            * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            in_features=dim, out_features=hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            in_features=hidden_dim, out_features=dim, bias=False
        )
        self.w3 = nn.Linear(
            in_features=dim, out_features=hidden_dim, bias=False
        )

        return

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        return

    def _norm(self, x: Tensor) -> Tensor:
        # [B, S, D] * [B, S, 1] = [B, S, D]
        return x * torch.rsqrt(
            input=x.pow(exponent=2).mean(dim=-1, keepdim=True) + self.eps,
        )

    def forward(self, x: Tensor) -> Tensor:
        # [D] * [B, S, D] = [B, S, D]
        return self.weight * self._norm(x=x)


class EncoderBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            n_kv_heads: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
            norm_eps: float,
    ) -> None:
        super().__init__()

        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self.attention_norm = RMSNorm(dim=dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim=dim, eps=norm_eps)

        return

    def forward(
            self,
            x: Tensor,
            freqs_complex: Tensor,
            mask: Optional[Tensor],
    ) -> Tensor:
        # [B, S, D] + [B, S, D] -> [B, S, D]
        h = x + self.attention.forward(
            x=self.attention_norm(x), freqs_complex=freqs_complex, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            n_layers: int,
            n_heads: int,
            n_kv_heads: Optional[int],
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
            norm_eps: float,
            max_seq_len: int,
            rope_theta: float,
            device: torch.device,
    ) -> None:
        super().__init__()

        assert vocab_size != -1, "Vocabulary size must be set!"

        self.tok_embeddings = nn.Embedding(
            num_embeddings=vocab_size,embedding_dim=dim
        )

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                EncoderBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    norm_eps=norm_eps,
                )
            )

        self.norm = RMSNorm(dim=dim, eps=norm_eps)
        self.output = nn.Linear(
            in_features=dim, out_features=vocab_size, bias=False
        )

        self.freqs_complex = precompute_theta_pos_frequencies(
            head_dim=dim // n_heads,
            seq_len=max_seq_len * 2,
            theta=rope_theta,
            device=device,
        )

        return

    def forward(self, tokens: Tensor, mask: Tensor):
        # [B, S] -> [B, S, D]
        h = self.tok_embeddings(tokens)

        # retrieve the pairs [m, theta] corresponding to the positions
        # [start_pos, start_pos + S]
        # freqs_complex = self.freqs_complex[start_pos:start_pos + 1]

        for layer in self.layers:
            h = layer(h, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()

        return output
