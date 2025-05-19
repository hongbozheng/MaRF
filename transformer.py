from torch import Tensor
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (
            torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim
        ))
        self.register_buffer(name="inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, pos_ids: Tensor) -> Tensor:
        # [D/2] -> [1, D/2, 1] -> [B, D/2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(pos_ids.shape[0], -1, 1)
        # [B, L] -> [B, 1, L]
        pos_ids_expanded = pos_ids[:, None, :].float()
        # [B, D/2, 1] @ [B, 1, L] -> [B, D/2, L] -> [B, L, D/2]
        freqs = (inv_freq_expanded @ pos_ids_expanded).transpose(dim0=1, dim1=2)
        # [B, L, D/2]
        freqs_cis = torch.polar(abs=torch.ones_like(input=freqs), angle=freqs)

        return freqs_cis


def apply_rotary_emb(
        q: Tensor,
        k: Tensor,
        freq_cis: Tensor,
) -> tuple[Tensor, Tensor]:
    # [B, L, H, H_D] -> [B, L, H, H_D/2, 2]
    q_ = torch.view_as_complex(input=q.float().reshape(*q.shape[:-1], -1, 2))
    # [B, L, H, H_D] -> [B, L, H, H_D/2, 2]
    k_ = torch.view_as_complex(input=k.float().reshape(*k.shape[:-1], -1, 2))
    # [B, L, H, H_D/2, 2] *  [B, L, H, H_D/2, 2] -> [B, L, H, H_D/2, 2]
    # [B, L, H, H_D/2, 2] -> [B, L, H, H_D/2, 2] -> [B, L, D]
    q_out = torch.view_as_real(input=q_ * freq_cis[:, :, None, :]).flatten(start_dim=3)
    # [B, L, H, H_D/2, 2] *  [B, L, H, H_D/2, 2] -> [B, L, H, H_D/2, 2]
    # [B, L, H, H_D/2, 2] -> [B, L, H, H_D]
    k_out = torch.view_as_real(input=k_ * freq_cis[:, :, None, :]).flatten(start_dim=3)

    return q_out.type_as(q), k_out.type_as(k)


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            n_kv_heads: Optional[int],
            base: int,
            max_seq_len: int,
    ) -> None:
        super().__init__()

        self.n_q_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads

        self.wq = nn.Linear(
            in_features=dim,
            out_features=self.n_q_heads * self.head_dim,
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
            in_features=self.n_q_heads * self.head_dim,
            out_features=dim,
            bias=False,
        )

        # self.rope = RotaryPositionalEmbeddings(
        #     head_dim=self.head_dim,
        #     base=base,
        #     max_seq_len=max_seq_len,
        # )

    def forward(
            self,
            x: Tensor,
            pos_emb: Tensor,
            mask: Optional[Tensor],
            cache_pos: Optional[Tensor],
    ) -> Tensor:
        # [B, L, D]
        batch_size, seq_len, _ = x.shape
        # [B, L, D] -> [B, L, H_Q * D_H]
        q = self.wq(x)
        # [B, L, D] -> [B, L, H_KV * D_H]
        k = self.wk(x)
        # [B, L, D] -> [B, L, H_KV * D_H]
        v = self.wv(x)

        # [B, L, H_Q * D_H] -> [B, L, H_Q, D_H]
        q = q.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        # [B, L, H_KV * D_H] -> [B, L, H_KV, D_H]
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # [B, L, H_KV * D_H] -> [B, L, H_KV, D_H]
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q=q, k=k, freq_cis=pos_emb)

        # # [B, L, H_Q * D_H] -> [B, L, H_Q, D_H]
        # q = self.rope(x=q, input_pos=input_pos)
        # # [B, L, H_KV * D_H] -> [B, L, H_KV, D_H]
        # k = self.rope(x=k, input_pos=input_pos)

        # [B, L, H_Q, D_H] -> [B, H_Q, L, D_H]
        q = q.transpose(dim0=1, dim1=2)
        # [B, L, H_KV, D_H] -> [B, H_KV, L, D_H]
        k = k.transpose(dim0=1, dim1=2)
        # [B, L, H_KV, D_H] -> [B, H_KV, L, D_H]
        v = v.transpose(dim0=1, dim1=2)

        # [B, H_Q, L, D_H] @ [B, H_KV, D_H, L] -> [B, H_Q, L, L]
        scores = q @ k.transpose(dim0=-2, dim1=-1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores.masked_fill(mask=mask, value=float('-inf'))
        scores = F.softmax(input=scores.float(), dim=-1).type_as(q)

        # [B, H_Q, L, L] @ [B, H_KV, L, D_H] -> [B, H_Q, L, D_H]
        output = scores @ v

        # [B, H_Q, L, D_H] -> [B, L, H_Q, D_H] -> [B, L, D]
        output = output.transpose(dim0=1, dim1=2).contiguous()\
            .view(batch_size, seq_len, -1)

        # [B, L, D] -> [B, L, D]
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

        self.gate_proj = nn.Linear(
            in_features=dim, out_features=hidden_dim, bias=False
        )
        self.up_proj = nn.Linear(
            in_features=dim, out_features=hidden_dim, bias=False
        )
        self.down_proj = nn.Linear(
            in_features=hidden_dim, out_features=dim, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        # [B, L, D] * [B, L, 1] = [B, L, D]
        return x * torch.rsqrt(
            input=x.pow(exponent=2).mean(dim=-1, keepdim=True) + self.eps,
        )

    def forward(self, x: Tensor) -> Tensor:
        # [D] * [B, L, D] = [B, L, D]
        return self.weight * self._norm(x=x)


class EncoderBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            n_kv_heads: int,
            base: int,
            max_seq_len: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
            norm_eps: float,
    ) -> None:
        super().__init__()

        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            base=base,
            max_seq_len=max_seq_len,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self.attention_norm = RMSNorm(dim=dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim=dim, eps=norm_eps)

    def forward(
            self,
            x: Tensor,
            pos_emb: Tensor,
            mask: Optional[Tensor],
            cache_pos: Optional[Tensor],
    ) -> Tensor:
        # [B, L, D] + [B, L, D] -> [B, L, D]
        h = x + self.attention.forward(
            x=self.attention_norm(x),
            pos_emb=pos_emb,
            mask=mask,
            cache_pos=cache_pos,
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
            base: int,
            max_seq_len: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
            norm_eps: float,
    ) -> None:
        super().__init__()

        assert vocab_size != -1, "Vocabulary size must be set!"

        self.tok_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim
        )

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                EncoderBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    base=base,
                    max_seq_len=max_seq_len,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    norm_eps=norm_eps,
                )
            )

        self.norm = RMSNorm(dim=dim, eps=norm_eps)
        self.rotary_emb = RotaryEmbedding(
            dim=dim // n_heads, base=base, max_seq_len=max_seq_len
        )
        # TODO: may need to have an additional linear layer to bring the
        # embedding dimension of equation to the same embedding dimension of
        # the query
        # self.output = nn.Linear(
        #     in_features=dim, out_features=vocab_size, bias=False
        # )

    def forward(
            self,
            tokens: Tensor,
            mask: Tensor,
            cache_pos: Optional[Tensor],
    ) -> Tensor:
        # [B, L] -> [B, L, D]
        h = self.tok_embeddings(tokens)

        pos_ids = torch.arange(
            start=0, end=h.shape[1], dtype=h.dtype, device=h.device
        ).unsqueeze(dim=0)
        freq_cis = self.rotary_emb(pos_ids=pos_ids)

        for layer in self.layers:
            h = layer(x=h, mask=mask, pos_emb=freq_cis, cache_pos=cache_pos)
        h = self.norm(h)
        # TODO
        # output = self.output(h).float()

        return h


# class RotaryPositionalEmbeddings(nn.Module):
#     def __init__(self, head_dim: int, base: int, max_seq_len: int) -> None:
#         super().__init__()
#         # [H_D/2]
#         theta = 1.0 / (
#             base ** (torch.arange(
#                         start=0, end=head_dim, step=2
#                     )[:(head_dim//2)].float() / head_dim)
#         )
#         self.register_buffer(name="theta", tensor=theta, persistent=False)
#         self.build_rope_cache(max_seq_len=max_seq_len)

#         return

#     def build_rope_cache(self, max_seq_len: int) -> None:
#         # [S_max]
#         seq_idx = torch.arange(
#             max_seq_len, dtype=self.theta.dtype, device=self.theta.device
#         )
#         idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
#         # [S_max, H_D/2, 2]
#         cache = torch.stack(
#             tensors=[torch.cos(input=idx_theta), torch.sin(input=idx_theta)],
#             dim=-1,
#         )
#         self.register_buffer(name="cache", tensor=cache, persistent=False)

#         return

#     def forward(self, x: Tensor, input_pos: Optional[Tensor]) -> Tensor:
#         seq_len = x.size(dim=1)
#         # [L, H_D/2, 2]
#         rope_cache = self.cache[:seq_len] if input_pos is None \
#             else self.cache[input_pos]
#         # [B, L, H, H_D] -> [B, L, H, H_D/2, 2]
#         xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
#         # [L, H_D/2, 2] -> [1, L, 1, H_D/2, 2]
#         rope_cache = rope_cache.view(
#             -1, xshaped.size(dim=1), 1, xshaped.size(dim=3), 2
#         )
#         # [B, L, H, H_D/2, 2]
#         x_out = torch.stack(
#             tensors=[
#                 xshaped[..., 0] * rope_cache[..., 0]
#                 - xshaped[..., 1] * rope_cache[..., 1],
#                 xshaped[..., 1] * rope_cache[..., 0]
#                 + xshaped[..., 0] * rope_cache[..., 1],
#             ],
#             dim=-1,
#         )
#         # [B, L, H, H_D/2, 2] -> [B, L, H, H_D]
#         x_out = x_out.flatten(start_dim=3)

#         return x_out.type_as(x)
