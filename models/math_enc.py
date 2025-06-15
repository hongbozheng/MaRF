from torch import Tensor
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model
from transformers import BertConfig, BertModel


def precompute_freqs_cis(dim: int, seq_len: int, theta: float):
    # [D/2]
    freqs = 1.0 / (theta ** (
        torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim
    ))
    # [L]
    m = torch.arange(start=0, end=seq_len)
    # [L, D/2]
    freqs = torch.outer(input=m, vec2=freqs)
    # [L, D/2] -> [L, D/2]
    freqs_complex = torch.polar(abs=torch.ones_like(input=freqs), angle=freqs)

    return freqs_complex


def apply_rotary_emb(x: Tensor, freqs_complex: Tensor) -> Tensor:
    # [B, L, H, H_D] -> [B, L, H, H_D/2]
    x_complex = torch.view_as_complex(input=x.reshape(*x.shape[:-1], -1, 2))
    # [L, H_D/2] -> [1, L, 1, H_D/2]
    freqs_complex = freqs_complex.unsqueeze(dim=0).unsqueeze(dim=2)
    # [B, L, H, H_D/2] * [1, L, 1, H_D/2]
    x_rotated = x_complex * freqs_complex
    # [B, L, H, H_D/2] -> [B, L, H, H_D/2, 2]
    x_out = torch.view_as_real(x_rotated)
    # [B, L, H, H_D/2, 2] -> [B, L, H, H_D]
    x_out = x_out.reshape(*x.shape)

    return x.type_as(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            n_kv_heads: Optional[int],
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

    def forward(
            self,
            x: Tensor,
            freqs_complex: Tensor,
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

        # [B, L, H_Q, H_D] -> [B, L, H_Q, H_D]
        q = apply_rotary_emb(x=q, freqs_complex=freqs_complex)
        # [B, L, H_KV, H_D] -> [B, L, H_KV, H_D]
        k = apply_rotary_emb(x=k, freqs_complex=freqs_complex)

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

    def forward(
            self,
            x: Tensor,
            freqs_complex: Tensor,
            attn_mask: Optional[Tensor],
            cache_pos: Optional[Tensor],
    ) -> Tensor:
        # [B, L, D] + [B, L, D] -> [B, L, D]
        h = x + self.attention.forward(
            x=self.attention_norm(x),
            freqs_complex=freqs_complex,
            mask=attn_mask,
            cache_pos=cache_pos,
        )
        out = h + self.feed_forward(self.ffn_norm(h))

        return out


class MathEnc(nn.Module):
    def __init__(
            self,
            tok_embeddings: Optional[nn.Embedding],
            vocab_size: Optional[int],
            dim: int,
            n_layers: int,
            n_heads: int,
            n_kv_heads: Optional[int],
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
            norm_eps: float,
            theta: int,
            max_seq_len: int,
    ) -> None:
        super().__init__()

        if (tok_embeddings is None) == (vocab_size is None):
            raise ValueError(
                "You must provide exactly one of `tok_embeddings` or `vocab_size`"
            )

        if tok_embeddings is not None:
            self.tok_embeddings = tok_embeddings
        else:
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
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    norm_eps=norm_eps,
                )
            )

        self.norm = RMSNorm(dim=dim, eps=norm_eps)

        self.freqs_complex = precompute_freqs_cis(
            dim=dim // n_heads, seq_len=max_seq_len, theta=theta
        )

    def forward(
            self,
            input_ids: Tensor,
            attn_mask: Tensor,
            cache_pos: Optional[Tensor],
    ) -> Tensor:
        # [B, L] -> [B, L, D]
        h = self.tok_embeddings(input_ids)

        freqs_complex = self.freqs_complex[:input_ids.shape[1]] \
            .to(device=input_ids.device)

        for layer in self.layers:
            h = layer(
                x=h,
                freqs_complex=freqs_complex,
                attn_mask=attn_mask,
                cache_pos=cache_pos,
            )
        h = self.norm(h)

        return h


@register_model(name="math_enc")
def build_model(cfg) -> nn.Module:
    if cfg.MODEL.MATH_ENC.TOK_EMBEDDINGS == "bert":
        bert_cfg = BertConfig.from_json_file(json_file=cfg.CKPT.BERT.CFG)
        bert = BertModel(
            config=bert_cfg, add_pooling_layer=cfg.MODEL.BERT.ADD_POOLING
        )

        return MathEnc(
            tok_embeddings=bert.get_input_embeddings(),
            vocab_size=cfg.MODEL.MATH_ENC.VOCAB_SIZE,
            dim=cfg.MODEL.MATH_ENC.DIM,
            n_layers=cfg.MODEL.MATH_ENC.N_LAYERS,
            n_heads=cfg.MODEL.MATH_ENC.N_HEADS,
            n_kv_heads=cfg.MODEL.MATH_ENC.N_KV_HEADS,
            multiple_of=cfg.MODEL.MATH_ENC.MULTIPLE_OF,
            ffn_dim_multiplier=cfg.MODEL.MATH_ENC.FFN_DIM_MULTIPLIER,
            norm_eps=cfg.MODEL.MATH_ENC.NORM_EPS,
            theta=cfg.MODEL.MATH_ENC.THETA,
            max_seq_len=cfg.MODEL.MATH_ENC.MAX_SEQ_LEN,
        )
    else:
        return MathEnc(
            tok_embeddings=cfg.MODEL.MATH_ENC.TOK_EMBEDDINGS,
            vocab_size=cfg.MODEL.MATH_ENC.VOCAB_SIZE,
            dim=cfg.MODEL.MATH_ENC.DIM,
            n_layers=cfg.MODEL.MATH_ENC.N_LAYERS,
            n_heads=cfg.MODEL.MATH_ENC.N_HEADS,
            n_kv_heads=cfg.MODEL.MATH_ENC.N_KV_HEADS,
            multiple_of=cfg.MODEL.MATH_ENC.MULTIPLE_OF,
            ffn_dim_multiplier=cfg.MODEL.MATH_ENC.FFN_DIM_MULTIPLIER,
            norm_eps=cfg.MODEL.MATH_ENC.NORM_EPS,
            theta=cfg.MODEL.MATH_ENC.THETA,
            max_seq_len=cfg.MODEL.MATH_ENC.MAX_SEQ_LEN,
        )
