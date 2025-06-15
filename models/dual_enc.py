from torch import Tensor
from typing import Optional, Tuple

import torch.nn as nn
from .math_enc import MathEnc
from transformers.models.bert.modeling_bert import BertPreTrainedModel


class DualEnc(nn.Module):
    def __init__(
            self,
            bert: BertPreTrainedModel,
            dim: int,
            n_layers: int,
            n_heads: int,
            n_kv_heads: Optional[int],
            multiple_of: int,
            norm_eps: float,
            theta: int,
            max_seq_len: int,
    ) -> None:
        super().__init__()

        self.bert = bert
        self.math_enc = MathEnc(
            word_embeddings=bert.get_input_embeddings(),
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            multiple_of=multiple_of,
            norm_eps=norm_eps,
            theta=theta,
            max_seq_len=max_seq_len,
        )

    def forward(
            self,
            qa: Tensor,
            qa_mask: Tensor,
            math: Tensor,
            math_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        qa_embs = self.bert(input_ids=qa, attention_mask=qa_mask)
        math_embs = self.math_enc(tokens=math, mask=math_mask)

        return qa_embs, math_embs
