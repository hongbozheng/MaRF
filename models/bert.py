from torch import Tensor

import torch.nn as nn
from .registry import register_model
from transformers import BertConfig, BertModel


class Bert(BertModel):
    def __init__(
            self,
            config: BertConfig,
            add_pooling_layer: bool,
            reduce_dim: bool,
            dim: int,
    ) -> None:
        super().__init__(config=config, add_pooling_layer=add_pooling_layer)
        self.reduce_dim = reduce_dim
        if self.reduce_dim:
            self.linear = nn.Linear(
                in_features=config.hidden_size,
                out_features=dim,
                bias=False,
            )

    def forward(self, token_ids: Tensor, attn_mask: Tensor) -> Tensor:
        x = super().forward(
            input_ids=token_ids, attention_mask=attn_mask
        ).last_hidden_state
        if self.reduce_dim:
            x = self.linear(x)

        return x


@register_model(name="bert")
def build_model(cfg) -> BertModel:
    config = BertConfig.from_json_file(json_file=cfg.CKPT.BERT.CFG)

    return Bert(
        config=config,
        add_pooling_layer=cfg.MODEL.BERT.ADD_POOLING,
        reduce_dim=cfg.MODEL.BERT.REDUCE_DIM,
        dim=cfg.MODEL.BERT.DIM,
    )
