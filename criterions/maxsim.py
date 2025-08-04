from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.registry import register_criterion


class MaxSim(nn.Module):
    def __init__(self, temperature: float, reduction: str) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
            self,
            query: Tensor,
            pos_key: Tensor,
            neg_key: Tensor,
            query_mask: Tensor,
            pos_mask: Tensor,
            neg_mask: Tensor,
    ) -> Tensor:
        # [B, L, D] -> [B, L, D]
        query = F.normalize(input=query, p=2.0, dim=-1, eps=1e-12)
        # [B, L, D] -> [B, L, D]
        pos_key = F.normalize(input=pos_key, p=2.0, dim=-1, eps=1e-12)
        # [B, NG, L, D] -> [B, NG, L, D]
        neg_key = F.normalize(input=neg_key, p=2.0, dim=-1, eps=1e-12)

        # [B. L, D] @ [B, D, L] -> [B, L, L]
        pos_logit = query @ pos_key.transpose(dim0=-2, dim1=-1)
        # [B, L] -> [B, 1, L]
        pos_mask = pos_mask.unsqueeze(dim=-2).to(dtype=pos_logit.dtype)
        # pos_mask = (1.0 - pos_mask) * -10000.0
        # [B, L, L] + [B, 1, L] -> [B, L, L]
        pos_logit = pos_logit * pos_mask
        # [B, L, L] -> [B, L]
        pos_logit = torch.max(input=pos_logit, dim=-1).values
        # [B, L] * [B, L] -> [B, 1]
        # pos_logit = (pos_logit * query_mask.to(dtype=pos_logit.dtype)) \
        #     .sum(dim=-1, keepdim=True)
        pos_logit = pos_logit.sum(dim=-1, keepdim=True)

        # [B, L, D] -> [B, 1, L, D]
        query = query.unsqueeze(dim=1)
        # [B, 1, L, D] @ [B, NG, D, L] -> [B, NG, L, L]
        neg_logit = query @ neg_key.transpose(dim0=-2, dim1=-1)
        # [B, NG, L] -> [B, NG, 1, L]
        neg_mask = neg_mask.unsqueeze(dim=-2).to(dtype=neg_logit.dtype)
        # neg_mask = (1.0 - neg_mask) * -10000.0
        # [B, NG, L, L] + [B, NG, 1, L] -> [B, NG, L, L]
        neg_logit = neg_logit * neg_mask
        # [B, NG, L, L] -> [B, NG, L]
        neg_logit = torch.max(input=neg_logit, dim=-1).values
        # [B, L] -> [B, 1, L]
        query_mask = query_mask.unsqueeze(dim=1)
        # [B, NG, L] * [B, 1, L] -> [B, NG]
        # neg_logit = (neg_logit * query_mask.to(dtype=neg_logit.dtype)) \
        #     .sum(dim=-1, keepdim=False)
        neg_logit = neg_logit.sum(dim=-1, keepdim=False)

        # [B, 1] [B, NG] -> [B, 1 + NG]
        logits = torch.cat(tensors=[pos_logit, neg_logit], dim=1)
        labels = torch.zeros(
            logits.size(dim=0),
            dtype=torch.int64,
            device=query.device,
        )
        loss = F.cross_entropy(
            input=logits/self.temperature,
            target=labels,
            reduction=self.reduction,
        )

        return loss


@register_criterion(name="maxsim")
def build_model(cfg) -> nn.Module:
    return MaxSim(
        temperature=cfg.CRITERION.MAXSIM.TEMPERATURE,
        reduction=cfg.CRITERION.MAXSIM.REDUCTION,
    )
