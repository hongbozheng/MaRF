from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(self, temperature: float, reduction: str) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
            self,
            query: Tensor,
            pos_key: Tensor,
            neg_key: Tensor,
    ) -> Tensor:
        query = F.normalize(input=query, p=2.0, dim=-1, eps=1e-12)
        pos_key = F.normalize(input=pos_key, p=2.0, dim=-1, eps=1e-12)
        neg_key = F.normalize(input=neg_key, p=2.0, dim=-1, eps=1e-12)

        pos_logit = torch.sum(query*pos_key, dim=1, keepdim=True)

        query = query.unsqueeze(dim=1)
        neg_logit = query @ neg_key.transpose(dim0=-2, dim1=-1)
        neg_logit = neg_logit.squeeze(dim=1)

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
