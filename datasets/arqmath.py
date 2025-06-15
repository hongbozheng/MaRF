from typing import Dict, List
from torch import Tensor

import torch
from .registry import register_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizer


class ARQMath(Dataset):
    def __init__(self, file_path: str, tokenizer: BertTokenizer) -> None:
        super().__init__()
        self.exprs = []
        self.tokenizer = tokenizer

        file = open(file=file_path, mode='r', encoding='utf-8')
        for line in file:
            expr = line.strip().split(sep='\t')
            self.exprs.append(expr)
        file.close()

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int) -> Dict[str, List[Tensor]]:
        expr = self.exprs[idx]
        tokens = [self.tokenizer.encode(expr=e) for e in expr]

        return {"tokens": tokens}

    def collate_fn(
            self,
            batch: List[Dict[str, List[Tensor]]],
    ) -> Dict[str, Tensor]:
        tokens = [token for item in batch for token in item["tokens"]]
        tokens = pad_sequence(
            sequences=tokens,
            batch_first=True,
            padding_value=self.tokenizer.word2idx["PAD"],
        )
        # https://gmongaras.medium.com/how-do-self-attention-masks-work-72ed9382510f
        # [batch_size, n_heads, 1, seq_len]
        attn_mask = torch.eq(input=tokens, other=self.tokenizer.word2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.bool)

        return {
            "tokens": tokens,
            "attn_mask": attn_mask,
        }


@register_dataset(name="arqmath")
def build_dataset(cfg) -> Dataset:
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.CKPT.BERT.TOKENIZER
    )

    return ARQMath(file_path=cfg.DATA.ARQMATH, tokenizer=tokenizer)
