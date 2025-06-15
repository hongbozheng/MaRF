from typing import Any, Dict, List
from torch import Tensor

from .registry import register_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer


class Math(Dataset):
    def __init__(
            self,
            file_path: str,
            tokenizer: BertTokenizer,
            max_seq_len: int,
    ) -> None:
        super().__init__()
        self.exprs = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        file = open(file=file_path, mode='r', encoding='utf-8')
        for line in file:
            expr = line.strip().split(sep='\t')
            self.exprs.append(expr)
        file.close()

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int) -> List[str]:
        return self.exprs[idx]

    def collate_fn(
            self,
            batch: List[List[Tensor]],
    ) -> Dict[str, Tensor]:
        exprs = [expr for item in batch for expr in item]
        tokens = self.tokenizer(
            text=exprs,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        return tokens


@register_dataset(name="math")
def build_dataset(cfg) -> Dataset:
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.CKPT.BERT.TOKENIZER
    )

    return Math(
        file_path=cfg.DATA.MATH,
        tokenizer=tokenizer,
        max_seq_len=cfg.MODEL.MATH_ENC.MAX_SEQ_LEN,
    )
