from typing import Dict, List
from torch import Tensor

from .registry import register_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer


class ContrastiveExpr(Dataset):
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
            expr[0] = f"[unused0] {expr[0]}"
            expr[1] = f"[unused1] {expr[1]}"
            expr[2] = f"[unused1] {expr[2]}"
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
        batch_enc = self.tokenizer(
            text=exprs,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        return batch_enc


@register_dataset(name="contrastive_expr")
def build_dataset(cfg) -> Dataset:
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.CKPT.BERT.TOKENIZER
    )
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[unused0]', '[unused1]']
    })

    return ContrastiveExpr(
        file_path=cfg.DATA.MATH,
        tokenizer=tokenizer,
        max_seq_len=cfg.MODEL.MATH_ENC.MAX_SEQ_LEN,
    )
