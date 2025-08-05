from typing import Dict, List
from torch import Tensor

import string
import torch
from .registry import register_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizer


class ARQMath(Dataset):
    def __init__(
            self,
            file_path: str,
            tokenizer: BertTokenizer,
            max_seq_len: int,
    ) -> None:
        super().__init__()
        self.posts = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        file = open(file=file_path, mode='r', encoding='utf-8')
        for line in file:
            post = line.strip().split(sep='\t')
            post[0] = f"[unused0] {post[0]}"
            post[1] = f"[unused1] {post[1]}"
            post[2] = f"[unused1] {post[2]}"
            self.posts.append(post)
        file.close()

    def __len__(self) -> int:
        return len(self.posts)

    def __getitem__(self, idx: int) -> List[str]:
        return self.posts[idx]

    def collate_fn(
            self,
            batch: List[List[Tensor]],
    ) -> Dict[str, Tensor]:
        posts = [post for item in batch for post in item]
        batch_enc = self.tokenizer(
            text=posts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        punct_ids = []
        for p in string.punctuation:
            punct_id = self.tokenizer(text=p, add_special_tokens=False)
            punct_ids.append(punct_id['input_ids'][0])

        input_ids = batch_enc['input_ids']
        punct_ids = torch.tensor(punct_ids, device=input_ids.device)
        punct_mask = torch.isin(input_ids, punct_ids)
        punct_mask = (~punct_mask).to(dtype=torch.int64)
        batch_enc['punct_mask'] = punct_mask

        return batch_enc


@register_dataset(name="arqmath")
def build_dataset(cfg) -> Dataset:
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.CKPT.BERT.TOKENIZER
    )
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[unused0]', '[unused1]']
    })

    return ARQMath(
        file_path=cfg.DATA.ARQMATH,
        tokenizer=tokenizer,
        max_seq_len=cfg.MODEL.MATH_ENC.MAX_SEQ_LEN,
    )
