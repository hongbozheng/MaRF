from torch import Tensor

import torch
from tokenizer import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ARQMath(Dataset):
    def __init__(self, file_path: str, tokenizer: Tokenizer, val: bool) -> None:
        super().__init__()
        self.exprs = []
        self.tokenizer = tokenizer
        self.val = val

        file = open(file=file_path, mode='r', encoding='utf-8')
        for line in file:
            expr = line.strip().split(sep='\t')
            self.exprs.append(expr)
        file.close()

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int) -> dict[str, list[Tensor]]:
        src = self.exprs[idx]
        src_tokens = [self.tokenizer.encode(expr=expr) for expr in src]

        return {"src": src_tokens}

    def collate_fn(
            self,
            batch: list[dict[str, list[Tensor]]],
    ) -> dict[str, Tensor]:
        src = [expr for item in batch for expr in item["src"]]
        src = pad_sequence(
            sequences=src,
            batch_first=True,
            padding_value=self.tokenizer.word2idx["PAD"],
        )
        # https://gmongaras.medium.com/how-do-self-attention-masks-work-72ed9382510f
        # [batch_size, n_heads, 1, seq_len]
        src_mask = torch.eq(input=src, other=self.tokenizer.word2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.bool)

        '''
        print(tgt)
        print(src)
        print("src_mask")
        print(src_mask, src_mask.size())
        print("tgt_mask")
        print(tgt_mask, tgt_mask.size())
        print("tgt_pad_mask")
        print(tgt_pad_mask, tgt_pad_mask.size())
        print(tgt_mask, tgt_mask.size())
        '''

        return {
            "src": src,
            "src_mask": src_mask,
        }
