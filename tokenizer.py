from torch import Tensor
from typing import List

import torch


class Tokenizer:
    def __init__(self) -> None:
        self.soe = "SOE"
        self.eoe = "EOE"
        self.pad = "PAD"

        self.vocabs = [self.pad, self.soe]
        # TODO: parse all tokens and add to vocabs list
        self.vocabs.append(self.eoe)  # put end-of-expr in the end

        self.word2idx = {w: i for i, w in enumerate(self.vocabs)}
        self.idx2word = {i: w for i, w in enumerate(self.vocabs)}

        return

    def encode(self, expr: List[str]) -> Tensor:
        tokens = []

        for word in self.vocabs:
            tokens.append(self.word2idx[word])

        tokens = torch.cat(
            tensors=(
                torch.tensor(data=[self.word2idx["SOE"]], dtype=torch.int64),
                torch.tensor(data=tokens, dtype=torch.int64),
                torch.tensor(data=[self.word2idx["EOE"]], dtype=torch.int64),
            ),
            dim=0,
        )

        return tokens

    def decode(self, tokens: Tensor) -> str:
        expr = []

        for token in tokens:
            expr.append(self.idx2word[token])
            if token == self.word2idx[self.eoe]:
                break

        expr = " ".join(expr)

        return expr
