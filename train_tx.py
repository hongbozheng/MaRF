#!/usr/bin/env python3


import torch

from config import get_config
from transformer import Transformer


def main() -> None:
    cfg = get_config(args=None)

    math_enc = Transformer(
        vocab_size=cfg.MODEL.TX.VOCAB_SIZE,
        dim=cfg.MODEL.TX.DIM,
        n_layers=cfg.MODEL.TX.N_LAYERS,
        n_heads=cfg.MODEL.TX.N_HEADS,
        n_kv_heads=cfg.MODEL.TX.N_KV_HEADS,
        base=cfg.MODEL.TX.BASE,
        max_seq_len=cfg.MODEL.TX.MAX_SEQ_LEN,
        multiple_of=cfg.MODEL.TX.MULTIPLE_OF,
        ffn_dim_multiplier=cfg.MODEL.TX.FFN_DIM_MULTIPLIER,
        norm_eps=cfg.MODEL.TX.NORM_EPS,
    )

    # temporary forward test
    input = torch.randint(low=0, high=100, size=(4, 64), dtype=torch.int64)
    mask = torch.triu(input=torch.ones(64, 64, dtype=torch.bool), diagonal=1)

    output = math_enc(tokens=input, mask=mask, input_pos=None)
    print(output.size())

    return


if __name__ == '__main__':
    main()
