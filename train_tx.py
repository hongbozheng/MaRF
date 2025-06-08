#!/usr/bin/env python3


from config import get_config, DEVICE
from criterion import build_criterion
from dataset import ARQMath
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from train import train_model
from transformer import Transformer


def main() -> None:
    cfg = get_config(args=None)

    tokenizer = Tokenizer(file_path=cfg.DATA.VOCAB_FILE)

    arqmath = ARQMath(
        file_path=cfg.DATA.FORMULA_FILE,
        tokenizer=tokenizer,
        val=False,
    )
    dataloader = DataLoader(
        dataset=arqmath,
        batch_size=cfg.LOADER.TRAIN.BATCH_SIZE,
        shuffle=cfg.LOADER.TRAIN.SHUFFLE,
        num_workers=cfg.LOADER.TRAIN.NUM_WORKERS,
        collate_fn=arqmath.collate_fn,
        pin_memory=cfg.LOADER.TRAIN.PIN_MEMORY,
    )
    # for batch in dataloader:
    #     src = batch["src"]
    #     print(src.shape)
    # return

    math_enc = Transformer(
        vocab_size=len(tokenizer.vocabs),
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

    # define optimizer
    optimizer = build_optimizer(cfg=cfg, model=math_enc)

    # define lr scheduler
    lr_scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)

    # define criterion
    criterion = build_criterion(cfg=cfg)

    train_model(
        model=math_enc,
        ckpt_best=cfg.CKPT.BEST,
        ckpt_last=cfg.CKPT.LAST,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        max_norm=cfg.TRAIN.MAX_NORM,
        device=DEVICE,
        n_epochs=cfg.TRAIN.N_EPOCHS,
        dataloader=dataloader,
        save_every_n_iters=cfg.TRAIN.SAVE_N_ITERS,
    )

    return


if __name__ == '__main__':
    main()
