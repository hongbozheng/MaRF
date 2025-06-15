#!/usr/bin/env python3


import argparse
from config import get_config, DEVICE
from criterion import build_criterion
from datasets.registry import build_dataset
from lr_scheduler import build_scheduler
from models.registry import build_model
from optimizer import build_optimizer
from torch.utils.data import DataLoader
from train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(prog='Math IR')
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        metavar="FILE",
        help='path to config file',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        metavar="FILE",
        help='path to dataset config file',
    )
    args, unparsed = parser.parse_known_args()
    cfg = get_config(args=args)

    # dataset
    dataset = build_dataset(cfg=cfg)

    # dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.LOADER.TRAIN.BATCH_SIZE,
        shuffle=cfg.LOADER.TRAIN.SHUFFLE,
        num_workers=cfg.LOADER.TRAIN.NUM_WORKERS,
        collate_fn=dataset.collate_fn,
        pin_memory=cfg.LOADER.TRAIN.PIN_MEMORY,
    )
    # for batch in dataloader:
    #     input_ids = batch["input_ids"]
    #     decoded = dataset.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    #     print(input_ids.shape)
    #     print(decoded)
    #     print(batch["attention_mask"])
    # return

    # model
    model = build_model(cfg=cfg)
    print(model)

    # optimizer
    optimizer = build_optimizer(cfg=cfg, model=model)

    # lr scheduler
    lr_scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)

    # criterion
    criterion = build_criterion(cfg=cfg)

    train_model(
        model=model,
        ckpt_last=cfg.CKPT.LAST,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        postprocess=cfg.POSTPROCESS.NAME,
        n_exprs=cfg.DATA.N_EXPRS,
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
