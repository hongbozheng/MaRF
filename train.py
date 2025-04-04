import logger
import os
import torch
import torch.nn as nn
import torch.optim as optim
from avg_meter import AverageMeter
from logger import timestamp
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        max_norm: float,
        device: torch.device,
        dataloader: DataLoader,
) -> float:
    model.train(mode=True)

    loader_tqdm = tqdm(iterable=dataloader, position=1, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)
    # print(optimizer.param_groups[0]["lr"])

    loss_meter = AverageMeter()

    for i, batch in enumerate(iterable=loader_tqdm):
        src = batch["src"].to(device=device)
        src_mask = batch["src_mask"].to(device=device)

        optimizer.zero_grad()
        embs = model(tokens=src, mask=src_mask, input_pos=None)
        query = embs[:, 0]
        pos_key = embs[:, 1]
        neg_key = embs[:, 2:]
        loss = criterion(query=query, pos_key=pos_key, neg_key=neg_key)
        # print(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        loss_meter.update(loss.item(), n=src.size(dim=0))

        loader_tqdm.set_description(
            desc=f"[{timestamp()}] [Batch {i+1}]: "
                 f"train loss {loss_meter.avg:.6f}",
            refresh=True,
        )

    return loss_meter.avg


def train_model(
        model: nn.Module,
        ckpt_best: str,
        ckpt_last: str,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        max_norm: float,
        device: torch.device,
        n_epochs: int,
        dataloader: DataLoader,
):
    model.to(device=device)
    model.train(mode=True)

    path, _ = os.path.split(p=ckpt_last)
    if not os.path.exists(path=path):
        os.makedirs(name=path, exist_ok=True)

    init_epoch = 0
    best_loss = float('inf')
    avg_losses = []  # TODO: Store train losses & val acc in JSON?

    if os.path.exists(path=ckpt_last):
        ckpt = torch.load(f=ckpt_last, map_location=device)
        model.load_state_dict(state_dict=ckpt["model_state"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state"])
        lr_scheduler.load_state_dict(state_dict=ckpt["lr_scheduler_state"])
        init_epoch = ckpt["epoch"]+1
        best_loss = ckpt["best_loss"]
        filename = os.path.basename(p=ckpt_last)
        logger.log_info(f"Loaded `{filename}`.")

    epoch_tqdm = tqdm(
        iterable=range(init_epoch, n_epochs),
        desc=f"[{timestamp()}] [Epoch {init_epoch}]",
        position=0,
        leave=True,
    )

    for epoch in epoch_tqdm:
        epoch_tqdm.set_description(
            desc=f"[{timestamp()}] [Epoch {epoch}]",
            refresh=True,
        )
        avg_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            max_norm=max_norm,
            device=device,
            dataloader=dataloader,
        )

        lr_scheduler.step()

        epoch_tqdm.write(
            s=f"[{timestamp()}] [Epoch {epoch}]: loss {avg_loss:.6f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                obj={
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                },
                f=ckpt_best,
            )
            epoch_tqdm.write(
                s=f"[{timestamp()}] [Epoch {epoch}]: Saved best model to "
                  f"`{ckpt_best}`"
            )

        torch.save(
            obj={
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
            },
            f=ckpt_last,
        )

    return
