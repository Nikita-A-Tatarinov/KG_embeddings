# train/optim.py
from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


def unique_params(parameters: Iterable[torch.nn.Parameter]):
    # dedupe by id to avoid duplicate-params warnings
    return list({id(p): p for p in parameters}.values())


def build_optimizer(params: Iterable[torch.nn.Parameter], cfg) -> Optimizer:
    p = unique_params(params)
    name = cfg.optim.name.lower()
    lr = cfg.optim.lr
    wd = getattr(cfg.optim, "weight_decay", 0.0)
    betas = tuple(getattr(cfg.optim, "betas", (0.9, 0.999)))
    if name == "adam":
        return torch.optim.Adam(p, lr=lr, weight_decay=wd, betas=betas)
    elif name == "adamw":
        return torch.optim.AdamW(p, lr=lr, weight_decay=wd, betas=betas)
    elif name == "sgd":
        return torch.optim.SGD(p, lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optim.name}")


class LinearWarmupDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            scale = step / float(self.warmup_steps)
        else:
            rem = max(1, self.total_steps - self.warmup_steps)
            scale = max(0.0, 1.0 - (step - self.warmup_steps) / float(rem))
        return [base_lr * scale for base_lr in self.base_lrs]


def build_scheduler(optimizer, cfg, total_steps: int):
    name = cfg.sched.name.lower()
    if name == "none":
        return None
    elif name == "linear":
        warm = int(cfg.sched.warmup_steps)
        tot = int(cfg.sched.total_steps) or int(total_steps)
        return LinearWarmupDecay(optimizer, warm, tot)

    elif name == "exponential":
        gamma = float(getattr(cfg.sched, "gamma", 0.99))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    else:
        raise ValueError(f"Unknown scheduler: {cfg.sched.name}")
