# runner/trainer.py
from __future__ import annotations

import random
from types import SimpleNamespace

import torch

from med.med_wrapper import MEDTrainer
from models import create_model
from runner.checkpoint import save_checkpoint
from runner.logger import CSVLogger, Timer
from runner.optim import build_optimizer, build_scheduler


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(opt: str):
    if opt == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return opt


def build_data(cfg):
    # Source
    use_hf = getattr(cfg.dataset, "source", "files").lower() == "hf"
    hf_name = getattr(cfg.dataset, "hf_name", None)
    hf_rev = getattr(cfg.dataset, "hf_revision", None)

    # Files-based options (safe defaults even if unused)
    root = getattr(cfg.dataset, "root", "./data")
    prefer_ids = bool(getattr(cfg.dataset, "prefer_ids", False))
    filtered_eval = bool(getattr(cfg.dataset, "filtered_eval", True))

    # Loader params
    neg_size = int(getattr(cfg.data, "neg_size", 64))
    train_bs = int(getattr(cfg.data, "train_bs", 1024))
    test_bs = int(getattr(cfg.data, "test_bs", 128))
    num_workers = int(getattr(cfg.data, "num_workers", 4))

    name = cfg.dataset.name.lower()
    if name in ("fb15k-237", "fb15k237"):
        from dataset.fb15k237 import prepare_fb15k237

        return prepare_fb15k237(
            root=root,
            prefer_ids=prefer_ids,
            neg_size=neg_size,
            train_bs=train_bs,
            test_bs=test_bs,
            num_workers=num_workers,
            filtered_eval=filtered_eval,
            use_hf=use_hf,
            hf_name=hf_name,
            hf_revision=hf_rev,
        )
    elif name == "wn18rr":
        from dataset.wn18rr import prepare_wn18rr

        return prepare_wn18rr(
            root=root,
            prefer_ids=prefer_ids,
            neg_size=neg_size,
            train_bs=train_bs,
            test_bs=test_bs,
            num_workers=num_workers,
            filtered_eval=filtered_eval,
            use_hf=use_hf,
            hf_name=hf_name,
            hf_revision=hf_rev,
        )
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset.name}")


@torch.no_grad()
def evaluate(model, loaders) -> dict[str, float]:
    """
    loaders: (head_loader, tail_loader), each yields (pos, candidates, mode)
    candidates have the gold at column 0.
    Metrics: MR, MRR, Hits@1/3/10
    """
    model.eval()

    def _eval_one(loader):
        ranks = []
        for pos, cands, mode in loader:
            pos = pos.to(next(model.parameters()).device)
            cands = cands.to(pos.device)
            # Build (pos, cands) tuple to feed model; our forward expects (positive, negative-list)
            logits = model((pos, cands), mode=mode)  # (B, N)
            gold = logits[:, 0:1]  # (B,1)
            # tie-aware rank: 1 + (# >) + 0.5*(# == minus 1)
            greater = (logits[:, 1:] > gold).sum(dim=1).float()
            equal = (logits[:, 1:] == gold).sum(dim=1).float()
            r = 1.0 + greater + 0.5 * equal
            ranks.append(r.cpu())
        if not ranks:
            return {"MR": float("nan"), "MRR": float("nan"), "H@1": 0.0, "H@3": 0.0, "H@10": 0.0}
        ranks = torch.cat(ranks, dim=0)
        mr = ranks.mean().item()
        mrr = (1.0 / ranks).mean().item()
        h1 = (ranks <= 1).float().mean().item()
        h3 = (ranks <= 3).float().mean().item()
        h10 = (ranks <= 10).float().mean().item()
        return {"MR": mr, "MRR": mrr, "H@1": h1, "H@3": h3, "H@10": h10}

    head, tail = loaders
    m1 = _eval_one(head)
    m2 = _eval_one(tail)
    # average head/tail
    out = {k: (m1[k] + m2[k]) / 2.0 for k in m1.keys()}
    return out


class Trainer:
    def __init__(self, cfg, out_dir: str):
        self.cfg = cfg
        self.out_dir = out_dir
        self.device = pick_device(cfg.device)
        set_seed(cfg.seed)

        # 1) Data
        self.train_iter, self.valid_loaders, self.test_loaders, meta = build_data(cfg)
        self.nentity, self.nrelation = meta["nentity"], meta["nrelation"]
        # steps/epoch = head_batches + tail_batches
        self.steps_per_epoch = len(self.train_iter._src_head) + len(self.train_iter._src_tail)

        # 2) Model (or MED)
        model = create_model(
            cfg.model.name,
            nentity=self.nentity,
            nrelation=self.nrelation,
            base_dim=cfg.model.base_dim,
            gamma=cfg.model.gamma,
            **{k: v for k, v in vars(cfg.model).items() if k not in ("name", "base_dim", "gamma")},
        ).to(self.device)

        if cfg.med.enabled:
            self.train_obj = MEDTrainer(model, d_list=cfg.med.dims, submodels_per_step=cfg.med.submodels_per_step).to(
                self.device
            )
        else:
            self.train_obj = model  # plain

        # 3) Optimizer & Scheduler
        self.optimizer = build_optimizer(self.train_obj.parameters(), cfg)
        total_steps = cfg.sched.total_steps or (cfg.train.epochs * self.steps_per_epoch)
        self.scheduler = build_scheduler(self.optimizer, cfg, total_steps)

        # 4) Logging
        self.logger = CSVLogger(out_dir)
        self.timer = Timer()

        # 5) Book-keeping
        self.global_step = 0
        self.best_metric = -1.0
        self.best_tag = None

    def train(self):
        log_every = int(self.cfg.train.log_every)
        eval_every = int(self.cfg.train.eval_every)
        save_every = int(self.cfg.train.save_every)
        grad_clip = float(getattr(self.cfg.optim, "grad_clip", 0.0))

        for epoch in range(1, self.cfg.train.epochs + 1):
            # ---- training epoch ----
            for _ in range(self.steps_per_epoch):
                self.global_step += 1
                # Fetch a batch (alternates head <-> tail)
                pos, neg, w, mode = next(self.train_iter)
                pos = pos.to(self.device)
                neg = neg.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                if self.cfg.med.enabled:
                    loss, stats = self.train_obj(pos, neg, mode=mode)
                    loss.backward()
                else:
                    # Use the OpenKE-compatible helper on the **underlying** model
                    loss_stats = self.train_obj.train_step(
                        self.train_obj,
                        self.optimizer,
                        iter([(pos, neg, w, mode)]),
                        SimpleNamespace(
                            cuda=True, negative_adversarial_sampling=False, uni_weight=True, regularization=0.0
                        ),
                    )
                    # train_step already does optimizer.step(), so undo zero_grad/step logic
                    # For consistent logging we reconstruct a loss number:
                    loss = torch.tensor(loss_stats["loss"], device=self.device)

                if self.cfg.med.enabled:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.train_obj.parameters(), grad_clip)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                # ---- logging ----
                if self.global_step % log_every == 0:
                    row = {
                        "step": self.global_step,
                        "epoch": epoch,
                        "loss": float(loss.detach()),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "time": round(self.timer.elapsed(), 2),
                    }
                    self.logger.log(row)

                # ---- eval/save ----
                if self.global_step % eval_every == 0:
                    metrics = evaluate(
                        self.train_obj.model if hasattr(self.train_obj, "model") else self.train_obj, self.valid_loaders
                    )
                    row = {"step": self.global_step, "epoch": epoch, **{f"val_{k}": v for k, v in metrics.items()}}
                    self.logger.log(row)

                    # save "last"
                    save_checkpoint(
                        self.out_dir,
                        "last",
                        {"step": self.global_step, "epoch": epoch, "metrics": metrics},
                        self.train_obj,
                        self.optimizer,
                        self.scheduler,
                    )

                    # save best
                    key = getattr(self.cfg.train, "save_best_metric", "MRR")
                    cur = float(metrics.get(key, -1.0))
                    if cur > self.best_metric:
                        self.best_metric = cur
                        self.best_tag = f"best_{key.lower()}"
                        save_checkpoint(
                            self.out_dir,
                            self.best_tag,
                            {"step": self.global_step, "epoch": epoch, "metrics": metrics},
                            self.train_obj,
                            self.optimizer,
                            self.scheduler,
                        )

                if self.global_step % save_every == 0:
                    save_checkpoint(
                        self.out_dir,
                        f"step{self.global_step}",
                        {"step": self.global_step, "epoch": epoch},
                        self.train_obj,
                        self.optimizer,
                        self.scheduler,
                    )

            # end epoch

        # final eval on test
        test_metrics = evaluate(
            self.train_obj.model if hasattr(self.train_obj, "model") else self.train_obj, self.test_loaders
        )
        self.logger.log({f"test_{k}": v for k, v in test_metrics.items()})
        print("Test:", {k: round(v, 4) for k, v in test_metrics.items()})
        self.logger.close()
