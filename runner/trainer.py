# runner/trainer.py
from __future__ import annotations

import os
import random
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from med.med_wrapper import MEDTrainer
from models import create_model
from runner.checkpoint import load_checkpoint, save_checkpoint
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

    # Sampling params for faster experimentation
    sample_ratio = getattr(cfg.data, "sample_ratio", None)
    max_entities = getattr(cfg.data, "max_entities", None)
    max_triples = getattr(cfg.data, "max_triples", None)
    sample_seed = int(getattr(cfg.data, "sample_seed", 42))
    # Additional validation sampling for even faster validation during training
    sample_valid_ratio = getattr(cfg.data, "sample_valid_ratio", None)

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
            sample_ratio=sample_ratio,
            max_entities=max_entities,
            max_triples=max_triples,
            sample_seed=sample_seed,
            sample_valid_ratio=sample_valid_ratio,
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
            sample_ratio=sample_ratio,
            max_entities=max_entities,
            max_triples=max_triples,
            sample_seed=sample_seed,
            sample_valid_ratio=sample_valid_ratio,
        )
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset.name}")


@torch.no_grad()
def evaluate(model, loaders) -> dict[str, float]:
    """
    loaders: (head_loader, tail_loader), each yields (pos, candidates, mode)
    candidates have the gold at column 0.
    Metrics: MR, MRR, Hits@1/3/10

    For MED models: evaluates the largest dimension only during training
    (full per-dimension evaluation is done in final evaluation via evaluate.py)
    """
    model.eval()

    # Check if this is a MED wrapper - if so, use the base model at max dimension
    is_med_wrapper = hasattr(model, "d_list") and hasattr(model, "model")
    if is_med_wrapper:
        # For MED during training, evaluate all dimensions to find the best
        eval_model = model.model  # Get the underlying KGModel
        all_metrics = {}
        best_mrr = -1
        best_metrics = None

        def _eval_one_dim(loader, dim):
            ranks = []
            for pos, cands, mode in loader:
                pos = pos.to(next(eval_model.parameters()).device)
                cands = cands.to(pos.device)
                # CRITICAL: Pass crop_dim to the model forward!
                logits = eval_model((pos, cands), mode=mode, crop_dim=dim)  # (B, N)
                gold = logits[:, 0:1]  # (B,1)
                # tie-aware rank: 1 + (# >) + 0.5*(# ==)
                greater = (logits > gold).sum(dim=1).float()
                equal = (logits == gold).sum(dim=1).float() - 1.0
                r = 1.0 + greater + 0.5 * equal
                ranks.append(r.cpu())
                # Free GPU memory to prevent fragmentation with large dimensions
                del logits, gold, greater, equal, r
            if not ranks:
                return {"MR": float("nan"), "MRR": float("nan"), "H@1": 0.0, "H@3": 0.0, "H@10": 0.0}
            ranks = torch.cat(ranks, dim=0)
            mr = ranks.mean().item()
            mrr = (1.0 / ranks).mean().item()
            h1 = (ranks <= 1).float().mean().item()
            h3 = (ranks <= 3).float().mean().item()
            h10 = (ranks <= 10).float().mean().item()
            return {"MR": mr, "MRR": mrr, "H@1": h1, "H@3": h3, "H@10": h10}

        for d in model.d_list:
            head, tail = loaders
            m1 = _eval_one_dim(head, d)
            m2 = _eval_one_dim(tail, d)
            # average head/tail
            metrics = {k: (m1[k] + m2[k]) / 2.0 for k in m1.keys()}
            if metrics["MRR"] > best_mrr:
                best_mrr = metrics["MRR"]
                best_metrics = metrics
            all_metrics[f"dim_{d}"] = metrics
            # Clear cache after each dimension to prevent fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return best_metrics, all_metrics

    else:
        eval_model = model

    def _eval_one(loader):
        ranks = []
        for pos, cands, mode in loader:
            pos = pos.to(next(eval_model.parameters()).device)
            cands = cands.to(pos.device)
            # Build (pos, cands) tuple to feed model; our forward expects (positive, negative-list)
            logits = eval_model((pos, cands), mode=mode)  # (B, N)
            gold = logits[:, 0:1]  # (B,1)
            # tie-aware rank: 1 + (# >) + 0.5*(# ==)
            # Compare gold score against ALL scores (including itself at position 0)
            greater = (logits > gold).sum(dim=1).float()
            equal = (logits == gold).sum(dim=1).float() - 1.0  # subtract 1 for gold itself
            r = 1.0 + greater + 0.5 * equal
            ranks.append(r.cpu())
            # Free GPU memory to prevent fragmentation
            del logits, gold, greater, equal, r
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

    return out, None


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
        # Avoid passing plugin-specific flags (use_rscf/use_mi/mi_*) into model ctor
        model_ctor_kwargs = {
            k: v
            for k, v in vars(cfg.model).items()
            if k
            not in (
                "name",
                "base_dim",
                "gamma",
                "use_rscf",
                "rscf_et",
                "rscf_rt",
                "rscf_alpha",
                "use_mi",
                "mi_lambda",
                "mi_q_dim",
                "mi_use_info_nce",
                "mi_use_jsd",
            )
        }

        # === CONTEXT POOLING INTEGRATION ===
        if cfg.model.name.lower() in ["cp", "contextpooling"]:
            # Extract raw triples from the dataset for graph construction
            train_triples = self.train_iter._src_head.dataset.triples
            model_ctor_kwargs["train_triples"] = train_triples
        # ===================================

        model = create_model(
            cfg.model.name,
            nentity=self.nentity,
            nrelation=self.nrelation,
            base_dim=cfg.model.base_dim,
            gamma=cfg.model.gamma,
            **model_ctor_kwargs,
        ).to(self.device)

        # Optionally attach RSCF as a plug-in filter module
        if getattr(cfg.model, "use_rscf", False):
            try:
                from models.rscf_wrapper import attach_rscf

                rscf_module = attach_rscf(model, use_et=getattr(cfg.model, "rscf_et", True), use_rt=getattr(cfg.model, "rscf_rt", True))
                # Move RSCF module to the same device as the model
                rscf_module.to(self.device)
            except Exception as exc:  # pragma: no cover - best-effort
                print(f"Warning: failed to attach RSCF: {exc}")

        # Optionally attach MI module if requested
        if getattr(cfg.model, "use_mi", False):
            try:
                from models.mi_wrapper import attach_mi

                # allow passing some mi options via cfg.model
                mi_q_dim = getattr(cfg.model, "mi_q_dim", None)
                mi_use_info_nce = bool(getattr(cfg.model, "mi_use_info_nce", True))
                mi_use_jsd = bool(getattr(cfg.model, "mi_use_jsd", False))
                mi_module = attach_mi(model, q_dim=mi_q_dim, use_info_nce=mi_use_info_nce, use_jsd=mi_use_jsd)
                # Move MI module to the same device as the model
                mi_module.to(self.device)
            except Exception as exc:  # pragma: no cover - best-effort
                print(f"Warning: failed to attach MI module: {exc}")

        if cfg.med.enabled:
            # pass MI-related kwargs to MEDTrainer via cfg.med if present (e.g., mi_lambda)
            med_kwargs = {}
            # carry over mi_lambda if set at config.model.mi_lambda
            med_kwargs["mi_lambda"] = float(getattr(cfg.model, "mi_lambda", 0.0))
            med_kwargs["mi_neg_size"] = int(getattr(cfg.data, "neg_size", 64))
            self.train_obj = MEDTrainer(model, d_list=cfg.med.dims, submodels_per_step=cfg.med.submodels_per_step, **med_kwargs).to(
                self.device
            )
        else:
            self.train_obj = model  # plain

        # 3) Optimizer & Scheduler
        self.optimizer = build_optimizer(self.train_obj.parameters(), cfg)
        total_steps = cfg.sched.total_steps or (cfg.train.epochs * self.steps_per_epoch)
        self.scheduler = build_scheduler(self.optimizer, cfg, total_steps)

        # 4) Logging
        base_cols = ["step", "epoch", "loss", "lr", "time"]
        med_cols = ["L_total", "L_ml", "L_ei"] if getattr(cfg.med, "enabled", False) else ["L_total"]

        metrics_keys = ["MR", "MRR", "H@1", "H@3", "H@10"]
        eval_cols = [f"val_{k}" for k in metrics_keys]

        test_cols = [f"test_{k}" for k in metrics_keys]
        dim_test_cols = []

        # If MED is enabled, pre-register per-dimension metric columns so CSV headers cover them.
        dim_eval_cols = []
        if getattr(cfg.med, "enabled", False):
            for d in cfg.med.dims:
                for k in metrics_keys:
                    dim_eval_cols.append(f"val_dim_{d}_{k}")
                    dim_test_cols.append(f"test_dim_{d}_{k}")

        self.logger = CSVLogger(out_dir, fieldnames=base_cols + med_cols + eval_cols + test_cols + dim_eval_cols + dim_test_cols)
        self.timer = Timer()

        # 5) Book-keeping
        self.global_step = 0
        self.best_metric = -1.0
        self.best_tag = None
        self.total_epochs = int(cfg.train.epochs)
        self._last_log_time = time.time()

        # 6) Build model config for checkpoints (enables proper evaluation)
        self.model_config = {
            "model_name": cfg.model.name,
            "base_dim": cfg.model.base_dim,
            "nentity": self.nentity,
            "nrelation": self.nrelation,
            "gamma": cfg.model.gamma,
        }

        # Add MED-specific config if applicable
        if cfg.med.enabled:
            self.model_config["med_enabled"] = True
            # Convert to list for JSON serialization
            self.model_config["d_list"] = list(cfg.med.dims)
        else:
            self.model_config["med_enabled"] = False

        # === RESUME LOGIC ===
        self.start_epoch = 1
        last_ckpt_path = os.path.join(self.out_dir, "ckpt_last.pt")
        if os.path.exists(last_ckpt_path):
            print(f"Found checkpoint at {last_ckpt_path}. Resuming training...")
            try:
                state = load_checkpoint(last_ckpt_path, self.train_obj, self.optimizer, self.scheduler)
                self.global_step = state.get("step", 0)
                # The checkpoint saves the 'completed' epoch, so we start from the next one
                self.start_epoch = state.get("epoch", 0) + 1

                # Try to restore best_metric
                metrics = state.get("metrics", {})
                metric_key = getattr(self.cfg.train, "save_best_metric", "MRR")
                if metric_key in metrics:
                    self.best_metric = float(metrics[metric_key])

                print(
                    f"Resumed successfully. Global Step: {self.global_step}, \
                        Next Epoch: {self.start_epoch}, Best {metric_key}: {self.best_metric:.4f}"
                )
            except Exception as e:
                print(f"Warning: Failed to resume from checkpoint: {e}")
                print("Starting training from scratch.")
        # ====================

    def _print_train_status(self, row: dict):
        """
        Pretty console line for training progress.
        `row` contains step, epoch, loss, lr, time and optionally L_total/L_ml/L_ei.
        """
        step = row.get("step")
        epoch = row.get("epoch")
        loss = row.get("loss")
        lr = row.get("lr")
        elapsed = row.get("time")
        # it/s since last print
        now = time.time()
        dt = max(1e-6, now - self._last_log_time)
        self._last_log_time = now
        it_per_s = 1.0 / dt

        parts = [
            f"[{epoch}/{self.total_epochs}]",
            f"step {step}",
            f"loss {loss:.4f}",
        ]
        for k, v in row.items():
            if k in ["step", "epoch", "loss", "lr", "time"] or not isinstance(v, float):
                continue
            parts.append(f"{k} {v:.6f}")

        parts += [
            f"lr {lr:.6g}",
            f"{it_per_s:.1f} it/s",
            f"elapsed {elapsed:.1f}s",
        ]
        print(" | ".join(parts))

    def train(self):
        log_every = int(self.cfg.train.log_every)
        eval_every = int(self.cfg.train.eval_every)
        save_every = int(self.cfg.train.save_every)
        grad_clip = float(getattr(self.cfg.optim, "grad_clip", 0.0))

        # Use self.start_epoch determined in __init__ (defaults to 1 if no ckpt)
        for epoch in range(self.start_epoch, self.cfg.train.epochs + 1):
            # ---- training epoch ----
            for _ in range(self.steps_per_epoch):
                self.global_step += 1
                # Fetch a batch (alternates head <-> tail)
                pos, neg, w, mode = next(self.train_iter)
                pos = pos.to(self.device)
                neg = neg.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                if getattr(self.cfg.model, "use_mi", False) and hasattr(self.train_obj, "compute_mi_loss"):
                    # manual OpenKE-like forward/backward so we can add MI loss
                    self.train_obj.train() if hasattr(self.train_obj, "train") else None
                    # positive score (single)
                    positive_score = self.train_obj(pos)
                    # negative score (batch)
                    negative_score = self.train_obj((pos, neg), mode=mode)

                    # compute OpenKE-style losses (simple version: BCE with logits)
                    # positive
                    pos_logits = positive_score.squeeze(1)
                    pos_loss = -F.logsigmoid(pos_logits).mean()
                    # negative
                    neg_loss = -F.logsigmoid(-negative_score).mean()
                    kge_loss = (pos_loss + neg_loss) / 2.0

                    # MI loss (on positive triples)
                    mi_lambda = float(getattr(self.cfg.model, "mi_lambda", 1.0))
                    mi_loss = self.train_obj.compute_mi_loss(pos, neg_size=int(getattr(self.cfg.data, "neg_size", 64)))

                    loss = kge_loss + mi_lambda * mi_loss
                    stats = {"L_total": float(loss.detach())}

                    # backward & step
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.train_obj.parameters(), grad_clip)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                if self.cfg.med.enabled:
                    loss, stats = self.train_obj(pos, neg, mode=mode)
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.train_obj.parameters(), grad_clip)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                else:
                    cuda_flag = True if (isinstance(self.device, str) and self.device.startswith("cuda")) else False
                    loss_stats = self.train_obj.train_step(
                        self.train_obj,
                        self.optimizer,
                        iter([(pos, neg, w, mode)]),
                        SimpleNamespace(
                            cuda=cuda_flag,
                            negative_adversarial_sampling=False,
                            uni_weight=True,
                            regularization=0.0,
                        ),
                    )
                    loss = torch.tensor(loss_stats["loss"], device=self.device)
                    stats = {"L_total": float(loss.detach())}
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
                    if isinstance(stats, dict):
                        # fills L_total (and L_ml/L_ei for MED)
                        row.update(stats)
                    self.logger.log(row)
                    self._print_train_status(row)

                # ---- eval/save ----
                if self.global_step % eval_every == 0:
                    best_metrics, all_metrics = evaluate(self.train_obj, self.valid_loaders)
                    # Always log the best metrics
                    row = {
                        "step": self.global_step,
                        "epoch": epoch,
                        **{f"val_{k}": v for k, v in best_metrics.items()},
                    }

                    # If we have per-dimension metrics, flatten and log them too
                    if all_metrics is not None:
                        for dim_name, m in all_metrics.items():  # e.g., dim_name = "dim_32"
                            for k, v in m.items():  # k in ["MR","MRR","H@1","H@3","H@10"]
                                row[f"val_{dim_name}_{k}"] = v

                    self.logger.log(row)
                    print(
                        f"[eval] step {self.global_step} | "
                        f"MRR {best_metrics['MRR']:.4f} | MR {best_metrics['MR']:.1f} | "
                        f"H@1 {best_metrics['H@1']:.4f} | H@3 {best_metrics['H@3']:.4f} | H@10 {best_metrics['H@10']:.4f}"
                    )

                    # save "last"
                    save_checkpoint(
                        self.out_dir,
                        "last",
                        {"step": self.global_step, "epoch": epoch, "metrics": best_metrics},
                        self.train_obj,
                        self.optimizer,
                        self.scheduler,
                        model_config=self.model_config,
                    )

                    # save best
                    key = getattr(self.cfg.train, "save_best_metric", "MRR")
                    cur = float(best_metrics.get(key, -1.0))
                    if cur > self.best_metric:
                        self.best_metric = cur
                        self.best_tag = f"best_{key.lower()}"
                        save_checkpoint(
                            self.out_dir,
                            self.best_tag,
                            {"step": self.global_step, "epoch": epoch, "metrics": best_metrics},
                            self.train_obj,
                            self.optimizer,
                            self.scheduler,
                            model_config=self.model_config,
                        )

                if self.global_step % save_every == 0:
                    save_checkpoint(
                        self.out_dir,
                        f"step{self.global_step}",
                        {"step": self.global_step, "epoch": epoch},
                        self.train_obj,
                        self.optimizer,
                        self.scheduler,
                        model_config=self.model_config,
                    )

            # end epoch

        # Save final checkpoint
        save_checkpoint(
            self.out_dir,
            "final",
            {"step": self.global_step, "epoch": epoch},
            self.train_obj,
            self.optimizer,
            self.scheduler,
            model_config=self.model_config,
        )

        # final eval on test (skip if requested, useful for quick smoke runs)
        if not bool(getattr(self.cfg.train, "skip_final_eval", False)):
            test_metrics, all_metrics = evaluate(self.train_obj, self.test_loaders)

            row = {"step": self.global_step, "epoch": epoch, **{f"test_{k}": v for k, v in test_metrics.items()}}

            if all_metrics is not None:
                for dim_name, m in all_metrics.items():  # e.g., "dim_32"
                    for k, v in m.items():  # "MR","MRR","H@1","H@3","H@10"
                        row[f"test_{dim_name}_{k}"] = v

            self.logger.log(row)
            print("Test:", {k: round(v, 6) for k, v in test_metrics.items()})
        else:
            print("Skipping final test evaluation (skip_final_eval=True)")
        self.logger.close()
