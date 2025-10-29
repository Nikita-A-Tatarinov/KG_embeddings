# train.py
from __future__ import annotations

import argparse

from runner.config import ensure_out_dir, load_config
from runner.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_out_dir(cfg)
    print(f"[config] {args.config}")
    print(f"[out] {out_dir}")

    trainer = Trainer(cfg, out_dir)
    trainer.train()


if __name__ == "__main__":
    main()
