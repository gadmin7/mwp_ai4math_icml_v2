#!/usr/bin/env python3
"""CLI entrypoint for a real training run.

    python scripts/run_baseline.py --config configs/baseline6.yaml [--no-push] [--batch-size 32]

Requires a real CUDA GPU (bitsandbytes 4-bit quantization) -- see cloud/jarvislabs_setup.sh.
Run scripts/smoke_test.py first, locally, before spending any cloud GPU time.
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_math_splits
from src.pipeline import run_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument("--map-num-proc", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    hf_token = os.environ.get("HF_TOKEN")
    if cfg.get("push_to_hub") and not args.no_push and not hf_token:
        raise SystemExit("HF_TOKEN env var required to push checkpoints (or pass --no-push)")

    splits = load_math_splits(seed=cfg["seed"])

    run_baseline(
        baseline_id=cfg["baseline_id"],
        base_model_id=cfg["base_model_id"],
        model_tag=cfg["model_tag"],
        splits=splits,
        output_dir=os.path.join(args.output_dir, cfg["model_tag"]),
        hf_org=cfg.get("hf_org"),
        push_to_hub=cfg.get("push_to_hub", True) and not args.no_push,
        quantize=cfg.get("quantize", True),
        hf_token=hf_token,
        seed=cfg["seed"],
        batch_size=args.batch_size or cfg.get("batch_size", 4),
        dataloader_num_workers=args.dataloader_num_workers,
        map_num_proc=args.map_num_proc,
        prompt=cfg.get("prompt", "default"),
    )


if __name__ == "__main__":
    main()
