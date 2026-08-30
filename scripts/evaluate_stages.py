#!/usr/bin/env python3
"""Evaluate a staged arm at EVERY intermediate stage, for forgetting analysis.

`scripts/evaluate.py` scores only the final model. This reconstructs the model as it
stood after each stage k (adapters 1..k active, later stages absent) and evaluates it,
which is what the forgetting / backward-transfer and error-trajectory analyses need.

This is only possible because each stage is saved as a separate frozen adapter -- under
an implementation that overwrites adapters between stages, intermediate models cannot be
recovered after the fact.

    # Level-1 forgetting curve: how does Level-1 accuracy move as harder stages arrive?
    python scripts/evaluate_stages.py --config configs/baseline6.yaml --scope level1

    # Full trajectory: at stage k, evaluate all test levels <= k.
    # Needed for the lost/recovered/newly_solved/stable_correct buckets.
    python scripts/evaluate_stages.py --config configs/baseline6.yaml --scope cumulative

Writes results/<arm>-stage<k>-<scope>-predictions.csv per stage, plus a combined
results/<arm>-<scope>-trajectory.csv with one row per (problem, stage).
"""

import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import _level_int, load_math_splits
from src.evaluate import accuracy, box_rate, evaluate_baseline
from src.lora_schedule import get_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scope", choices=["level1", "cumulative", "all"], default="level1",
                        help="level1: only Level-1 problems (cheap forgetting curve). "
                             "cumulative: all levels <= k (enables bucket analysis). "
                             "all: the entire test set regardless of stage -- needed for "
                             "zero-shot (--stages 0), where 'levels <= 0' would be empty.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prompt", choices=["default", "cot"], default="default",
                        help="cot: explicitly request the '## Step N' scaffold the "
                             "instruct model uses zero-shot, to test whether fine-tuning "
                             "merely overwrote the default answering format")
    parser.add_argument("--stages", default=None,
                        help="comma-separated stages to evaluate (default: all but the "
                             "last, which scripts/evaluate.py already covers)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    baseline = get_baseline(cfg["baseline_id"])
    if baseline.num_stages == 1:
        raise SystemExit(f"{baseline.id} is single-stage; nothing intermediate to evaluate")

    # stage 0 is legal and means "no adapters": the zero-shot base model.
    stages = ([int(s) for s in args.stages.split(",")] if args.stages
              else list(range(1, baseline.num_stages)))

    splits = load_math_splits(seed=cfg["seed"])
    local_dir = os.path.join(args.runs_dir, cfg["model_tag"])
    os.makedirs(args.out_dir, exist_ok=True)

    frames = []
    for k in stages:
        if args.scope == "level1":
            subset = splits.test.filter(lambda x: _level_int(x) == 1)
        elif args.scope == "all":
            subset = splits.test
        else:
            subset = splits.test.filter(lambda x: _level_int(x) <= k)
        if len(subset) == 0:
            raise SystemExit(
                f"empty evaluation set for stage {k} with --scope {args.scope}; "
                "use --scope all for zero-shot (stage 0)"
            )
        print(f"\n=== {baseline.id} @ stage {k}/{baseline.num_stages} "
              f"({args.scope}, n={len(subset)}) ===")

        df = evaluate_baseline(
            baseline=baseline,
            base_model_id=cfg["base_model_id"],
            hf_org=cfg.get("hf_org"),
            model_tag=cfg["model_tag"],
            test_ds=subset,
            quantize=cfg.get("quantize", True),
            token=os.environ.get("HF_TOKEN"),
            batch_size=args.batch_size,
            local_dir=local_dir,
            prompt=args.prompt,
            through_stage=k,          # <-- adapters 1..k only
        )
        df["stage"] = k
        df.to_csv(os.path.join(args.out_dir, f"{baseline.id}-stage{k}-{args.scope}{'' if args.prompt=='default' else '-'+args.prompt}-predictions.csv"),
                  index=False)
        frames.append(df)
        print(f"  stage {k}: EM {100 * accuracy(df):.2f}%  box {100 * box_rate(df):.1f}%")

    traj = pd.concat(frames, ignore_index=True)
    traj_path = os.path.join(args.out_dir, f"{baseline.id}-{args.scope}-trajectory.csv")
    traj.to_csv(traj_path, index=False)

    print(f"\n=== {baseline.id} trajectory ({args.scope}) ===")
    print(f"{'stage':<8}{'n':<8}{'EM %'}")
    for k, g in traj.groupby("stage"):
        print(f"{k:<8}{len(g):<8}{100 * accuracy(g):.2f}")
    print(f"\ntrajectory -> {traj_path}")
    print("(combine with the final-stage predictions from scripts/evaluate.py for the "
          "full per-problem correctness vector)")


if __name__ == "__main__":
    main()
