#!/usr/bin/env python3
"""Run inference on the held-out TEST set and report exact-match accuracy.

This is the ONLY place the test split is ever touched -- training uses the 5% val
split carved from train (see src/data.py), so nothing here has leaked into model
selection.

    # cheap sanity check first (~1-2 min): does the model emit boxed answers at all?
    python scripts/evaluate.py --config configs/baseline9.yaml --limit 100

    # full 5000-problem evaluation
    python scripts/evaluate.py --config configs/baseline9.yaml

Adapters load from local runs/ when present, else from the Hub. All of a baseline's
stage adapters are activated together, so the scored model is the full frozen stack
plus its final stage.
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_math_splits
from src.evaluate import evaluate_baseline, report
from src.lora_schedule import get_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs-dir", default="runs", help="where training wrote checkpoints")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prompt", choices=["default", "cot"], default="default",
                        help="cot: explicitly request the '## Step N' scaffold the "
                             "instruct model uses zero-shot, to test whether fine-tuning "
                             "merely overwrote the default answering format")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N test problems (sanity check)")
    parser.add_argument("--hub-only", action="store_true",
                        help="ignore local checkpoints and pull every stage from the Hub")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    baseline = get_baseline(cfg["baseline_id"])
    local_dir = None if args.hub_only else os.path.join(args.runs_dir, cfg["model_tag"])

    splits = load_math_splits(seed=cfg["seed"])
    test_ds = splits.test
    if args.limit:
        test_ds = test_ds.select(range(min(args.limit, len(test_ds))))
        print(f"NOTE: sanity-check run on {len(test_ds)} of {len(splits.test)} test problems; "
              f"these numbers are NOT the reported result.")

    print(f"\nevaluating {baseline.id}: {baseline.name}")
    df = evaluate_baseline(
        baseline=baseline,
        base_model_id=cfg["base_model_id"],
        hf_org=cfg.get("hf_org"),
        model_tag=cfg["model_tag"],
        test_ds=test_ds,
        quantize=cfg.get("quantize", True),
        token=os.environ.get("HF_TOKEN"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        local_dir=local_dir,
        prompt=args.prompt,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = (f"-limit{args.limit}" if args.limit else "") + ("" if args.prompt == "default" else f"-{args.prompt}")
    out_csv = os.path.join(args.out_dir, f"{baseline.id}{suffix}-test-predictions.csv")
    df.to_csv(out_csv, index=False)

    summary = report(df, label=f"{baseline.id} — {baseline.name}")
    print("\n" + summary)
    with open(os.path.join(args.out_dir, f"{baseline.id}{suffix}-summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\npredictions -> {out_csv}")


if __name__ == "__main__":
    main()
