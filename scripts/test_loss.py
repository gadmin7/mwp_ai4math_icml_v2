#!/usr/bin/env python3
"""Per-level loss on the TEST set -- the selection-bias-free version of our headline metric.

pipeline.py reports per-level loss on the VAL split, but BestAdapterCallback also
*selects* each stage's checkpoint by val loss. Reporting the same split we selected on
biases the number optimistically, and unevenly: with eval_steps=150 a single-stage arm
restores once from ~14 candidates while a 5-stage arm restores five times from far
fewer. The direction of that bias is not obvious, which is precisely why it cannot be
waved away at margins of ~0.007.

The test split has never touched selection, and loss needs no generation, so this
re-derives the comparison cleanly in ~2 min per arm.

    python scripts/test_loss.py --configs configs/staged.yaml configs/jointu.yaml
"""
import argparse, json, os, sys
import torch, yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import _level_int, load_math_splits
from src.evaluate import load_stacked_model
from src.lora_schedule import get_baseline
from src.prompts import PROMPT_TEMPLATES


@torch.no_grad()
def per_level_loss(model, tok, ds, template, batch_size, max_len):
    model.eval()
    out = {}
    for lv in (1, 2, 3, 4, 5):
        rows = [x for x in ds if _level_int(x) == lv]
        total, n = 0.0, 0
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            texts = [template.format(problem=r["problem"], solution=r["solution"]) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(model.device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            total += model(**enc, labels=labels).loss.item() * len(chunk)
            n += len(chunk)
        out[f"L{lv}"] = total / n
        print(f"    L{lv}  n={n:<6} loss={out[f'L{lv}']:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="results/test_loss.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    results = {}
    for cfg_path in args.configs:
        cfg = yaml.safe_load(open(cfg_path))
        baseline = get_baseline(cfg["baseline_id"])
        prompt = cfg.get("prompt", "default")
        print(f"\n=== {baseline.id} (prompt={prompt}) ===", flush=True)
        tok = AutoTokenizer.from_pretrained(cfg["base_model_id"], token=os.environ.get("HF_TOKEN"))
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = load_stacked_model(
            baseline=baseline, base_model_id=cfg["base_model_id"],
            hf_org=cfg.get("hf_org"), model_tag=cfg["model_tag"],
            quantize=cfg.get("quantize", True), token=os.environ.get("HF_TOKEN"),
            local_dir=os.path.join("runs", cfg["model_tag"]),
        )
        splits = load_math_splits(seed=cfg["seed"])
        results[baseline.id] = per_level_loss(model, tok, splits.test,
                                              PROMPT_TEMPLATES[prompt], args.batch_size, args.max_len)
        del model
        torch.cuda.empty_cache()

    n = {"L1": 437, "L2": 894, "L3": 1131, "L4": 1214, "L5": 1324}; N = sum(n.values())
    print(f"\n{'arm':<12}" + "".join(f"{k:>9}" for k in n) + f"{'test-wtd':>11}")
    for arm, d in results.items():
        w = sum(d[k] * n[k] for k in n) / N
        results[arm]["weighted_mean"] = w
        print(f"{arm:<12}" + "".join(f"{d[k]:>9.4f}" for k in n) + f"{w:>11.4f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
