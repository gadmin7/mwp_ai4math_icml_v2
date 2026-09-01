#!/usr/bin/env python3
"""Does training on level 1 actually HELP level 2, or do they merely resemble each other?

scripts/gradient_overlap.py showed that adjacent levels want nearly identical gradient
directions. That is necessary for the guidance hypothesis but not sufficient: it was
measured at fixed weights, so it cannot distinguish

    "L1 and L2 are the same task"                    (no hierarchy)
    "both need primitives that NEITHER has yet"      (hierarchy, seen before training)

Both predict high overlap. The discriminating measurement is sequential -- train on
level 1, then ask whether level 2's remaining gradient got smaller:

    ||g_L2|| drops a lot   -> L1 already solved part of L2's problem: real transfer
    ||g_L2|| unchanged     -> L1 taught L2 nothing

THE CONTROL THAT MAKES IT VALID: training on anything shrinks gradients generally, as
the model adapts to the domain and output format. So a drop on L2 alone proves nothing.
We also measure word-shuffled text, which shares the token distribution but no
mathematics. Transfer means L2's drop must EXCEED the shuffled drop.

Everything is paired -- the same held-out problems are scored before and after -- so
sampling noise largely cancels and n=64 is adequate, unlike the unpaired cross-level
comparison where n was the limiting factor.

Defaults to the BASE model: on an instruct model, training also destroys the
chain-of-thought scaffold (measured: 7.18 -> 0.00 step markers), which would confound
"learned L1" with "lost CoT". A base model has nothing to lose, so the change is pure
addition. Full fine-tuning, not LoRA -- no low-rank confound and no peft dependency.

    python scripts/transfer_test.py --n-train 400 --n-measure 64

~45 min on an A100.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import _level_int, load_math_splits
from src.prompts import PROMPT_TEMPLATES
from scripts.gradient_overlap import (mean_overlap, overlap, shuffled_pairs,
                                      target_params)


def measure(model, params, tok, pairs, k, batch_size, max_len, device, template):
    """Return (mean per-matrix gradient norm, top-k right-singular bases).

    The NORM answers "how much work is left on this task"; the BASES answer "in which
    directions". Both are needed: a drop in norm means the task got easier, a rotation
    in direction means what remains is different in kind.
    """
    for p in params.values():
        p.grad = None
    n_batches = 0
    for i in range(0, len(pairs), batch_size):
        texts = [template.format(problem=p, solution=s) for p, s in pairs[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        model(**enc, labels=labels).loss.backward()
        n_batches += 1

    norms, bases = {}, {}
    for name, p in params.items():
        if p.grad is None:
            continue
        G = (p.grad / n_batches).float()
        norms[name] = G.norm().item()
        _, _, Vh = torch.linalg.svd(G, full_matrices=False)
        bases[name] = Vh[:k].T.contiguous().cpu()
        p.grad = None
    return sum(norms.values()) / len(norms), bases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B",
                    help="BASE by default; training an instruct model also destroys its "
                         "CoT scaffold, which confounds the measurement")
    ap.add_argument("--prompt", default="plain", choices=sorted(PROMPT_TEMPLATES))
    ap.add_argument("--n-train", type=int, default=400, help="level-1 problems to train on")
    ap.add_argument("--n-measure", type=int, default=64, help="held-out problems per level")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--every-n-layers", type=int, default=4)
    ap.add_argument("--modules", default="q_proj,down_proj")
    ap.add_argument("--out", default="results/transfer_test.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    template = PROMPT_TEMPLATES[args.prompt]
    tok = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # fp32: we run a real optimizer here, and bf16 Adam states lose too much precision
    # for a short run. 1.24B params -> ~20GB with grads and Adam state; fine on 80GB.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, token=os.environ.get("HF_TOKEN")).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    modules = tuple(m.strip() for m in args.modules.split(","))
    measured = target_params(model, modules, args.every_n_layers)
    print(f"model={args.model}  prompt={args.prompt}  measuring {len(measured)} matrices")

    splits = load_math_splits(seed=args.seed)
    train = splits.train
    g = torch.Generator().manual_seed(args.seed)

    by_level = {}
    for lv in (1, 2, 3, 4, 5):
        idx = [i for i, x in enumerate(train) if _level_int(x) == lv]
        perm = torch.randperm(len(idx), generator=g).tolist()
        by_level[lv] = [idx[j] for j in perm]

    def pairs(indices):
        return [(train[i]["problem"], train[i]["solution"]) for i in indices]

    # Level 1 is split: some problems to TRAIN on, a disjoint set to MEASURE on.
    # Measuring on trained problems would report memorisation, not transfer.
    l1_train = pairs(by_level[1][:args.n_train])
    probe = {f"L{lv}": pairs(by_level[lv][args.n_train if lv == 1 else 0:][:args.n_measure])
             for lv in (1, 2, 3, 4, 5)}
    probe["shuffled"] = shuffled_pairs(pairs(by_level[1][:args.n_measure]), args.seed)
    print(f"train on {len(l1_train)} level-1 problems; probe sets: "
          + ", ".join(f"{k}={len(v)}" for k, v in probe.items()))

    for p in model.parameters():
        p.requires_grad_(True)

    print("\n=== BEFORE ===")
    before = {}
    for name, pr in probe.items():
        n, b = measure(model, measured, tok, pr, args.k, args.batch_size,
                       args.max_len, device, template)
        before[name] = (n, b)
        print(f"  {name:<9} ||g|| = {n:.6f}")

    # The BEFORE pass already computed each level's gradient basis at the base weights,
    # which is exactly what gradient_overlap.py measures -- so report cross-level overlap
    # here for free. (No same-task ceiling: that needs two disjoint halves per level.
    # The shuffled row is the floor, and it is the reference that matters most.)
    print("\n=== cross-level gradient overlap at BASE weights (free from the above) ===")
    names = ["L1", "L2", "L3", "L4", "L5", "shuffled"]
    print("           " + "".join(f"{n:>10}" for n in names))
    xlevel = {}
    for a in names:
        row = f"  {a:<9}"
        for b in names:
            v = mean_overlap(before[a][1], before[b][1])
            xlevel[f"{a}-{b}"] = v
            row += "         -" if a == b else f"{v:>10.4f}"
        print(row)

    print(f"\n=== TRAINING on level 1 ({args.epochs} epochs, lr={args.lr}) ===")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    step = 0
    for ep in range(args.epochs):
        order = torch.randperm(len(l1_train), generator=g).tolist()
        for i in range(0, len(order), args.batch_size):
            chunk = [l1_train[j] for j in order[i:i + args.batch_size]]
            texts = [template.format(problem=p, solution=s) for p, s in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len).to(device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            loss = model(**enc, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                print(f"    step {step:>4}  loss {loss.item():.4f}")
    print(f"  done: {step} steps")

    print("\n=== AFTER ===")
    after = {}
    for name, pr in probe.items():
        n, b = measure(model, measured, tok, pr, args.k, args.batch_size,
                       args.max_len, device, template)
        after[name] = (n, b)

    shuf_drop = 100 * (1 - after["shuffled"][0] / before["shuffled"][0])
    print(f"\n{'probe':<10}{'||g|| before':<15}{'||g|| after':<14}{'drop %':<10}"
          f"{'vs shuffled':<13}{'direction kept'}")
    print("-" * 78)
    rows = {}
    for name in ["L1", "L2", "L3", "L4", "L5", "shuffled"]:
        nb, bb = before[name]
        na, ba = after[name]
        drop = 100 * (1 - na / nb)
        rot = mean_overlap(bb, ba)
        excess = drop - shuf_drop
        tag = "" if name == "shuffled" else f"{excess:+.1f}pp"
        print(f"{name:<10}{nb:<15.6f}{na:<14.6f}{drop:<10.1f}{tag:<13}{rot:.4f}")
        rows[name] = {"before": nb, "after": na, "drop_pct": drop,
                      "excess_over_shuffled_pp": None if name == "shuffled" else excess,
                      "direction_overlap": rot}

    print("\n=== READING IT ===")
    graded = [rows[f"L{i}"]["excess_over_shuffled_pp"] for i in (1, 2, 3, 4, 5)]
    if max(graded) < 1.0:
        print("  NO TRANSFER: no level drops meaningfully more than shuffled text.")
        print("  Training on L1 produced generic adaptation only -- a curriculum has")
        print("  nothing to exploit, and the schedule sweep is not worth running.")
    elif graded[0] > graded[4] + 1.0:
        print("  GRADED TRANSFER: nearer levels benefit more than distant ones.")
        print("  L1 genuinely solved part of what the later levels need -> a curriculum")
        print("  has real structure to exploit; the schedule sweep is justified.")
    else:
        print("  UNIFORM TRANSFER: all levels benefit about equally.")
        print("  L1 training helps, but not in a difficulty-graded way -- that argues for")
        print("  training on everything at once rather than staging by level.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "shuffled_drop_pct": shuf_drop, "probes": rows,
           "cross_level_overlap_at_base": xlevel},
              open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
