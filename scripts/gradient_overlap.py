#!/usr/bin/env python3
"""Do different difficulty levels want to move the weights in the SAME directions?

This tests the guidance hypothesis -- "does level 1's subspace help level 2?" -- WITHOUT
training anything, and without the confound that ruined our earlier adapter measurement.

Why the adapter measurement could not answer it: LoRA initialises A randomly and B=0, and
gradient descent finds a solution near that init. Two independently-initialised adapters
therefore occupy near-orthogonal subspaces no matter how related the tasks are. Our own
data proved this -- stages 2 and 5 accidentally shared an RNG state and ended up with 0.96
subspace overlap despite training on different data, while every independently-initialised
pair sat at chance. We measured our initialisation scheme, not the levels.

Gradients have no such confound. dL/dW at a fixed set of weights is a property of the task.
So: for each level, take the top-k singular directions of the gradient and compare across
levels via principal angles.

Reading the numbers requires three reference points, not one:

    random baseline   k/d              two unrelated subspaces still overlap this much
    SAME-TASK CEILING overlap(L_a,L_b) two disjoint halves of ONE level -- this is the
                                       most agreement any pair can show, given that a
                                       finite sample estimates the gradient noisily
    measured          overlap(L_i,L_j)

A cross-level overlap near the ceiling means the levels genuinely share structure and a
pipeline should reuse it (warm-start, nested subspaces). Near chance means they are
effectively different tasks and stages should get orthogonal capacity instead.

    python scripts/gradient_overlap.py --n-per-level 64 --k 32

Runs in ~15 min on an A100. No adapters, no training, no checkpoints.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import _level_int, load_math_splits
from src.prompts import PROMPT_TEMPLATE

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def target_params(model, modules, every_n_layers):
    """The weight matrices we differentiate with respect to."""
    out = {}
    for name, p in model.named_parameters():
        if not name.endswith(".weight") or p.ndim != 2:
            continue
        if not any(f".{m}." in name for m in modules):
            continue
        layer = None
        for part in name.split("."):
            if part.isdigit():
                layer = int(part)
                break
        if layer is not None and every_n_layers > 1 and layer % every_n_layers:
            continue
        out[name] = p
    return out


def grad_subspaces(model, params, tokenizer, problems, k, batch_size, max_len, device):
    """Mean gradient over `problems`, reduced to its top-k right-singular directions.

    We SVD immediately and keep only the k basis vectors: holding full gradients for
    every level would be ~4GB each.
    """
    for p in params.values():
        p.grad = None
    n_batches = 0
    for i in range(0, len(problems), batch_size):
        chunk = problems[i:i + batch_size]
        texts = [PROMPT_TEMPLATE.format(problem=p, solution=s) for p, s in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_len).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        loss = model(**enc, labels=labels).loss
        loss.backward()          # accumulates
        n_batches += 1

    bases = {}
    for name, p in params.items():
        if p.grad is None:
            continue
        G = (p.grad / n_batches).float()          # [d_out, d_in]
        # right singular vectors span the INPUT directions the task pushes on,
        # which is the same object as rowspace(A) in a LoRA adapter.
        _, _, Vh = torch.linalg.svd(G, full_matrices=False)
        bases[name] = Vh[:k].T.contiguous().cpu()  # [d_in, k], orthonormal columns
        p.grad = None
    return bases


def shuffled_pairs(pairs, seed):
    """Same tokens, destroyed semantics -- the DIFFERENT-TASK floor.

    Gradient top-directions can be dominated by generic structure (token frequency,
    positional effects) rather than by what the task is actually about. If that happens,
    every level looks identical and a naive reading reports spurious "shared structure".
    Word-shuffling each example preserves the token distribution while removing meaning,
    so overlap(real, shuffled) measures how much agreement comes from generic structure
    alone. If that floor sits near the same-task ceiling, the instrument cannot
    discriminate and the cross-level numbers mean nothing.
    """
    import random
    rng = random.Random(seed)
    out = []
    for prob, sol in pairs:
        w = (prob + " " + sol).split()
        rng.shuffle(w)
        cut = max(1, len(w) // 2)
        out.append((" ".join(w[:cut]), " ".join(w[cut:])))
    return out


def overlap(Q1, Q2):
    """Normalised subspace overlap in [0,1]: mean squared cosine of principal angles."""
    k = min(Q1.shape[1], Q2.shape[1])
    s = torch.linalg.svdvals(Q1.T @ Q2)
    return (s[:k] ** 2).sum().item() / k


def mean_overlap(b1, b2):
    common = sorted(set(b1) & set(b2))
    if not common:
        return float("nan")
    return sum(overlap(b1[n], b2[n]) for n in common) / len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--n-per-level", type=int, default=64,
                    help="problems per HALF (each level uses 2x this)")
    ap.add_argument("--k", type=int, default=32, help="subspace dimension")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--every-n-layers", type=int, default=4,
                    help="subsample layers for speed; 1 = all")
    ap.add_argument("--modules", default="q_proj,down_proj")
    ap.add_argument("--out", default="results/gradient_overlap.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # No quantisation and no LoRA: we need gradients w.r.t. the real weight matrices.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, token=os.environ.get("HF_TOKEN")).to(device)
    model.gradient_checkpointing_enable()
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    modules = tuple(m.strip() for m in args.modules.split(","))
    params = target_params(model, modules, args.every_n_layers)
    for p in params.values():
        p.requires_grad_(True)
    d_in = {n: p.shape[1] for n, p in params.items()}
    print(f"differentiating {len(params)} weight matrices "
          f"(modules={modules}, every {args.every_n_layers} layers), k={args.k}")

    splits = load_math_splits(seed=args.seed)
    train = splits.train
    rng = torch.Generator().manual_seed(args.seed)

    halves = {}
    for lv in (1, 2, 3, 4, 5):
        idx = [i for i, x in enumerate(train) if _level_int(x) == lv]
        perm = torch.randperm(len(idx), generator=rng).tolist()
        need = 2 * args.n_per_level
        if len(idx) < need:
            print(f"  level {lv}: only {len(idx)} problems, using {len(idx)//2} per half")
        take = [idx[j] for j in perm[:min(need, len(idx) - len(idx) % 2)]]
        half = len(take) // 2
        for tag, sel in (("a", take[:half]), ("b", take[half:])):
            probs = [(train[i]["problem"], train[i]["solution"]) for i in sel]
            print(f"  gradient: level {lv}{tag}  (n={len(probs)})")
            halves[(lv, tag)] = grad_subspaces(model, params, tok, probs, args.k,
                                               args.batch_size, args.max_len, device)

    # DIFFERENT-TASK floor: level-1 problems with their words shuffled.
    lv1_idx = [i for i, x in enumerate(train) if _level_int(x) == 1][:args.n_per_level]
    lv1_pairs = [(train[i]["problem"], train[i]["solution"]) for i in lv1_idx]
    print(f"  gradient: level 1 SHUFFLED  (n={len(lv1_pairs)})  [different-task floor]")
    shuffled = grad_subspaces(model, params, tok, shuffled_pairs(lv1_pairs, args.seed),
                              args.k, args.batch_size, args.max_len, device)

    chance = sum(args.k / d for d in d_in.values()) / len(d_in)
    ceiling = {lv: mean_overlap(halves[(lv, "a")], halves[(lv, "b")]) for lv in range(1, 6)}
    ceil_mean = sum(ceiling.values()) / len(ceiling)
    floor = mean_overlap(halves[(1, "a")], shuffled)

    print("\n=== REFERENCE POINTS ===")
    print(f"  random-subspace baseline (k/d) : {chance:.4f}")
    print(f"  DIFFERENT-TASK floor (shuffled): {floor:.4f}   ({floor/chance:.2f}x chance)")
    print("  SAME-TASK ceiling (disjoint halves of one level):")
    for lv, v in ceiling.items():
        print(f"      level {lv}: {v:.4f}   ({v/chance:.2f}x chance)")
    print(f"      mean  : {ceil_mean:.4f}")

    span = ceil_mean - floor
    if span <= 0.02:
        print("\n  !! WARNING: ceiling and different-task floor are nearly equal.")
        print("     Gradient directions here are dominated by generic structure, not task")
        print("     content -- this instrument cannot discriminate. Increase --k, use more")
        print("     layers/modules, or subtract the mean gradient before comparing.")

    print("\n=== CROSS-LEVEL OVERLAP (half 'a' of each) ===")
    print("       " + "".join(f"   L{j}   " for j in range(1, 6)))
    cross = {}
    for i in range(1, 6):
        row = f"  L{i}  "
        for j in range(1, 6):
            if i == j:
                row += "    -   "
                continue
            v = mean_overlap(halves[(i, "a")], halves[(j, "a")])
            cross[f"{i}-{j}"] = v
            row += f" {v:.4f}"
        print(row)

    print("\n=== INTERPRETATION ===")
    print("  scale: 0% = different-task floor (generic structure only), "
          "100% = same-task ceiling")
    for key in [f"1-{j}" for j in range(2, 6)]:
        v = cross[key]
        frac = (v - floor) / span if span > 0 else float("nan")
        print(f"  L{key[-1]} vs L1: {v:.4f}  = {100*frac:6.1f}% of the way to ceiling")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"chance": chance, "floor_shuffled": floor, "ceiling": ceiling,
               "ceiling_mean": ceil_mean, "cross": cross, "args": vars(args)},
              open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")
    print("\nnear ceiling -> levels share structure; reuse it (warm-start / nested subspace)"
          "\nnear chance  -> levels are different tasks; give stages orthogonal capacity")


if __name__ == "__main__":
    main()
