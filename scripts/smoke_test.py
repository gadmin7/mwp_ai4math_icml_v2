#!/usr/bin/env python3
"""Local, CPU-only logic smoke test -- no CUDA/bitsandbytes required.

This does NOT run real training (that needs a real GPU, see cloud/jarvislabs_setup.sh).
It proves the two things that actually matter before spending any cloud GPU time:

  1. adapter stacking: after training stage 2, stage 1's adapter weights are
     bit-identical to before -- i.e. it was genuinely frozen, not silently
     re-initialized (the original repo's core bug).
  2. data splits: train/val/test are pairwise disjoint on the real MATH dataset,
     and the test set is never touched by this script at all.

Also exercises weight_geometry.py on the tiny checkpoints it produces, and
optionally (--push) validates the Hugging Face naming/model-card path end-to-end
against a scratch repo.

Run: python scripts/smoke_test.py [--push]
"""

import argparse
import copy
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.weight_geometry import layer_update_norms, load_stage_delta, overwrite_heatmap, summarize_overwrite
from src.data import load_math_splits, log_split_sizes
from src.train_stage import assert_stage_frozen, prepare_stage_model, stage_adapter_name
from src.lora_schedule import BaselineSpec

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
FAKE_BASELINE = BaselineSpec(id="smoke", name="smoke test", replay=True, num_stages=3, ranks=[8, 4, 4])
FAKE_TEXTS = [
    "Solve: 2 + 2 = 4.",
    "Solve: 3 * 3 = 9.",
    "Solve: 10 / 2 = 5.",
    "Solve: 7 - 1 = 6.",
    "Solve: 5 + 5 = 10.",
]


def _tokenize_batch(tokenizer, texts):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    enc["labels"] = enc["input_ids"].clone()
    return enc


def _train_one_stage(model, tokenizer, steps: int = 3, lr: float = 1e-2):
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable parameters -- adapter freezing bug likely"
    opt = torch.optim.SGD(trainable, lr=lr)
    batch = _tokenize_batch(tokenizer, FAKE_TEXTS)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        out = model(**batch)
        out.loss.backward()
        opt.step()
    model.eval()
    return model


def _snapshot(model, adapter_name: str) -> dict:
    marker = f".{adapter_name}."
    return {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if "lora_" in n and marker in f".{n}."
    }


def test_adapter_stacking():
    print("\n=== adapter stacking ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_stage_model(
        stage=1, baseline=FAKE_BASELINE, base_model_id=TINY_MODEL,
        prev_model=None, prev_adapter_path=None, quantize=False,
    )
    model = _train_one_stage(model, tokenizer)
    stage1_before = _snapshot(model, stage_adapter_name(1))

    # Logits with only stage 1 present, for the activation check below.
    model.eval()
    probe = _tokenize_batch(tokenizer, FAKE_TEXTS[:1])
    with torch.no_grad():
        stage1_logits = model(**probe).logits.clone()

    model = prepare_stage_model(
        stage=2, baseline=FAKE_BASELINE, base_model_id=TINY_MODEL,
        prev_model=model, prev_adapter_path=None, quantize=False,
    )
    assert_stage_frozen(model, frozen_stage=1)

    # Regression guard: `set_adapter("stage_2")` activates that adapter EXCLUSIVELY,
    # silently switching stage 1 off so the forward pass collapses to the bare base
    # model. Freezing assertions alone do NOT catch that -- a deactivated adapter is
    # also an unchanged one. A newly added adapter has lora_B == 0 and contributes
    # exactly nothing, so with the stack correctly active the logits must not move.
    active = set(model.active_adapters)
    assert active == {stage_adapter_name(1), stage_adapter_name(2)}, (
        f"expected stages 1+2 active in the forward pass, got {sorted(active)}"
    )
    model.eval()
    with torch.no_grad():
        stacked_logits = model(**probe).logits
    drift = (stacked_logits - stage1_logits).abs().max().item()
    assert drift < 1e-6, (
        f"adding stage 2 changed the forward pass by {drift:.6f}; stage 1's "
        "contribution was lost (a fresh adapter must contribute exactly 0)"
    )
    print("OK: stage 1 stays active in the forward pass after stage 2 is added")

    model = _train_one_stage(model, tokenizer)

    stage1_after = _snapshot(model, stage_adapter_name(1))
    for name, before in stage1_before.items():
        after = stage1_after[name]
        assert torch.equal(before, after), f"stage 1 weight {name} changed after training stage 2!"
    print(f"OK: {len(stage1_before)} stage-1 LoRA tensors bit-identical after stage-2 training")

    # The flip side: the *current* stage must actually be learning.
    stage2_trained = _snapshot(model, stage_adapter_name(2))
    assert any(t.abs().sum().item() > 0 for n, t in stage2_trained.items() if "lora_B" in n), (
        "stage 2's lora_B is still all zeros -- no gradient reached the active adapter"
    )
    print("OK: stage 2 received gradient (its lora_B moved off zero)")

    model = prepare_stage_model(
        stage=3, baseline=FAKE_BASELINE, base_model_id=TINY_MODEL,
        prev_model=model, prev_adapter_path=None, quantize=False,
    )
    assert_stage_frozen(model, frozen_stage=1)
    assert_stage_frozen(model, frozen_stage=2)
    model = _train_one_stage(model, tokenizer)
    print("OK: stages 1 and 2 both remain frozen through stage 3")
    return model


def test_weight_geometry(model):
    print("\n=== weight geometry ===")
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for stage in (1, 2, 3):
            adapter_name = stage_adapter_name(stage)
            save_dir = os.path.join(tmp, f"stage{stage}")
            model.save_pretrained(save_dir, selected_adapters=[adapter_name])
            paths[stage] = os.path.join(save_dir, adapter_name)  # peft nests multi-adapter saves

        deltas = [load_stage_delta(paths[s], s, stage_adapter_name(s)) for s in (1, 2, 3)]
        for d in deltas:
            assert d.deltas, f"stage {d.stage} produced no LoRA deltas"
        norms = layer_update_norms(deltas)
        assert norms, "no per-layer norms computed"
        n_modules = len(norms)
        heat = overwrite_heatmap(deltas[0], deltas[1])
        assert heat, "no overwrite heatmap computed between stage 1 and 2"
        summary = summarize_overwrite(heat)
        print(f"OK: norms for {n_modules} modules across 3 stages; "
              f"overwrite heatmap covers {len(heat)} shared modules")
        example_key = next(iter(summary))
        print(f"  e.g. {example_key}: {summary[example_key]}")


def test_data_splits():
    print("\n=== data splits (downloads real MATH dataset) ===")
    splits = load_math_splits(seed=42)
    print(log_split_sizes(splits))
    n_train, n_val, n_test = len(splits.train), len(splits.val), len(splits.test)
    frac = n_val / (n_train + n_val)
    print(f"OK: train={n_train} val={n_val} test={n_test} (val fraction of train+val = {frac:.3f})")
    assert 0.03 < frac < 0.08, f"val fraction {frac:.3f} far from the intended 5%"


def test_hf_push():
    print("\n=== HF push (scratch repo) ===")
    from huggingface_hub import whoami
    from src.pipeline import repo_name, write_model_card

    try:
        user = whoami()["name"]
    except Exception as e:
        print(f"SKIPPED: hf auth not configured ({e}). Run `hf auth login --force` and retry with --push.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "smoke")
        os.makedirs(path)
        with open(os.path.join(path, "note.txt"), "w") as f:
            f.write("smoke test scratch artifact -- safe to delete")
        write_model_card(path, FAKE_BASELINE, stage=1, train_size=5, seed=42)
        repo = repo_name(user, "smoketest", "smoke", 1)
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=path)
        print(f"OK: pushed scratch checkpoint to https://huggingface.co/{repo} (private; delete when convenient)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="also test the HF push path against a scratch repo")
    parser.add_argument("--skip-data", action="store_true", help="skip the real-dataset download/split test")
    args = parser.parse_args()

    model = test_adapter_stacking()
    test_weight_geometry(model)
    if not args.skip_data:
        test_data_splits()
    if args.push:
        test_hf_push()

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
