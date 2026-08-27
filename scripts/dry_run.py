#!/usr/bin/env python3
"""End-to-end dry run of the REAL pipeline on CPU with a tiny model.

scripts/smoke_test.py checks the adapter/data logic in isolation. This instead
drives src.pipeline.run_baseline itself -- the same function the cloud run calls --
so it exercises the parts nothing else does:

  - build_training_args against the installed transformers API
  - Trainer construction, the training loop, eval, and early stopping
  - load_best_model_at_end against a MULTI-ADAPTER PeftModel (a real risk: Trainer
    checkpointing was written for single-adapter models)
  - the per-stage save path and the nested <stage_N>/ checkpoint layout

Fake data is sized so eval/save steps actually fire, as they would on real data.
No GPU, no bitsandbytes, no network pushes.

Run: python scripts/dry_run.py [--baseline b6]
"""

import argparse
import os
import sys
import tempfile

from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import MathSplits
from src.lora_schedule import BASELINES, get_baseline
from src.pipeline import run_baseline
from src.train_stage import stage_adapter_name

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def fake_splits(n_per_level: int = 60) -> MathSplits:
    def build(n_each, tag):
        rows = {"problem": [], "solution": [], "level": [], "type": []}
        for level in range(1, 6):
            for i in range(n_each):
                rows["problem"].append(f"[{tag} L{level} #{i}] What is {i} + {level}?")
                rows["solution"].append(f"The answer is $\\boxed{{{i + level}}}$.")
                rows["level"].append(f"Level {level}")
                rows["type"].append("Algebra")
        return Dataset.from_dict(rows)

    return MathSplits(train=build(n_per_level, "tr"), val=build(8, "va"), test=build(2, "te"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="b6", choices=sorted(BASELINES))
    parser.add_argument("--n-per-level", type=int, default=60)
    args = parser.parse_args()

    spec = get_baseline(args.baseline)
    splits = fake_splits(args.n_per_level)
    print(f"dry run: {spec.id} ({spec.name}), {spec.num_stages} stage(s), "
          f"train={len(splits.train)} val={len(splits.val)}")

    with tempfile.TemporaryDirectory() as tmp:
        run_baseline(
            baseline_id=spec.id,
            base_model_id=TINY_MODEL,
            model_tag="tiny",
            splits=splits,
            output_dir=tmp,
            hf_org=None,
            push_to_hub=False,      # no network writes
            quantize=False,         # no CUDA / bitsandbytes
            hf_token=None,
            seed=42,
            batch_size=4,
            dataloader_num_workers=0,
            map_num_proc=1,
        )

        print("\n=== verifying on-disk checkpoints ===")
        for stage in range(1, spec.num_stages + 1):
            adapter_dir = os.path.join(tmp, f"{spec.id}-stage{stage}", stage_adapter_name(stage))
            for fname in ("adapter_model.safetensors", "adapter_config.json", "README.md"):
                path = os.path.join(adapter_dir, fname)
                assert os.path.exists(path), f"missing {path}"
            import json

            cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
            expected_r = spec.stage_config(stage).r
            assert cfg["r"] == expected_r, f"stage {stage}: saved r={cfg['r']}, expected {expected_r}"
            print(f"  stage {stage}: r={cfg['r']} alpha={cfg['lora_alpha']} OK")

        log_path = os.path.join(tmp, f"{spec.id}-log_history.json")
        assert os.path.exists(log_path), f"missing {log_path}"
        print(f"  log history written OK")

    print("\nDry run passed.")


if __name__ == "__main__":
    main()
