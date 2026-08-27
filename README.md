# SeqFT + PLRS v2 — corrected pipeline

Corrected reimplementation of the SeqFT+PLRS training pipeline from
[mwp_ai4math_icml](https://github.com/gadmin7/mwp_ai4math_icml) (AI4MATH@ICML 2025),
fixing two bugs found in that repo's notebooks:

1. **PLRS didn't actually stack adapters.** The original code called `get_peft_model()`
   on a model that was already a `PeftModel`, silently discarding each prior stage's
   LoRA weights instead of freezing and stacking them as the paper's Figure 1 depicts.
   Fixed in [`src/train_stage.py`](src/train_stage.py) using peft's named multi-adapter
   API (`add_adapter` / `set_adapter` / explicit freezing).
2. **The test set doubled as the validation set.** The official 5000-item MATH test
   split — later reported as the paper's benchmark numbers — was passed directly as
   `eval_dataset` with `load_best_model_at_end`, so checkpoint selection was guided by
   the same data used for final evaluation. Fixed in [`src/data.py`](src/data.py): a
   dedicated 5%-of-train, level-stratified validation split now handles early stopping;
   the test set is only ever touched once, by [`src/evaluate.py`](src/evaluate.py),
   after training is fully done.

Scope of this pass: **LLaMA-3.2-1B, all 6 baselines**. 3B/Qwen backbones are a follow-up
once this is validated (see the plan this repo was built from).

## Layout

```
src/
  data.py               MATH loading + stratified val split + replay/level filters
  lora_schedule.py       single source of truth for every baseline's rank/alpha schedule
  train_stage.py          per-stage adapter preparation (the core bug fix)
  pipeline.py              5-stage training loop, checkpoint save + HF push
  evaluate.py               generation-based EM on the held-out test set only
  analysis/weight_geometry.py   per-layer update norms + overwrite heatmap across stages
configs/baseline{1..6}.yaml    hyperparams per baseline
scripts/smoke_test.py           local, CPU-only, no CUDA/bitsandbytes required
scripts/run_baseline.py          real training entrypoint (needs a CUDA GPU)
cloud/jarvislabs_setup.sh         instance bootstrap for JarvisLabs
```

## Checkpoint naming

`{hf_org}/mwp-v2-{model_tag}-{baseline_id}-stage{n}`, e.g. `GT1999/mwp-v2-llama1b-b6-stage5`.
Each repo holds exactly one stage's adapter (never a merged model), with a model card
recording rank/alpha, cumulative train size, replay policy, validation seed, and the
code commit that produced it.

## Running

**1. Local verification first, always** — no GPU needed, runs in about a minute:

```bash
python scripts/smoke_test.py --push        # --push also exercises the HF upload path
python scripts/dry_run.py --baseline b6    # drives the REAL pipeline on a tiny model
```

`smoke_test.py` checks the adapter/data logic in isolation: that each stage stays
**active in the forward pass** while earlier stages are frozen, that train/val/test are
pairwise disjoint, and that the weight-geometry tooling and HF naming work.

`dry_run.py` drives `run_baseline` itself, so it covers what unit-level checks cannot:
`TrainingArguments` construction against the installed transformers, the `Trainer` loop,
early stopping, best-checkpoint restore on a multi-adapter model, and the on-disk
checkpoint layout. Run it on the cloud box too before starting a multi-hour job —
`cloud/jarvislabs_setup.sh` does this automatically.

**2. Cloud GPU** (JarvisLabs, 1× A100 80GB / 28 vCPU / 112GB RAM / 100GB storage —
more than this 1B model at r=256 strictly needs, but the extra memory bandwidth over
the 40GB tier speeds up the generation-heavy eval tail; batch sizes below are tuned
for this instance, halve them on the 40GB/16-vCPU tier):

```bash
bash cloud/jarvislabs_setup.sh
export HF_TOKEN=$(hf auth token)
python scripts/run_baseline.py --config configs/baseline1.yaml   # cheapest, run first
python scripts/run_baseline.py --config configs/baseline2.yaml
# then baseline3..6 once 1/2 are confirmed working end-to-end
```

**3. Evaluate + analyze** once a baseline's stages are all trained/pushed — see
`src/evaluate.py:evaluate_baseline` (generation-based EM on the untouched test set)
and `src/analysis/weight_geometry.py` (what each stage's update actually touches,
and how much later stages reinforce vs. partially cancel earlier ones).
