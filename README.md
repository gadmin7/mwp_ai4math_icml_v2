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

Scope of this pass: **LLaMA-3.2-1B**. 3B/Qwen backbones are a follow-up once this is
validated (see the plan this repo was built from).

## Baseline arms

`b1`–`b6` reproduce the paper's configurations. `b7`–`b10` are controls the original
study lacks, defined in [`src/lora_schedule.py`](src/lora_schedule.py):

| id | schedule | replay | cumulative trainable | purpose |
|----|----------|--------|----------------------|---------|
| b1 | 32 | – | 22.5M | direct baseline |
| b2 | 32×5 | no | 112.7M | sequential, no replay |
| b3 | 256→16 | no | 349.4M | SNR + PLRS |
| b4 | 32×5 | yes | 112.7M | full replay, fixed rank |
| b5 | 256→16 | yes | 349.4M | SFR + PLRS |
| b6 | 256→128→64→32→32 | yes | 360.7M | **paper's final recipe** |
| b7 | 32→64→96→128→128 | yes | 315.6M (87.5% of b6) | **expanding** rank |
| b8 | 102×5 | yes | 359.3M (99.6% of b6) | **constant** rank |
| b9 | 256 (single pass) | – | 180.4M | Table 2's heavy baseline |
| b10 | 256 + rsLoRA | – | 180.4M | same, `alpha/√r` scaling |
| b11 | 102×5, **random** partition | yes | 359.3M | is difficulty ordering needed? |
| b12 | 102×5, **reverse** partition | yes | 359.3M | anti-curriculum (hard→easy) |
| b13 | 102, single pass, **5:4:3:2:1 weighted** | – | 71.9M | is staging just reweighting? |

**Why b7/b8.** No arm in b1–b6 *expands*, so "shrinking helps" is never tested against
its opposite; and b4-vs-b6 confounds schedule *shape* with total capacity (b6 has ~3×
the trainable parameters). b7/b8 hold cumulative capacity roughly fixed against b6 so
only the shape differs. Motivation: cumulative training data grows 13.3× across stages
(536→7124) while b6's rank shrinks 8×, so capacity per example falls ~106× — even as
each stage introduces harder, previously unseen material. The paper's single largest
ablation gain also came from shrinking *less* at the final stage (+3.10 EM).

Note too that PLRS was motivated as preventing "overwriting of parameters learned in
earlier levels" — but with adapter stacking implemented correctly, earlier adapters are
frozen and *cannot* be overwritten. Much of the original rationale for shrinking was
compensating for the bug described above.

**Why b9/b10.** The paper reads "single-pass r=256 underperforms r=32" as evidence that
capacity *scheduling* rather than rank budget drives the gains. But `alpha/r` scaling is
known to under-serve large ranks: at r=256 the effective gain is 2.0, versus 32.0 under
rank-stabilized scaling. b9 vs b10 tests whether that result is a scaling artifact.

**Why b11/b12/b13 — does easy→hard actually matter?** Comparing the direct baseline to a
sequential run cannot answer this: it varies stage count, total steps, ordering, per-level
exposure, adapter structure and rank simultaneously. b8/b11/b12 instead fix rank, stage
sizes and total steps, varying only how difficulty is distributed across stages
(`src/data.py::assign_stages` chunks a rank-ordering into blocks whose sizes are the level
counts, so the difficulty arm reproduces the natural level boundaries exactly).

Note that under cumulative replay, ordering and exposure are **coupled** — whatever trains
first is replayed most — so they cannot be varied independently. Measured mean number of
stages each level is trained in:

| arm | L1 | L2 | L3 | L4 | L5 | |
|---|---|---|---|---|---|---|
| b8 difficulty | 5.00 | 4.00 | 3.00 | 2.00 | 1.00 | the 5:4:3:2:1 gradient |
| b11 random | 2.58 | 2.39 | 2.55 | 2.50 | 2.47 | flat |
| b12 reverse | 1.00 | 1.00 | 1.75 | 2.71 | 4.07 | inverted |

Each arm therefore tests an ordering *package* (order plus the exposure it induces).
**b13** supplies the missing piece by isolating the reweighting alone — a single pass over
a dataset where each example is repeated `6 − level` times, which reproduces the staged
pipeline's per-level exposure *and* its exact total step count (17,741 examples = the sum
of the staged cumulative sizes). Together they decompose the effect:

- `b13 ≈ b8` → the gain is just reweighting; staging adds nothing.
- `b11 ≈ b8` → staging helps, but the difficulty ordering does not.
- `b8 > b11, b12, b13` → the curriculum genuinely contributes.

The per-level breakdown of **b8 vs b11** is also the direct test of the paper's *positive
backward transfer* claim: level-1 data receives 5× the gradient exposure of level-5 under
the difficulty curriculum, so the Level-1 gain (42.1→53.3) may be exposure rather than
transfer. b11 flattens exposure, so a Level-1 gain that survives there is genuine transfer.

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

Full launch-to-teardown procedure — SSH key registration, instance settings, run order,
pause/resume caveats — is in **[`cloud/RUNBOOK.md`](cloud/RUNBOOK.md)**.

```bash
bash cloud/jarvislabs_setup.sh
source ~/mwp-venv/bin/activate
export HF_TOKEN=$(hf auth token)
python scripts/run_baseline.py --config configs/baseline1.yaml   # cheapest, run first
python scripts/run_baseline.py --config configs/baseline2.yaml
# then baseline3..6 once 1/2 are confirmed working end-to-end
```

**3. Evaluate + analyze** once a baseline's stages are all trained/pushed — see
`src/evaluate.py:evaluate_baseline` (generation-based EM on the untouched test set)
and `src/analysis/weight_geometry.py` (what each stage's update actually touches,
and how much later stages reinforce vs. partially cancel earlier ones).
