# difficulty-geometry

**Does training on easy maths problems help a model learn hard ones?**

Yes — but not along the ladder the difficulty labels describe. Transfer tracks how close
two problem sets sit in the model's own **gradient geometry**, and that geometry has
little to do with human difficulty ratings.

Base Llama-3.2-1B · MATH · LoRA r=32 · everything reproducible, most of it on a laptop.

---

## The three findings

### 1. MATH difficulty labels mostly track solution length

Across the 5,000-problem test set, per-token loss barely moves while accuracy collapses:

| | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| solution length (chars) | 226 | 325 | 440 | 535 | **804** |
| per-token loss | 0.930 | 0.947 | 0.927 | 0.939 | **0.922** |
| exact match | 12.36% | 5.03% | 3.01% | 1.32% | **0.53%** |

Exact match needs every token right, so with roughly constant per-token accuracy `p` and
length `n`, success behaves like `p^n`. Flat `p`, growing `n`, and accuracy falls off a
cliff. **Length moves 3.55x; per-token loss moves 0.99x.**

### 2. Transfer is real, and strictly local

Train on Level 1 alone, then measure how much gradient remains on every level — crediting
transfer only where the drop exceeds a shuffled-text control:

```
L2   +31.8 pp     strong
L3   +26.2 pp
L4   +10.4 pp
L5   -16.7 pp     below the control -- no transfer at all
```

Transfer tracked **subspace proximity** and stopped where proximity stopped. Easy problems
move you further along a direction you were already going; they do not open directions you
were not already heading in.

### 3. Curriculum ordering helps slightly, and costs more than it gives

Four compute- and capacity-matched arms separate data *order* from the per-level *exposure*
that cumulative replay silently forces (a Level-1 example is replayed in all five stages,
a Level-5 example in one):

| arm | order | exposure | test loss |
|---|---|---|---|
| `jointu` | none | uniform | **0.9323** |
| `staged` | easy → hard | 5:4:3:2:1 | 0.9350 |
| `jointw` | none | 5:4:3:2:1 | 0.9390 |
| `staged_nr` | easy → hard | uniform, no replay | 0.9607 |

```
ORDER      staged - jointw     -0.0040   ordering helps, slightly
EXPOSURE   jointw - jointu     +0.0067   the 5:4:3:2:1 skew hurts, more
NET        staged - jointu     +0.0027   staging loses overall
REPLAY     staged_nr - staged  +0.0257   the largest term by far
```

Mathematically, cumulative staging is **importance sampling with the importance weights
left out** — you get the variance reduction and pay for it with a permanent bias toward
easy examples. Its implicit objective correlates **−0.03** with the test set's composition.

---

## The methodological result, which may matter more

Any similarity metric on high-dimensional weights needs a floor built from the *same inputs
with the structure of interest destroyed*. We shuffle the words inside each problem: same
tokens, same length, no mathematics.

```
cross-level gradient overlap    0.51 - 0.59
shuffled-text floor             0.39            <- word salad scores this much
random chance                   0.018
interpretable range             ~0.10 of a measured 0.55
```

Most of what looks like shared task structure is present in nonsense. **And the floor
depends on your prompt** — same weights, same problems, only the wrapper changed:

```
plain  "Problem: … Solution: …"   5.3% boilerplate   floor 0.4501
chat template with system prompt  38.5% boilerplate  floor 0.8990
```

A weight matrix's gradient is a sum of outer products over tokens, so its directions are
the directions of the *activations* — and tokens appearing verbatim in every example
contribute the same component to every gradient. A long fixed preamble is shared content
in every sample, and it inflates any similarity measured on top of it.

---

## Start here

[`notebooks/gradient_geometry_lab.ipynb`](notebooks/gradient_geometry_lab.ipynb) — eight
observations, **nothing trained**, runs on a free T4 or a laptop CPU. Reproduces every
result above plus a hand-written LoRA showing `B = 0` is exactly a no-op. Two executed
reference runs are included (`run_qwen05b_n8`, `run_qwen05b_n64`).

```bash
python scripts/gradient_overlap.py --n-per-level 64 --k 32   # subspaces + shuffled floor
python scripts/transfer_test.py    --n-train 400             # does L1 help L2..L5?
python scripts/run_baseline.py --config configs/jointu.yaml  # one training arm
python scripts/test_loss.py --configs configs/*.yaml         # per-level loss on TEST
```

## Layout

```
src/            data splits, LoRA schedules, per-stage training, evaluation
scripts/        the experiments above, plus dry_run.py and smoke_test.py
configs/        one YAML per arm; staged / staged_nr / jointw / jointu are the four arms
notebooks/      the teaching lab and two executed runs
REFERENCES.md   every paper this work draws on, with what each was used for
```

## Caveats, stated plainly

- **Single seed.** The order and exposure terms are 0.003–0.007. Only the replay effect
  (0.026) and the transfer asymmetry sit comfortably outside plausible seed noise.
- **Backward transfer is positive**, so the no-replay arm is *not* catastrophic
  forgetting. Every level ends better than when it was learned; Level 1 declines 0.034
  from its peak. Compute BWT before calling a declining curve forgetting.
- **Scale is untested.** Everything here is 0.5B–1B. Related work finds structural
  coherence rises sharply with model size, so these results may not survive at frontier
  scale.
- The `plain` template has **no EOS token**, so generations do not terminate; exact-match
  numbers are depressed for every arm equally. `append_eos` exists in the config and is
  off by default to keep the arms comparable.

## Validation

The same four arms independently reproduce DART-Math's central finding — easy-biased
exposure underperforms uniform — at 1B scale. The nulls therefore come from an instrument
demonstrated to detect a published effect, not one too noisy to see anything.
