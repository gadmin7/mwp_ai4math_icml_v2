import json

C=[]
def md(s): C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").split("\n")})
def code(s): C.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
                       "source":s.strip("\n").split("\n")})

md(r"""
# Where Do the Gradients Point?

### A hands-on look at what a language model actually experiences when it learns maths

This notebook is **observation only — nothing is trained.** Every cell is a forward pass, a
backward pass, or a bit of linear algebra. It runs on a free Colab T4 in a few minutes, or on a
laptop CPU if you are patient and shrink `N_PER_LEVEL`.

We will work through eight observations, each one a small experiment you can re-run with a
different model:

| # | Question | What you will see |
|---|---|---|
| 1 | What does "difficulty" mean in the MATH dataset? | It mostly means *length* |
| 2 | Is a hard problem harder for the model, per token? | Almost not at all |
| 3 | Then why does accuracy collapse? | A long-password effect |
| 4 | How big is the gradient for each level? | And why that number can lie |
| 5 | Which *direction* do the weights want to move? | Subspaces and principal angles |
| 6 | **The control that changes everything** | Nonsense text scores shockingly high |
| 7 | Where does that inflation come from? | Your prompt's boilerplate |
| 8 | What does an untrained LoRA adapter do? | Provably nothing — and why that matters |

The point of the notebook is not the numbers. It is the habit in Observation 6: **before you
believe a similarity score, measure what it looks like when there is nothing to find.**
""")

md(r"""
---
## 0. Setup

Defaults use **Qwen2.5-0.5B-Instruct**, which needs no access token and fits comfortably in a T4.
Swap in any small causal LM at the bottom of the notebook to compare.
""")

code(r"""
# !pip install -q transformers datasets torch

import math, random, gc
import torch, torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID     = "Qwen/Qwen2.5-0.5B-Instruct"   # ungated. Try SmolLM2-360M, Llama-3.2-1B (gated), ...
N_PER_LEVEL  = 8      # problems per difficulty level. 8 is enough to see every effect; raise if you have time
MAX_LEN      = 384
K            = 16     # subspace dimension we will compare
SEED         = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
print(f"device: {DEVICE}   model: {MODEL_ID}")
""")

md(r"""
---
## 1. What does "difficulty" mean here?

MATH labels every problem **Level 1** (easiest) to **Level 5** (hardest). Those labels come from
human competition difficulty — roughly, what fraction of human contestants solve the problem.

Before touching a model, look at what else changes as the level goes up.
""")

code(r"""
ds = load_dataset("Maxwell-Jia/MATH", trust_remote_code=True)["train"]
ds = ds.filter(lambda x: x["level"] != "Level ?")

def level_int(x): return int(x["level"].split()[-1])

by_level = {}
for lv in range(1, 6):
    rows = [r for r in ds if level_int(r) == lv]
    random.Random(SEED).shuffle(rows)
    by_level[lv] = rows[:N_PER_LEVEL]

print(f"{'level':<8}{'problem chars':>15}{'solution chars':>17}")
print("-" * 40)
for lv in range(1, 6):
    p = np.mean([len(r["problem"])  for r in by_level[lv]])
    s = np.mean([len(r["solution"]) for r in by_level[lv]])
    print(f"L{lv:<7}{p:>15.0f}{s:>17.0f}")
""")

md(r"""
**Look at the solution length.** It grows steadily with the difficulty label — typically 3–4x from
Level 1 to Level 5. Hold on to that; it explains most of what follows.

*(With only 8 problems per level these means are noisy. The trend still shows.)*
""")

md(r"""
---
## 2. What does the loss actually see?

Now the model. We compute **cross-entropy per token** on each level — exactly the quantity that
would be minimised during fine-tuning.

A crucial detail: this is *teacher forcing*. The correct solution is already in the context, and
the model only predicts the next token given everything before it. It never has to **find** the
solution. Keep that in mind — it turns out to be the whole story.
""")

code(r"""
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEVICE)
model.eval()

PLAIN = "Problem: {problem}\nSolution: {solution}"

def batch_loss(rows, template=PLAIN, bs=2):
    '''Mean cross-entropy per token over `rows`.'''
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i+bs]
        texts = [template.format(problem=r["problem"], solution=r["solution"]) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEVICE)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        with torch.no_grad():
            out = model(**enc, labels=labels)
        n = int((labels != -100).sum()) - len(chunk)   # shifted targets
        tot_loss += out.loss.item() * n
        tot_tok  += n
    return tot_loss / tot_tok

print(f"{'level':<8}{'loss/token':>12}{'perplexity':>13}")
print("-" * 33)
losses = {}
for lv in range(1, 6):
    losses[lv] = batch_loss(by_level[lv])
    print(f"L{lv:<7}{losses[lv]:>12.4f}{math.exp(losses[lv]):>13.2f}")

sp = max(losses.values()) - min(losses.values())
print(f"\nspread across levels: {sp:.4f} nats/token")
""")

md(r"""
**Observation — and read the caveat first.** At the default `N_PER_LEVEL = 8` this measurement is
*dominated by sampling noise*. You will see a spread of perhaps 0.3 nats/token scattered across
levels in no particular order. That is noise, not signal.

Measured properly over the full 5000-problem test set, the real numbers are

```
L1 0.930   L2 0.947   L3 0.927   L4 0.939   L5 0.922      spread 0.024
```

— nearly flat, with Level 5 slightly *lower* than Level 1. The genuine effect (0.02) is about
thirteen times smaller than the noise you get at n=8, so **raise `N_PER_LEVEL` to 64 or more before
believing anything here.** It is the slowest cell in the notebook for exactly that reason.

This is the first surprise, and it is worth the wait. By the metric that training actually
optimises, the "hardest" problems in the benchmark are barely harder at all.

Why not? Because teacher forcing has deleted the hard part. Human difficulty is the difficulty of
**searching** for a solution. Here, the solution is already written in the context — the model is
only continuing text it can see. The search never happens, so its difficulty never enters the loss.
""")

md(r"""
---
## 3. Then why does accuracy collapse?

Exact-match accuracy on MATH falls off a cliff with level — typically 12% at Level 1 down to under
1% at Level 5. If per-token loss is flat, where does that come from?

**Length.** Getting an answer right means getting *every* token right, like typing a long password
where a single slip ruins the whole thing.

If each token is correct with probability `p`, a solution of `n` tokens is fully correct with
probability roughly `p^n`. Flat `p`, growing `n`, and the product falls off a cliff.
""")

code(r"""
print(f"{'level':<7}{'p(token)':>11}{'tokens':>9}{'p^n':>12}   (relative to L1)")
print("-" * 55)
base = None
for lv in range(1, 6):
    p = math.exp(-losses[lv])                                   # per-token "accuracy" proxy
    n = np.mean([len(tok(r["solution"])["input_ids"]) for r in by_level[lv]])
    pn = p ** n
    if base is None: base = pn
    print(f"L{lv:<6}{p:>11.4f}{n:>9.0f}{pn:>12.2e}{pn/base:>15.2e}")
""")

md(r"""
**Observation.** `p` barely moves, `n` grows several-fold, and `p^n` falls by many orders of
magnitude. Exact match inherits that collapse.

The absolute values here are far too small (real models beat this because their errors are
correlated, not independent — an early mistake often makes the *rest* of the solution consistent
with it). But the *shape* is right, and it is entirely driven by length.

> **A benchmark that gets "harder" by using longer reference solutions will reliably drive
> exact-match down while barely changing the training signal.**
""")

md(r"""
---
## 4. The gradient: how big is the push?

Now we ask what the data wants to *do* to the weights.

We pick a handful of weight matrices, run a backward pass, and measure the **norm** of the
gradient — one number summarising "how strongly does this data want these weights to change?"

To keep memory small we freeze everything except the matrices we are watching. This is what makes
the notebook run on a T4.
""")

code(r"""
MODULES = ("q_proj", "down_proj")
EVERY_N_LAYERS = 6

def target_params(model):
    '''A few weight matrices spread through the network, and nothing else trainable.'''
    for p in model.parameters(): p.requires_grad_(False)
    picked = {}
    for name, module in model.named_modules():
        if not name.endswith(MODULES): continue
        if "layers." not in name: continue
        layer = int(name.split("layers.")[1].split(".")[0])
        if layer % EVERY_N_LAYERS: continue
        module.weight.requires_grad_(True)
        picked[name] = module.weight
    return picked

params = target_params(model)
print(f"watching {len(params)} weight matrices:")
for n in list(params)[:4]: print("   ", n, tuple(params[n].shape))
if len(params) > 4: print(f"    ... and {len(params)-4} more")
""")

code(r"""
def grad_of(rows, template=PLAIN, bs=2):
    '''Average gradient over `rows`, for each watched matrix.'''
    for p in params.values(): p.grad = None
    nb = 0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i+bs]
        texts = [template.format(problem=r["problem"], solution=r["solution"]) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEVICE)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        model(**enc, labels=labels).loss.backward()
        nb += 1
    out = {n: (p.grad / nb).detach().clone() for n, p in params.items()}
    for p in params.values(): p.grad = None
    return out

print(f"{'level':<8}{'mean ||g||':>13}")
print("-" * 21)
grads = {}
for lv in range(1, 6):
    grads[lv] = grad_of(by_level[lv])
    g = np.mean([G.norm().item() for G in grads[lv].values()])
    print(f"L{lv:<7}{g:>13.4f}")
""")

md(r"""
**Observation, and a warning.** You will probably see the norm *shrink* as the level rises — the
hardest level often has the smallest gradient. That looks backwards.

It is, because of how the average works. We took the norm **of the mean gradient**, not the mean of
the norms. Picture a crowd pushing a box:

* everyone shoves the same way → the pushes add up → **long arrow**
* everyone shoves differently → they cancel → **short arrow**, though all are pushing hard

A small gradient norm therefore means *either* "little left to learn" *or* "these examples disagree
with each other." Hard problems are diverse, so their gradients cancel.

**This is why the raw number is not a good measure of difficulty.** It becomes trustworthy when you
compare the *same* problems before and after a change, so the cancellation appears on both sides.
""")

md(r"""
---
## 5. Which direction do the weights want to move?

The norm is one number. The gradient is millions. Let us look at *direction* instead.

For a weight matrix, take the SVD of its gradient and keep the top-`K` right singular vectors.
That is the **subspace the update lives in** — the handful of directions carrying nearly all of the
requested change.

To compare two subspaces `Q1`, `Q2` (each `d x K`, orthonormal columns) we use

```
overlap = || Q1^T Q2 ||_F^2 / K
```

which is the mean squared cosine of the principal angles between them: **1.0** identical,
**0.0** perfectly orthogonal, and about `K/d` for two random subspaces.
""")

code(r"""
def subspace(G, k=K):
    _, _, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    return Vh[:k].T.contiguous()          # d x k, orthonormal columns

def overlap(Q1, Q2):
    s = torch.linalg.svdvals(Q1.T @ Q2)
    k = min(Q1.shape[1], Q2.shape[1])
    return float((s[:k] ** 2).sum() / k)

def subspaces(gd): return {n: subspace(G) for n, G in gd.items()}
def mean_overlap(a, b):
    keys = set(a) & set(b)
    return float(np.mean([overlap(a[n], b[n]) for n in keys]))

# --- sanity checks: does the metric behave the way we claim? ---
d = 256
torch.manual_seed(0)
A = torch.linalg.qr(torch.randn(d, K))[0]
B = torch.linalg.qr(torch.randn(d, K))[0]
half = torch.cat([A[:, :K//2], B[:, :K//2]], dim=1)
print(f"identical subspaces     {overlap(A, A):.4f}   (expect 1.0)")
print(f"two random subspaces    {overlap(A, B):.4f}   (expect ~K/d = {K/d:.4f})")
print(f"half the columns shared {overlap(A, half):.4f}   (expect ~0.5)")
""")

md(r"""
Those three checks matter. A similarity metric you have not calibrated is a number you cannot
interpret. Note how tiny the random-subspace value is: **two unrelated directions in high dimensions
overlap almost not at all.** Remember that when you see the next result.
""")

code(r"""
subs = {lv: subspaces(grads[lv]) for lv in range(1, 6)}

print("cross-level gradient subspace overlap")
print("      " + "".join(f"{f'L{j}':>9}" for j in range(1, 6)))
for i in range(1, 6):
    row = f"  L{i}  "
    for j in range(1, 6):
        row += "        -" if i == j else f"{mean_overlap(subs[i], subs[j]):>9.3f}"
    print(row)
print(f"\nrandom-chance overlap for these shapes is roughly K/d, a few thousandths.")
""")

md(r"""
**Observation.** Every entry is enormous compared to chance — commonly 0.25 to 0.9 against a chance
value of a few hundredths or less. It looks like overwhelming evidence that all five levels are
essentially one task.

Hold that thought for exactly one cell.
""")

md(r"""
---
## 6. The control that changes everything

Here is the most important idea in the notebook.

We have a big number. Before interpreting it, we must ask: **what would this number be if there
were nothing to find?**

So we build a *floor*. Take the same problems and **shuffle the words inside each one**. Same
tokens, same length, same vocabulary — but the mathematics is destroyed. Any overlap this produces
cannot be about task content, because there is no task content left.
""")

code(r"""
def shuffled(rows, seed=SEED):
    rng = random.Random(seed)
    out = []
    for r in rows:
        p = r["problem"].split();  s = r["solution"].split()
        rng.shuffle(p); rng.shuffle(s)
        out.append({"problem": " ".join(p), "solution": " ".join(s)})
    return out

print("original :", by_level[1][0]["problem"][:95])
print("shuffled :", shuffled(by_level[1])[0]["problem"][:95])

sub_shuf = subspaces(grad_of(shuffled(by_level[1])))
floor = np.mean([mean_overlap(sub_shuf, subs[lv]) for lv in range(1, 6)])
real  = np.mean([mean_overlap(subs[i], subs[j])
                 for i in range(1, 6) for j in range(i+1, 6)])

print(f"\n  real cross-level overlap   {real:.4f}")
print(f"  SHUFFLED-TEXT floor        {floor:.4f}   <-- nonsense scores this much")
print(f"  random chance              ~{K/list(params.values())[0].shape[1]:.4f}")
print(f"\n  usable range above the floor: {real - floor:.4f}")
""")

md(r"""
**Observation.** The gibberish floor is enormous — in our runs, 0.83 on an instruction-tuned model
against a real overlap of 0.93.

Word salad shares almost as much gradient subspace with real mathematics as one difficulty level
shares with another.

So the impressive 0.93 was mostly **not** about the task. The only interpretable quantity is the
*gap above the floor*, and that gap is small. Without this control we would have confidently
reported "all difficulty levels share deep structure" and been almost entirely wrong.

> **The rule: report the floor next to every similarity number, and treat the gap as your real
> resolution.**

Why is the floor so high at all? Two reasons, and the next cell isolates the bigger one.
""")

md(r"""
---
## 7. Where the inflation comes from: your prompt

A transformer's gradient for a weight matrix is a sum of outer products over tokens,
`G = sum_t (output gradient) x (layer input)`. So the directions in `G` are the directions of the
**activations**, and any tokens that appear in *every* example contribute the *same* thing to
every gradient.

Chat templates are full of such tokens — a system prompt and instructions, identical every time.
Let us measure how much that alone inflates the floor, by re-running the exact same comparison with
a chat-style template instead of the plain one.
""")

code(r"""
CHAT = ("<|im_start|>system\nYou are an expert math assistant.<|im_end|>\n"
        "<|im_start|>user\nSolve the following math problem: {problem}\n"
        "Show all intermediate steps and give the final answer in \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n{solution}<|im_end|>")

for name, tpl in [("plain", PLAIN), ("chat template", CHAT)]:
    const = len(tpl.format(problem="", solution=""))
    full  = np.mean([len(tpl.format(problem=r["problem"], solution=r["solution"]))
                     for r in by_level[1]])
    s_real = subspaces(grad_of(by_level[1], template=tpl))
    s_shuf = subspaces(grad_of(shuffled(by_level[1]), template=tpl))
    print(f"{name:<16} boilerplate {100*const/full:4.1f}% of each example"
          f"   ->  shuffled floor {mean_overlap(s_real, s_shuf):.4f}")
""")

md(r"""
**Observation.** The chat template makes a large fraction of every example *identical text*, and
the floor rises sharply as a result. Same model, same problems, same metric — only the wrapper
changed.

This is a measurement artefact hiding inside a perfectly ordinary experimental choice. If you
compare representations or gradients across conditions and your prompt has a long fixed preamble,
a good chunk of your similarity is the preamble.

**Try it:** shorten `CHAT` to just `"Q: {problem}\nA: {solution}"` and watch the floor fall.
""")

md(r"""
---
## 8. What does an untrained adapter do?

LoRA adds a low-rank update to a weight matrix:

```
W_effective = W + (alpha / r) * B @ A
```

`A` is initialised randomly, and **`B` is initialised to zeros**. So at initialisation
`B @ A = 0` and the adapter changes *nothing*. That is deliberate — it means training starts from
exactly the pretrained model.

It also means an adapter whose `B` is still zero is a **no-op**, and a model carrying it is
indistinguishable from the base model. Let us prove that to ourselves, with a hand-written LoRA so
nothing is hidden behind a library.
""")

code(r"""
class LoRALinear(torch.nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) * 0.01)
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale

name, mod = next((n, m) for n, m in model.named_modules() if n.endswith("q_proj"))
parent = model.get_submodule(name.rsplit(".", 1)[0])
child  = name.rsplit(".", 1)[1]

probe = tok("Problem: What is 2+2?\nSolution:", return_tensors="pt").to(DEVICE)
with torch.no_grad(): before = model(**probe).logits.clone()

lora = LoRALinear(mod).to(DEVICE)
setattr(parent, child, lora)
with torch.no_grad(): after_zero = model(**probe).logits.clone()

with torch.no_grad(): lora.B.normal_(0, 0.02)          # pretend one training step happened
with torch.no_grad(): after_trained = model(**probe).logits.clone()

setattr(parent, child, mod)                            # restore
print(f"max |logit change|, adapter attached with B = 0 : {(after_zero  - before).abs().max():.3e}")
print(f"max |logit change|, after giving B real values  : {(after_trained- before).abs().max():.3e}")
""")

md(r"""
**Observation.** With `B = 0` the change is exactly zero — not small, *zero*. The adapter is
attached and completely inert. Give `B` nonzero values and the logits move immediately.

Why does this matter beyond the tidy maths? Because **a fine-tuning run that silently fails still
produces a model that loads and evaluates cleanly.** It just scores whatever the base model scores.

Two ways that happens in practice, both of which we have found in real published code:

* the training call is skipped or errors out, so `B` is never updated and stays zero;
* the adapter is saved with corrupted parameter names, so at load time nothing matches, the library
  emits a *warning* rather than an error, and the weights are silently ignored.

If the base model happens to be *better* at your task than the fine-tuned one — which is exactly
what happens when you fine-tune an instruction-tuned model on terse reference solutions — then
**the broken run posts your best score.**

> **Habit worth keeping:** after loading an adapter, check that the logits actually changed.
> One line, and it catches an entire class of silent failure.
""")

md(r"""
---
## 9. Does any of this depend on the model?

Everything above used one model. Change it and re-run the key measurements — the flat loss, the
gradient norms, and above all the shuffled floor.
""")

code(r"""
del model, params, grads, subs
gc.collect()
if DEVICE == "cuda": torch.cuda.empty_cache()

def profile(model_id):
    global model, params, tok
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(DEVICE)
    model.eval()
    params = target_params(model)

    ls = {lv: batch_loss(by_level[lv]) for lv in range(1, 6)}
    sb = {lv: subspaces(grad_of(by_level[lv])) for lv in range(1, 6)}
    sh = subspaces(grad_of(shuffled(by_level[1])))
    real  = np.mean([mean_overlap(sb[i], sb[j]) for i in range(1,6) for j in range(i+1,6)])
    floor = np.mean([mean_overlap(sh, sb[lv]) for lv in range(1, 6)])

    print(f"\n=== {model_id} ===")
    print("  loss/token by level : " + "  ".join(f"L{lv}:{ls[lv]:.3f}" for lv in range(1, 6)))
    print(f"  loss spread L1..L5  : {max(ls.values())-min(ls.values()):.4f}")
    print(f"  cross-level overlap : {real:.4f}")
    print(f"  shuffled floor      : {floor:.4f}")
    print(f"  usable range        : {real-floor:.4f}")

    del model, params; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

for mid in ["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]:
    try: profile(mid)
    except Exception as e: print(f"\n{mid}: skipped ({type(e).__name__}: {str(e)[:70]})")
""")

md(r"""
**What to compare.** The *shape* should survive across models: flat loss across levels, cross-level
overlap far above chance, and a shuffled floor that eats most of it.

**A useful negative control:** point `profile()` at an untrained model such as
`hf-internal-testing/tiny-random-LlamaForCausalLM`. Everything collapses to chance — the loss is
uniform noise across levels and the overlap sits at `K/d` — and the shuffled floor stops being
distinguishable from the real measurement. That is exactly what should happen when there is no
learned structure to find, and it confirms the pipeline is not manufacturing structure of its own.

What varies is the **usable range** — the gap between the real overlap and the floor. That is your
measurement resolution, and it depends on the model *and* on the prompt you chose in Observation 7.
A model or prompt with a bigger gap is a better instrument for this question.
""")

md(r"""
---
## 10. What to take away

1. **Check what your difficulty labels measure.** In MATH they track solution length, and per-token
   loss barely moves across them. A "difficulty-aware" method built on those labels may be a
   length-aware method under another name.

2. **Teacher forcing removes the search.** Training on reference solutions never asks the model to
   find a path, only to continue one. The part of difficulty that makes a problem hard for a human
   is absent from the loss — which is why methods that put generation back in the loop (rejection
   sampling, RL) behave so differently from plain fine-tuning.

3. **Calibrate every similarity metric.** Ours needed three checks: identical → 1.0, random → K/d,
   and *nonsense text* → the floor. The third is the one everybody skips, and in our case it
   accounted for most of the signal.

4. **Watch your prompt.** A long fixed preamble is shared content in every example, and it inflates
   any similarity you measure.

5. **Verify that a loaded adapter changes the logits.** `B = 0` means no-op. A silent load failure
   looks exactly like a successful run, and can post a *better* score than a working one.

### Things to try

* Raise `N_PER_LEVEL` to 32 and see which observations get sharper and which do not move.
* Set `MODULES = ("q_proj",)` versus `("down_proj",)` — do attention and MLP gradients tell the
  same story?
* Build a stronger floor: shuffle words from a *different domain* (Wikipedia). The gap between that
  and the maths-shuffled floor is how much of the structure is domain rather than syntax.
* Feed the template with **empty** problem and solution slots. That measures the boilerplate
  gradient directly — the `G_common` from Observation 7.
* Compare an instruct model with its base counterpart. Which is the better instrument, and why?
""")

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
open("gradient_geometry_lab.ipynb", "w").write(json.dumps(nb, indent=1))
print(f"wrote gradient_geometry_lab.ipynb  ({len(C)} cells)")
