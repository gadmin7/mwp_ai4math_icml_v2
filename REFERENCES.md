# References

Every paper this work draws on, with what it was used for.

**On verification.** Entries marked ✓ were fetched or searched during this project and the
claim attributed to them was read in the source. Entries marked ~ were cited from memory
in discussion and **not** re-checked — treat those as leads, and verify before citing.

---

## The dataset

| | |
|---|---|
| ✓ | **Measuring Mathematical Problem Solving With the MATH Dataset** — Hendrycks et al., [arXiv:2103.03874](https://arxiv.org/abs/2103.03874) |

The benchmark throughout. Its Level 1–5 labels are the object of the first finding: they
track solution length far more than they track per-token model difficulty.

## Curriculum learning

| | |
|---|---|
| ~ | **Curriculum Learning** — Bengio, Louradour, Collobert, Weston, ICML 2009 |
| ~ | **When Do Curricula Work?** — Wu, Dyer, Neyshabur, ICLR 2021 |

Bengio et al. frame curriculum learning as a **continuation method** — optimise a *smoothed*
version of the objective, then anneal. That is the theoretical warrant people cite, and it
does not cover difficulty *subsetting*: restricting to easy examples changes the objective
rather than smoothing it. Wu et al. find curricula help mainly under limited data or noisy
labels, which is the empirical result our arms are consistent with.

## Difficulty-aware data and selection

| | |
|---|---|
| ✓ | **DART-Math: Difficulty-Aware Rejection Tuning** — Tong et al., [arXiv:2407.13690](https://arxiv.org/abs/2407.13690), NeurIPS 2024 |
| ~ | **RHO-LOSS: Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt** — Mindermann et al., ICML 2022 |
| ✓ | **Probing the Difficulty Perception Mechanism of LLMs** — [arXiv:2510.05969](https://arxiv.org/abs/2510.05969) |

DART-Math is the closest prior work to our exposure arms: vanilla rejection sampling is
biased toward easy queries (zero responses for 51.1% of the hardest MATH training queries),
and both their Uniform and Prop2Diff corrections beat it. **Our `jointw` vs `jointu`
comparison independently reproduces this at 1B scale**, which is what validates our
instrument. RHO-LOSS is the model-relative alternative to static difficulty labels.

## Teacher forcing, search, and why SFT plateaus on maths

| | |
|---|---|
| ~ | **Scheduled Sampling** — Bengio et al., NeurIPS 2015 · **Sequence Level Training (exposure bias)** — Ranzato et al., ICLR 2016 |
| ~ | **STaR: Self-Taught Reasoner** — Zelikman et al., NeurIPS 2022 |
| ✓ | **Scaling Relationship on Learning Mathematical Reasoning (RFT)** — Yuan et al., [arXiv:2308.01825](https://arxiv.org/abs/2308.01825) |
| ✓ | **The Challenge of Teaching Reasoning to LLMs Without RL or Distillation** — Du et al., [arXiv:2507.09850](https://arxiv.org/abs/2507.09850) |

Teacher-forced training never asks the model to *find* a solution path — it is handed one
and asked to continue it. The part of difficulty that makes a problem hard is therefore
absent from the loss, which is our explanation for flat loss against collapsing accuracy.
This is **exposure bias**, and it is the standard motivation for rejection-sampling methods
that put generation back inside the training loop.

> ⚠ 2507.09850 is included as a correction to the record. An automated summary of it during
> this project appeared to confirm two of our claims; the actual abstract is about inducing
> long chain-of-thought from ~20 examples and says no such thing. The summary was
> confabulated from a leading prompt.

## Knowledge structure — the convergent result

| | |
|---|---|
| ✓ | **Do LLMs Exhibit Coherent Knowledge Structures in Mathematical Reasoning? A Perspective from Knowledge Space Theory** — Cui, Do, Sachan, [arXiv:2609.05245](https://arxiv.org/abs/2609.05245) |
| ~ | **Knowledge Spaces** — Doignon & Falmagne (the KST formalism they build on) |

The closest thing to independent confirmation. They test *behaviourally* whether
prerequisite examples scaffold better than similar ones, and find they do not: prerequisite
context gives SG = −0.56 / +0.70 / +0.50 across three models, consistently below same-skill
and semantically-similar retrieval, with random exemplars often matching it. Their
conclusion — benefit comes from "exposure to relevant solution patterns rather than from
activating prerequisite knowledge" — is our result reached from the opposite direction.

Their scale trend is also the strongest caveat on our work: prerequisite coherence rises
sharply with capability (macro PSR 0.353 → 0.925 from Mistral-7B to Qwen3-80B, against
0.942 for human learners), so effects measured at 0.5B–1B may not survive at frontier size.

## Instruction tuning

| | |
|---|---|
| ✓ | **Finetuned Language Models Are Zero-Shot Learners (FLAN)** — Wei et al., [arXiv:2109.01652](https://arxiv.org/abs/2109.01652) |
| ~ | **LIMA: Less Is More for Alignment** — Zhou et al., NeurIPS 2023 |

FLAN's mechanism is *task diversity* — generalisation to unseen tasks improves with the
number of task clusters. Narrow single-task fine-tuning is the opposite operation, so
degradation there is not in tension with FLAN. LIMA's superficial-alignment hypothesis is
the natural competing explanation for such degradation.

## Continual learning and orthogonal subspaces

| | |
|---|---|
| ✓ | **Gradient Projection Memory for Continual Learning** — Saha, Garg, Roy, [arXiv:2103.09762](https://arxiv.org/abs/2103.09762), ICLR 2021 |
| ✓ | **O-LoRA: Orthogonal Subspace Learning for Language Model Continual Learning** — Wang et al., [arXiv:2310.14152](https://arxiv.org/abs/2310.14152), EMNLP Findings 2023 |
| ~ | **Orthogonal Gradient Descent** — Farajtabar et al., AISTATS 2020 |
| ~ | **GEM / A-GEM** — Lopez-Paz & Ranzato, NeurIPS 2017 (also the source of BWT/FWT) |
| ~ | **EWC** — Kirkpatrick et al., PNAS 2017 · **OWM** — Zeng et al., Nature MI 2019 |

GPM uses the same primitive we do — SVD of task-specific structure to obtain a subspace
basis — then projects new-task gradients into the orthogonal complement. **It is evaluated
on "diverse image classification datasets"**, and that matters: vision CL benchmarks
(permuted MNIST, split CIFAR) are constructed so tasks are near-orthogonal. Language tasks
share vocabulary, syntax and objective, so they rarely occupy that regime. O-LoRA is the
main language-side member of this family.

Our measurement says orthogonalisation would be *actively wrong* for MATH difficulty
levels — they overlap at 0.46–0.54 against a 0.39 floor, so forcing them apart would deny
later levels directions they need. BWT, used to check whether our no-replay arm actually
forgot, comes from GEM.

## Task similarity and grouping

| | |
|---|---|
| ✓ | **Efficiently Identifying Task Groupings for Multi-Task Learning (TAG)** — Fifty et al., [arXiv:2109.04617](https://arxiv.org/abs/2109.04617), NeurIPS 2021 |
| ~ | **Taskonomy** — Zamir et al., CVPR 2018 · **Which Tasks Should Be Learned Together?** — Standley et al., ICML 2020 |
| ~ | **PCGrad (gradient surgery)** — Yu et al., NeurIPS 2020 |

TAG's inter-task affinity — how much one task's gradient step changes another's loss — is
the first-order relative of our geometric measure. Ours is cheaper and needs no training;
theirs is more directly predictive.

## Measurement methodology

| | |
|---|---|
| ~ | **Designing and Interpreting Probes with Control Tasks** — Hewitt & Liang, EMNLP 2019 |
| ~ | **Similarity of Neural Network Representations Revisited (CKA)** — Kornblith et al., ICML 2019, and subsequent reliability critiques |
| ~ | **Are Emergent Abilities of LLMs a Mirage?** — Schaeffer, Miranda, Koyejo, NeurIPS 2023 |

Hewitt & Liang's control tasks are the direct precedent for our shuffled-text floor: build
a null from the same inputs with the structure of interest destroyed, and report it beside
the result. The CKA literature is where high similarity scores driven by dominant shared
directions have been most argued over. Schaeffer et al. explain why a thresholded metric
like exact match can collapse while the underlying per-token quantity moves smoothly —
which is the shape of our first finding.

## Optimisation

| | |
|---|---|
| ~ | **Importance sampling for SGD** — Needell, Ward & Srebro, NeurIPS 2014 · Zhao & Zhang, ICML 2015 |
| ~ | **LoRA** — Hu et al., ICLR 2022 · **QLoRA** — Dettmers et al., NeurIPS 2023 · **rsLoRA** — Kalajdzievski, 2023 |

Importance sampling is the rigorous version of what a curriculum does informally: sample
unevenly, then **reweight by 1/p to keep the gradient unbiased**. A curriculum takes the
uneven sampling and omits the correction, which is the cleanest statement of why it
converges toward the wrong objective. QLoRA supplied the hyperparameter prescriptions used
here, though several are specific to 4-bit training and were dropped once quantisation was.

## Cognitive science — invoked, and worth caution

| | |
|---|---|
| ~ | **Expectation-based syntactic comprehension** — Levy, 2008 · **Smith & Levy**, 2013 (surprisal predicts reading time) |
| ~ | **Schrimpf et al.**, PNAS 2021 (LM predictions track neural responses in the language network) |
| ~ | **Fedorenko et al.** on the dissociation between the language network and the multiple-demand system |

Raised when discussing whether next-token difficulty should correspond to human difficulty.
The surprisal results concern the **language network**; mathematical reasoning recruits
multiple-demand regions, so the analogy does not transfer straightforwardly. None of these
were verified during this project and the argument does not depend on them.
