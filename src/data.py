"""MATH dataset loading with a genuine held-out validation split.

Fixes the original pipeline's core data-integrity bug: `dataset['test']` (the
5000-item official test split, later reported as the paper's benchmark numbers)
was passed directly as `eval_dataset` to SFTTrainer with `load_best_model_at_end`
and early stopping -- i.e. checkpoint selection was guided by the test set itself.

Here, `test` is loaded but never touched until `evaluate.py` runs, once, after a
baseline's final stage is fully trained. Early stopping uses a dedicated `val`
split carved out of `train` only.
"""

import random
from dataclasses import dataclass

from datasets import Dataset, load_dataset

VAL_FRACTION = 0.05
LEVELS = [1, 2, 3, 4, 5]


@dataclass
class MathSplits:
    train: Dataset
    val: Dataset
    test: Dataset


def _level_int(example) -> int:
    return int(example["level"].split()[-1])


def load_math_splits(seed: int = 42) -> MathSplits:
    raw = load_dataset("Maxwell-Jia/MATH")
    train_all = raw["train"].filter(lambda x: x["level"] != "Level ?")
    test = raw["test"].filter(lambda x: x["level"] != "Level ?")

    rng = random.Random(seed)
    val_indices, train_indices = [], []
    for level in LEVELS:
        level_idx = [i for i, x in enumerate(train_all) if _level_int(x) == level]
        rng.shuffle(level_idx)
        n_val = max(1, round(len(level_idx) * VAL_FRACTION))
        val_indices.extend(level_idx[:n_val])
        train_indices.extend(level_idx[n_val:])

    train = train_all.select(sorted(train_indices))
    val = train_all.select(sorted(val_indices))

    assert_disjoint(train, val, test)
    return MathSplits(train=train, val=val, test=test)


def _problem_ids(ds: Dataset) -> set:
    # MATH has no native id column; (problem, solution) text pair is a stable identity key.
    return set(zip(ds["problem"], ds["solution"]))


def assert_disjoint(train: Dataset, val: Dataset, test: Dataset) -> None:
    train_ids, val_ids, test_ids = _problem_ids(train), _problem_ids(val), _problem_ids(test)
    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids
    assert not tv, f"train/val overlap: {len(tv)} problems"
    assert not tt, f"train/test overlap: {len(tt)} problems"
    assert not vt, f"val/test overlap: {len(vt)} problems"


def stage_slice(ds: Dataset, stage_level: int, replay: bool) -> Dataset:
    """D_i for a given stage: cumulative (replay=True) or level-only (replay=False)."""
    if replay:
        return ds.filter(lambda x: _level_int(x) <= stage_level)
    return ds.filter(lambda x: _level_int(x) == stage_level)


def log_split_sizes(splits: MathSplits) -> str:
    lines = ["level  train  val  test"]
    for level in LEVELS:
        n_train = sum(1 for x in splits.train if _level_int(x) == level)
        n_val = sum(1 for x in splits.val if _level_int(x) == level)
        n_test = sum(1 for x in splits.test if _level_int(x) == level)
        lines.append(f"{level:>5}  {n_train:>5}  {n_val:>3}  {n_test:>4}")
    return "\n".join(lines)
