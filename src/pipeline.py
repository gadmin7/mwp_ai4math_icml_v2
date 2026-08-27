"""Orchestrates the multi-stage training loop for one baseline.

Checkpoint naming: `{hf_org}/mwp-v2-llama1b-{baseline_id}-stage{n}` -- each repo
holds exactly one stage's adapter (see train_stage.prepare_stage_model), never a
merged model, so stage checkpoints stay small and separately inspectable.
"""

import dataclasses
import json
import math
import os
import subprocess

import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainerCallback, TrainingArguments

from src.data import MathSplits, assign_stages, exposure_weighted, log_split_sizes, stage_slice
from src.lora_schedule import BaselineSpec, get_baseline
from src.train_stage import prepare_stage_model, stage_adapter_name

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are an expert math assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "Solve the following math problem: {problem} Show all intermediate steps and please "
    "mandatorily include the final answer in LaTeX format in a box like \\boxed{{}}."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{solution}<|eot_id|>"
)


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return "unknown"


def dataset_for_stage(splits: MathSplits, baseline: BaselineSpec, stage: int) -> tuple:
    if baseline.num_stages == 1:
        train = exposure_weighted(splits.train) if baseline.exposure_weighted else splits.train
        return train, splits.val
    return stage_slice(splits.train, stage, baseline.replay), stage_slice(splits.val, stage, baseline.replay)


def partitioned_splits(splits: MathSplits, baseline: BaselineSpec, seed: int) -> MathSplits:
    """Attach the `stage` column both train and val are sliced by.

    Single-stage arms need no partition. The test split is never partitioned -- it is
    untouched until evaluate.py.
    """
    if baseline.num_stages == 1:
        return splits
    return MathSplits(
        train=assign_stages(splits.train, baseline.partition, seed, baseline.num_stages),
        val=assign_stages(splits.val, baseline.partition, seed, baseline.num_stages),
        test=splits.test,
    )


def make_tokenizer(base_model_id: str, token: str = None) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(base_model_id, token=token)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def make_preprocess_fn(tokenizer, max_length: int = 1024):
    def _fn(examples):
        texts = [
            PROMPT_TEMPLATE.format(problem=p, solution=s)
            for p, s in zip(examples["problem"], examples["solution"])
        ]
        return tokenizer(texts, truncation=True, max_length=max_length)

    return _fn


def repo_name(hf_org: str, model_tag: str, baseline_id: str, stage: int) -> str:
    return f"{hf_org}/mwp-v2-{model_tag}-{baseline_id}-stage{stage}"


WARMUP_RATIO = 0.1
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
EARLY_STOPPING_PATIENCE = 2


class BestAdapterCallback(TrainerCallback):
    """Early stopping + best-checkpoint restore, done ourselves.

    Trainer's own `load_best_model_at_end` SILENTLY FAILS on this setup: because we
    use *named* adapters, PeftModel.save_pretrained nests weights under
    `checkpoint-N/<adapter_name>/adapter_model.safetensors`, while Trainer looks for
    them at `checkpoint-N/` root. It logs "Could not locate the best model ..." and
    carries on, leaving the LAST weights in place rather than the best ones -- which
    silently contradicts the documented protocol (best model by eval_loss).

    So we track eval_loss ourselves, snapshot the trainable adapter's tensors to CPU
    whenever it improves, restore them at train end, and drive early stopping off the
    same counter. Only the current stage's adapter is trainable, so the snapshot is
    just that one adapter (~700MB at r=256 for a 1B model; earlier stages are frozen
    and by definition unchanged).
    """

    def __init__(self, model, patience: int = EARLY_STOPPING_PATIENCE, min_delta: float = 0.0):
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.num_bad = 0
        self.snapshot = None
        self.best_step = None

    def _trainable_state(self):
        return {
            name: param.detach().clone().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return control
        loss = metrics["eval_loss"]
        if loss < self.best - self.min_delta:
            self.best = loss
            self.num_bad = 0
            self.snapshot = self._trainable_state()
            self.best_step = state.global_step
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                print(f"[early stopping] no eval_loss improvement in {self.patience} evals; "
                      f"stopping at step {state.global_step}")
                control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.snapshot is None:
            print("[best model] no eval improvement recorded; keeping final weights")
            return control
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, saved in self.snapshot.items():
                params[name].copy_(saved.to(params[name].device))
        print(f"[best model] restored adapter from step {self.best_step} (eval_loss {self.best:.4f})")
        self.snapshot = None
        return control


def build_training_args(
    output_dir: str, batch_size: int, n_train_examples: int, quantize: bool, dataloader_num_workers: int,
) -> TrainingArguments:
    """Construct TrainingArguments in a way that survives the transformers API churn.

    Two fields this pipeline depends on were renamed/removed across versions:
      - `evaluation_strategy` -> `eval_strategy`   (renamed in 4.46, gone in 5.x)
      - `warmup_ratio`        -> removed in 5.x; only `warmup_steps` survives
    Pinning transformers to the old window would force a downgrade on the cloud
    image's CUDA-matched build, so we adapt to whatever is installed instead.
    """
    supported = {f.name for f in dataclasses.fields(TrainingArguments)}

    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=100,
        eval_steps=150,
        report_to="none",
        dataloader_num_workers=dataloader_num_workers,
        # Best-model restore and early stopping are handled by BestAdapterCallback,
        # not by Trainer: its load_best_model_at_end cannot find nested named-adapter
        # weights and silently keeps the last checkpoint instead. Trainer-side
        # checkpointing is therefore off -- we save each stage's adapter ourselves
        # once training completes, and mid-stage checkpoints were never resumable here.
        save_strategy="no",
    )

    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "steps"

    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = WARMUP_RATIO
    else:
        steps_per_epoch = max(1, math.ceil(n_train_examples / batch_size))
        kwargs["warmup_steps"] = max(1, int(WARMUP_RATIO * steps_per_epoch * NUM_EPOCHS))

    if quantize:
        # 8-bit paged optimizer + fp16 only make sense on a real CUDA device.
        kwargs["fp16"] = True
        kwargs["optim"] = "paged_adamw_8bit"
    else:
        kwargs["optim"] = "adamw_torch"

    unsupported = sorted(set(kwargs) - supported)
    if unsupported:
        print(f"[training args] dropping unsupported on this transformers build: {unsupported}")
        kwargs = {k: v for k, v in kwargs.items() if k in supported}

    return TrainingArguments(**kwargs)


def write_model_card(path: str, baseline: BaselineSpec, stage: int, train_size: int, seed: int) -> None:
    cfg = baseline.stage_config(stage)
    card = f"""---
tags: [mwp-v2, seqft, plrs, math-word-problems]
---
# {baseline.id} stage {stage} -- {baseline.name}

- LoRA rank / alpha: {cfg.r} / {cfg.lora_alpha}  (scaling: {"alpha/sqrt(r), rsLoRA" if cfg.use_rslora else "alpha/r"})
- Full rank schedule: {" -> ".join(str(x) for x in baseline.ranks)}
- Replay (cumulative levels): {baseline.replay}
- Stage partition: {baseline.partition}{" (exposure-weighted single pass)" if baseline.exposure_weighted else ""}
- Cumulative train examples this stage: {train_size}
- Validation split seed: {seed} (5% of train, stratified by level; test set never used for selection)
- Code commit: {_git_commit_hash()}
"""
    with open(os.path.join(path, "README.md"), "w") as f:
        f.write(card)


def run_baseline(
    baseline_id: str,
    base_model_id: str,
    model_tag: str,
    splits: MathSplits,
    output_dir: str,
    hf_org: str = None,
    push_to_hub: bool = True,
    quantize: bool = True,
    hf_token: str = None,
    seed: int = 42,
    batch_size: int = 4,
    dataloader_num_workers: int = 8,
    map_num_proc: int = 8,
) -> dict:
    """dataloader_num_workers/map_num_proc default to 8 -- tuned for a ~28 vCPU
    instance (JarvisLabs A100 80GB tier); drop to 2-4 on a 16-vCPU box.
    """
    # Plain transformers Trainer, not trl's SFTTrainer: we pre-tokenize in
    # make_preprocess_fn and supply our own collator, so SFTTrainer's extras
    # (dataset_text_field / packing / peft_config) are all unused -- and its
    # peft_config path re-wraps an already-PEFT model, the exact bug this repo
    # exists to fix. One less dependency and one less API to track.
    from transformers import Trainer

    baseline = get_baseline(baseline_id)
    print(log_split_sizes(splits))
    splits = partitioned_splits(splits, baseline, seed)
    if baseline.num_stages > 1:
        print(f"stage partition strategy: {baseline.partition}")
    if baseline.exposure_weighted:
        print("train set is exposure-weighted (each example repeated 6-level times)")

    tokenizer = make_tokenizer(base_model_id, token=hf_token)
    preprocess = make_preprocess_fn(tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = None
    all_log_history = {}
    for stage in range(1, baseline.num_stages + 1):
        print(f"\n=== {baseline_id} stage {stage}/{baseline.num_stages} ===")
        train_raw, val_raw = dataset_for_stage(splits, baseline, stage)
        train_tok = train_raw.map(preprocess, batched=True, num_proc=map_num_proc)
        val_tok = val_raw.map(preprocess, batched=True, num_proc=map_num_proc)

        prev_path = None
        if stage > 1 and not push_to_hub:
            prev_stage_name = stage_adapter_name(stage - 1)
            prev_path = os.path.join(output_dir, f"{baseline_id}-stage{stage - 1}", prev_stage_name)

        model = prepare_stage_model(
            stage=stage,
            baseline=baseline,
            base_model_id=base_model_id,
            prev_model=model,
            prev_adapter_path=prev_path,
            quantize=quantize,
            token=hf_token,
        )

        stage_out = os.path.join(output_dir, f"{baseline_id}-stage{stage}")
        training_args = build_training_args(
            output_dir=stage_out,
            batch_size=batch_size,
            n_train_examples=len(train_tok),
            quantize=quantize,
            dataloader_num_workers=dataloader_num_workers,
        )
        trainer = Trainer(
            model=model,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            args=training_args,
            data_collator=collator,
            callbacks=[BestAdapterCallback(model, patience=EARLY_STOPPING_PATIENCE)],
        )
        trainer.train()
        all_log_history[f"stage_{stage}"] = trainer.state.log_history

        adapter_name = stage_adapter_name(stage)
        # peft nests a multi-adapter model's save under save_directory/<adapter_name>/ --
        # that nested folder is the actual flat checkpoint we want at the repo root.
        # save_embedding_layers=False: we never resize the vocab (make_tokenizer maps
        # pad_token onto the existing eos_token rather than adding one), so the
        # embedding and lm_head are byte-identical to the base model's. Saving them
        # would add ~1GB of redundant weights to every stage checkpoint.
        model.save_pretrained(stage_out, selected_adapters=[adapter_name], save_embedding_layers=False)
        adapter_dir = os.path.join(stage_out, adapter_name)
        tokenizer.save_pretrained(adapter_dir)
        write_model_card(adapter_dir, baseline, stage, len(train_raw), seed)

        if push_to_hub and hf_org:
            repo = repo_name(hf_org, model_tag, baseline_id, stage)
            from huggingface_hub import HfApi

            HfApi().create_repo(repo, exist_ok=True, token=hf_token)
            HfApi().upload_folder(repo_id=repo, folder_path=adapter_dir, token=hf_token)
            print(f"pushed {repo}")

    with open(os.path.join(output_dir, f"{baseline_id}-log_history.json"), "w") as f:
        json.dump(all_log_history, f, indent=2)

    return all_log_history
