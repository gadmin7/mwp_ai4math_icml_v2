"""Orchestrates the multi-stage training loop for one baseline.

Checkpoint naming: `{hf_org}/mwp-v2-llama1b-{baseline_id}-stage{n}` -- each repo
holds exactly one stage's adapter (see train_stage.prepare_stage_model), never a
merged model, so stage checkpoints stay small and separately inspectable.
"""

import json
import os
import subprocess

from transformers import AutoTokenizer, DataCollatorForLanguageModeling, EarlyStoppingCallback, TrainingArguments

from src.data import MathSplits, log_split_sizes, stage_slice
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
        return splits.train, splits.val
    return stage_slice(splits.train, stage, baseline.replay), stage_slice(splits.val, stage, baseline.replay)


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


def write_model_card(path: str, baseline: BaselineSpec, stage: int, train_size: int, seed: int) -> None:
    cfg = baseline.stage_config(stage)
    card = f"""---
tags: [mwp-v2, seqft, plrs, math-word-problems]
---
# {baseline.id} stage {stage} -- {baseline.name}

- LoRA rank / alpha: {cfg.r} / {cfg.lora_alpha}
- Replay (cumulative levels): {baseline.replay}
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
    from trl import SFTTrainer  # deferred: only the real training path needs trl

    baseline = get_baseline(baseline_id)
    print(log_split_sizes(splits))

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
        training_args = TrainingArguments(
            output_dir=stage_out,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_ratio=0.1,
            num_train_epochs=5,
            learning_rate=5e-5,
            fp16=quantize,
            logging_steps=100,
            optim="paged_adamw_8bit" if quantize else "adamw_torch",
            evaluation_strategy="steps",
            eval_steps=150,
            save_steps=450,
            save_total_limit=2,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=dataloader_num_workers,
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            args=training_args,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        all_log_history[f"stage_{stage}"] = trainer.state.log_history

        adapter_name = stage_adapter_name(stage)
        # peft nests a multi-adapter model's save under save_directory/<adapter_name>/ --
        # that nested folder is the actual flat checkpoint we want at the repo root.
        model.save_pretrained(stage_out, selected_adapters=[adapter_name], save_embedding_layers=True)
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
