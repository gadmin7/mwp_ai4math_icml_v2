"""Per-stage model preparation and training.

This is the fix for the original repo's core bug: every multi-stage baseline
called `get_peft_model()` on a model that was already a `PeftModel` (loaded via
`PeftModel.from_pretrained` from the previous stage's checkpoint). That discards
the prior stage's LoRA weights instead of freezing+stacking them -- verified by
reproducing the exact call sequence: peft warns about re-wrapping, and the old
`lora_B` values are unrecoverable afterward.

Here, each stage gets its own *named* adapter (`stage_1`, `stage_2`, ...) added
via `add_adapter`, with every earlier adapter's parameters explicitly frozen
before training. Nothing is ever re-wrapped or merged, so each stage's
contribution stays a separate, inspectable low-rank update (needed for
analysis/weight_geometry.py).
"""

from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.lora_schedule import LORA_DROPOUT, TARGET_MODULES, BaselineSpec

STAGE_PREFIX = "stage_"


def stage_adapter_name(stage: int) -> str:
    return f"{STAGE_PREFIX}{stage}"


def adapter_name_for(baseline, stage: int) -> str:
    """Which adapter this stage actually writes to.

    When stacking, each stage owns its own adapter. When not stacking there is only
    ever one adapter ("stage_1") that keeps training as the data grows, so every
    stage's checkpoint is a snapshot of that same adapter.
    """
    return stage_adapter_name(stage if baseline.stack_adapters else 1)


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def _lora_config(r: int, lora_alpha: int, use_rslora: bool = False) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=use_rslora,
    )


def _freeze_all_but(model: PeftModel, active_adapter: str) -> None:
    active_marker = f".{active_adapter}."
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        param.requires_grad_(active_marker in f".{name}.")


def prepare_stage_model(
    stage: int,
    baseline: BaselineSpec,
    base_model_id: str,
    prev_model: Optional[PeftModel],
    prev_adapter_path: Optional[str],
    quantize: bool,
    token: Optional[str] = None,
) -> PeftModel:
    """Return a model ready to train `stage`, with every earlier stage's adapter frozen.

    stage == 1: fresh base model + a single new adapter, "stage_1".
    stage  > 1: `prev_model` (already carrying stages 1..stage-1 as named, trained
                adapters) gets a new adapter added for this stage; everything else
                is frozen. If `prev_model` is None (e.g. resuming from disk), the
                prior stage's adapter is loaded from `prev_adapter_path` first.
    """
    cfg = baseline.stage_config(stage)
    lora_cfg = _lora_config(cfg.r, cfg.lora_alpha, cfg.use_rslora)
    adapter_name = stage_adapter_name(stage)

    if stage == 1:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=_bnb_config() if quantize else None,
            device_map="auto" if quantize else None,
            token=token,
        )
        model = get_peft_model(base_model, lora_cfg, adapter_name=adapter_name)
        return model

    if prev_model is not None:
        model = prev_model
        if not baseline.stack_adapters:
            # Continue training the SAME adapter rather than adding a new one, so a
            # 5-stage arm has exactly the capacity of a 1-stage arm and the comparison
            # isolates data ORDER instead of parameter count.
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.requires_grad_(True)
            return model
    else:
        if prev_adapter_path is None:
            raise ValueError("stage > 1 requires either prev_model or prev_adapter_path")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=_bnb_config() if quantize else None,
            device_map="auto" if quantize else None,
            token=token,
        )
        model = PeftModel.from_pretrained(
            base_model, prev_adapter_path, adapter_name=stage_adapter_name(stage - 1)
        )
        for s in range(stage - 2, 0, -1):
            model.load_adapter(prev_adapter_path_for_stage(prev_adapter_path, s), adapter_name=stage_adapter_name(s))

    model.add_adapter(adapter_name, lora_cfg)
    activate_stack(model, through_stage=stage)
    _freeze_all_but(model, adapter_name)
    return model


def activate_stack(model: PeftModel, through_stage: int) -> None:
    """Make stages 1..through_stage ALL active in the forward pass.

    Critical: `set_adapter("stage_N")` activates that adapter *exclusively*, which
    silently switches earlier stages off -- the forward pass then collapses back to
    the bare base model and the new stage learns as if no curriculum had happened.
    The whole premise of PLRS is that stage i trains on top of the frozen stack of
    stages 1..i-1, so every stage must stay active; only trainability differs
    (see _freeze_all_but, which must be called AFTER this -- activating an adapter
    also re-enables requires_grad on it).
    """
    stack = [stage_adapter_name(s) for s in range(1, through_stage + 1)]
    # Go through base_model (the LoraModel tuner): it accepts a list of adapters,
    # whereas PeftModel.set_adapter takes a single name.
    model.base_model.set_adapter(stack)


def prev_adapter_path_for_stage(any_stage_path: str, stage: int) -> str:
    """Given one stage's checkpoint dir, derive another stage's sibling dir.

    pipeline.py lays checkpoints out as `<output_dir>/<baseline_id>-stage{N}/stage_{N}`
    (the inner directory is peft's per-adapter nesting). Both occurrences of the stage
    number have to be rewritten.

    Raises rather than returning an unchanged path: silently handing back the wrong
    stage's directory would load the wrong weights and quietly corrupt the results.
    """
    import re

    pattern = re.compile(r"-stage\d+/" + re.escape(STAGE_PREFIX) + r"\d+/?$")
    if not pattern.search(any_stage_path):
        raise ValueError(
            f"cannot derive stage-{stage} path from {any_stage_path!r}: expected a path "
            f"ending in '-stage<N>/{STAGE_PREFIX}<N>'"
        )
    return pattern.sub(f"-stage{stage}/{stage_adapter_name(stage)}", any_stage_path)


def assert_stage_frozen(model: PeftModel, frozen_stage: int) -> None:
    """Defensive check used by the smoke test: a frozen stage's params require no grad
    and (by construction) were not touched by the current optimizer step.
    """
    marker = f".{stage_adapter_name(frozen_stage)}."
    for name, param in model.named_parameters():
        if "lora_" in name and marker in f".{name}.":
            assert not param.requires_grad, f"{name} should be frozen but requires_grad=True"
