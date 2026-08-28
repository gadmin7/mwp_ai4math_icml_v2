"""Generation-based exact-match evaluation on the held-out TEST set only.

Run exactly once, after a baseline's final stage finishes training -- never
during training (that's what src/data.py's val split is for). The scoring logic
(`_strip_string`, `_fix_fracs`, etc.) is ported verbatim from
experiment_notebooks/evaluations/exact_match_accuracy.ipynb in the original repo
so results stay comparable to the paper's methodology.
"""

import os
import re

import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from src.lora_schedule import BaselineSpec
from src.pipeline import PROMPT_TEMPLATE, repo_name
from src.train_stage import stage_adapter_name

# ---------------------------------------------------------------------------
# Scoring logic, ported verbatim from exact_match_accuracy.ipynb
# ---------------------------------------------------------------------------


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr:
                if substr[0] == "{":
                    new_str += substr
                else:
                    if len(substr) < 2:
                        return string
                    a, b = substr[0], substr[1]
                    if b != "{":
                        post_substr = substr[2:] if len(substr) > 2 else ""
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        post_substr = substr[2:] if len(substr) > 2 else ""
                        new_str += "{" + a + "}" + b + post_substr
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a, b = int(a), int(b)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        return string.split("\\text{ ")[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string):
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.strip("$").replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("\\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string


def find_box(pred_str: str) -> str:
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack, a = 1, ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
        return a
    return ans.split("$")[0].strip()


def extract_math_answer(pred_str: str, answer_flag: bool = True) -> str:
    if "boxed" in pred_str:
        pred = find_box(pred_str)
    elif answer_flag:
        pred_str = pred_str.split("=")[-1].strip()
        if re.match(r"[\d\.]+\s\D+$", pred_str):
            pred_str = pred_str.split(" ")[0]
        pred = pred_str
    else:
        preds = re.findall(r"-?\d*\.?\d+", pred_str)
        pred = preds[-1] if preds else ""
    return _strip_string(pred)


def clean_predicted_solution(text: str) -> str:
    text = text.replace(
        "system You are an expert math assistantuser Solve the following math problem:", ""
    )
    text = text.replace(
        "Show all intermediate steps and please mandatorily include the final answer "
        "in LaTeX format in a box like \\boxed{{}}. assistant",
        "",
    )
    return text.strip()


def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["predicted_solution_cleaned"] = df["predicted_solution"].apply(clean_predicted_solution)
    df["gt_processed"] = df["ground_truth"].apply(lambda x: extract_math_answer(x, True))
    df["pred_processed"] = df["predicted_solution_cleaned"].apply(lambda x: extract_math_answer(x, True))
    df["correct"] = df["gt_processed"] == df["pred_processed"]
    return df


def accuracy(df: pd.DataFrame) -> float:
    return df["correct"].mean()


def box_rate(df: pd.DataFrame) -> float:
    """Fraction of predictions that actually contained a \\boxed{...}.

    Worth reporting alongside EM: `extract_math_answer(..., answer_flag=True)` falls
    back to the raw text when no box is present, which then essentially never matches.
    Unboxed generations are therefore scored as wrong -- usually the right call (the
    prompt mandates a box), but a low box rate means EM is measuring format compliance
    as much as mathematical correctness, so it should not go unnoticed.
    """
    return df["predicted_solution"].str.contains("boxed", regex=False).mean()


def accuracy_by_level(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("level")["correct"].mean().reset_index(name="accuracy")


# ---------------------------------------------------------------------------
# Model loading (all stages' adapters active simultaneously) + generation
# ---------------------------------------------------------------------------


def _bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )


def stage_sources(
    baseline: BaselineSpec, hf_org: str, model_tag: str, local_dir: str = None,
    through_stage: int = None,
) -> list:
    """Where each stage's adapter lives: local checkpoint dirs if available, else the Hub.

    Prefers local (already on disk from training, so no re-download); falls back to the
    Hub per stage, which also lets you evaluate an arm trained on a different machine.
    """
    last = through_stage or baseline.num_stages
    sources = []
    for stage in range(1, last + 1):
        local = None
        if local_dir:
            candidate = os.path.join(
                local_dir, f"{baseline.id}-stage{stage}", stage_adapter_name(stage)
            )
            if os.path.isdir(candidate):
                local = candidate
        sources.append(local or repo_name(hf_org, model_tag, baseline.id, stage))
    return sources


def load_stacked_model(
    base_model_id: str, baseline: BaselineSpec, hf_org: str, model_tag: str,
    quantize: bool = True, token: str = None, local_dir: str = None,
    through_stage: int = None,
):
    """Load stages 1..through_stage (default: all) and activate them together.

    The evaluated model must reflect the full frozen stack plus the newest stage, not
    any single adapter in isolation -- activating only the last one would score a model
    that was never trained in that configuration.

    `through_stage` reconstructs an INTERMEDIATE model: stages 1..k as they stood after
    stage k, before later stages existed. This is what makes retrospective forgetting
    and error-trajectory analysis possible without retraining, and it only works because
    each stage is kept as a separate frozen adapter.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=_bnb_config() if quantize else None,
        device_map="auto" if quantize else None,
        token=token,
    )
    sources = stage_sources(baseline, hf_org, model_tag, local_dir, through_stage)
    for i, src in enumerate(sources):
        print(f"  stage {i + 1} adapter <- {src}")

    model = PeftModel.from_pretrained(
        base_model, sources[0], adapter_name=stage_adapter_name(1), token=token
    )
    for stage, src in enumerate(sources[1:], start=2):
        model.load_adapter(src, adapter_name=stage_adapter_name(stage), token=token)

    all_adapters = [stage_adapter_name(s) for s in range(1, len(sources) + 1)]
    model.base_model.set_adapter(all_adapters)
    active = set(model.active_adapters)
    assert active == set(all_adapters), (
        f"expected all {len(all_adapters)} stage adapters active, got {sorted(active)}"
    )
    print(f"  active adapters: {sorted(active)}")
    model.eval()
    return model


def generate_predictions(
    model, tokenizer, test_ds, batch_size: int = 64, max_new_tokens: int = 512, num_workers: int = 4,
) -> pd.DataFrame:
    """batch_size/num_workers default to values tuned for an 80GB-class GPU with
    ~28 vCPUs (JarvisLabs A100 80GB tier); drop batch_size to 32 and num_workers
    to 2 on a 40GB/16-vCPU box.
    """
    samples = [
        {
            "input_text": PROMPT_TEMPLATE.format(problem=x["problem"], solution=""),
            "problem": x["problem"],
            "level": x["level"],
            "type": x["type"],
            "ground_truth": x["solution"],
        }
        for x in test_ds
    ]

    def collate(batch):
        # Stay on CPU here -- this may run in a DataLoader worker subprocess;
        # moving to the GPU happens in the main loop below instead.
        texts = [s["input_text"] for s in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        return enc, batch

    loader = DataLoader(samples, batch_size=batch_size, collate_fn=collate, num_workers=num_workers)
    rows = []
    with torch.no_grad():
        for enc, batch in tqdm(loader, desc="generating"):
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out_ids = model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
            for i in range(len(batch)):
                text = tokenizer.decode(out_ids[i], skip_special_tokens=True)
                rows.append({**{k: batch[i][k] for k in ("problem", "level", "type", "ground_truth")},
                             "predicted_solution": text})
    return pd.DataFrame(rows)


def evaluate_baseline(
    baseline: BaselineSpec, base_model_id: str, hf_org: str, model_tag: str,
    test_ds, quantize: bool = True, token: str = None, batch_size: int = 64,
    num_workers: int = 4, local_dir: str = None, through_stage: int = None,
) -> pd.DataFrame:
    model = load_stacked_model(
        base_model_id, baseline, hf_org, model_tag, quantize, token, local_dir, through_stage
    )
    from src.pipeline import make_tokenizer
    tokenizer = make_tokenizer(base_model_id, token=token)
    preds = generate_predictions(model, tokenizer, test_ds, batch_size=batch_size, num_workers=num_workers)
    return score(preds)


def report(df: pd.DataFrame, label: str = "") -> str:
    """Human-readable EM summary: overall, per level, and the box rate."""
    lines = [f"=== {label} ===" if label else "==="]
    lines.append(f"exact match (overall): {100 * accuracy(df):.2f}%   n={len(df)}")
    lines.append(f"predictions containing \\boxed{{}}: {100 * box_rate(df):.1f}%")
    lines.append("")
    lines.append(f"{'level':<10}{'n':<8}{'EM %'}")
    for _, row in accuracy_by_level(df).iterrows():
        n = (df["level"] == row["level"]).sum()
        lines.append(f"{row['level']:<10}{n:<8}{100 * row['accuracy']:.2f}")
    return "\n".join(lines)
