"""Prompt templates, deliberately free of heavy imports.

Kept separate from pipeline.py so that analysis scripts which need only a prompt
string (e.g. scripts/gradient_overlap.py) don't transitively import Trainer and peft.
That coupling is not hypothetical: peft 0.20.0 does `from transformers import
BloomPreTrainedModel`, which transformers 5.x removed, so importing pipeline.py fails
outright on a modern transformers even when the script never touches an adapter.
"""

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are an expert math assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "Solve the following math problem: {problem} Show all intermediate steps and please "
    "mandatorily include the final answer in LaTeX format in a box like \\boxed{{}}."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{solution}<|eot_id|>"
)

# Explicitly asks for the "## Step N:" scaffold the INSTRUCT model produces on its own
# (measured: 7.18 step markers per zero-shot generation, 0.00 after fine-tuning).
PROMPT_TEMPLATE_COT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are an expert math assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "Solve the following math problem: {problem}\n"
    "Reason step by step. Format your reasoning as '## Step 1:', '## Step 2:', and so on, "
    "writing out every intermediate calculation explicitly rather than doing arithmetic "
    "in your head. Then give the final answer in LaTeX in a box like \\boxed{{}}."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{solution}<|eot_id|>"
)

PROMPT_TEMPLATES = {"default": PROMPT_TEMPLATE, "cot": PROMPT_TEMPLATE_COT}
