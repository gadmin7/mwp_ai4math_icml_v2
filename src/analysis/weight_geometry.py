"""Per-layer weight-update analysis across PLRS stages.

Only meaningful now that each stage is a separately addressable, frozen adapter
(see src/train_stage.py) -- under the original buggy pipeline there was nothing
to compare, since each stage discarded the previous one's weights before saving.

Each stage's realized low-rank update is dW_i = (alpha_i / r_i) * B_i @ A_i,
one such update per {layer, target module}. Two views:

  - layer_update_norms: ||dW_i||_F per (layer, module, stage) -- where in the
    network each stage concentrates its capacity.
  - overwrite_heatmap: element-wise comparison of dW_i vs dW_j (i < j) on the
    same weight matrix -- same-sign entries reinforce the earlier stage's
    contribution, opposite-sign entries partially cancel ("overwrite") it.
"""

import json
import os
import re
from dataclasses import dataclass

import torch
from safetensors.torch import load_file

# On-disk single-adapter checkpoints (what save_pretrained(selected_adapters=[...])
# actually writes) drop the adapter-name infix entirely -- it's implied by the
# directory being scoped to one adapter. Accept both forms for robustness.
_LORA_KEY_RE = re.compile(r"^(?P<prefix>.*)\.lora_(?P<AB>[AB])(?:\.(?P<adapter>[^.]+))?\.weight$")


@dataclass
class StageDelta:
    stage: int
    adapter_name: str
    deltas: dict  # module_key -> torch.Tensor, shape [out, in]


def load_stage_delta(checkpoint_dir: str, stage: int, adapter_name: str) -> StageDelta:
    weights_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    state = load_file(weights_path)
    with open(config_path) as f:
        cfg = json.load(f)
    scaling = cfg["lora_alpha"] / cfg["r"]

    pairs = {}
    for key, tensor in state.items():
        m = _LORA_KEY_RE.match(key)
        if not m:
            continue
        found_adapter = m.group("adapter")
        if found_adapter is not None and found_adapter != adapter_name:
            continue
        module_key = m.group("prefix")
        pairs.setdefault(module_key, {})[m.group("AB")] = tensor

    deltas = {}
    for module_key, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        deltas[module_key] = scaling * (ab["B"].float() @ ab["A"].float())
    return StageDelta(stage=stage, adapter_name=adapter_name, deltas=deltas)


def layer_update_norms(stage_deltas: list) -> dict:
    """{module_key: {stage: frobenius_norm}}"""
    out = {}
    for sd in stage_deltas:
        for module_key, dW in sd.deltas.items():
            out.setdefault(module_key, {})[sd.stage] = torch.linalg.norm(dW).item()
    return out


def overwrite_heatmap(stage_i: StageDelta, stage_j: StageDelta) -> dict:
    """{module_key: elementwise sign-overlap tensor} for modules present in both stages.

    overlap = sign(dW_i) * sign(dW_j) * min(|dW_i|, |dW_j|)  -- positive where stage j
    reinforces stage i's update, negative where it partially cancels ("overwrites") it,
    magnitude bounded by whichever stage's update is smaller at that entry.
    """
    common = set(stage_i.deltas) & set(stage_j.deltas)
    out = {}
    for module_key in common:
        a, b = stage_i.deltas[module_key], stage_j.deltas[module_key]
        if a.shape != b.shape:
            continue
        overlap = torch.sign(a) * torch.sign(b) * torch.minimum(a.abs(), b.abs())
        out[module_key] = overlap
    return out


def summarize_overwrite(heatmaps: dict) -> dict:
    """{module_key: {'reinforced_frac': ..., 'cancelled_frac': ...}} -- fraction of
    nonzero entries that reinforce vs. partially cancel the earlier stage's update.
    """
    out = {}
    for module_key, overlap in heatmaps.items():
        nonzero = overlap[overlap != 0]
        if nonzero.numel() == 0:
            out[module_key] = {"reinforced_frac": 0.0, "cancelled_frac": 0.0}
            continue
        out[module_key] = {
            "reinforced_frac": (nonzero > 0).float().mean().item(),
            "cancelled_frac": (nonzero < 0).float().mean().item(),
        }
    return out
