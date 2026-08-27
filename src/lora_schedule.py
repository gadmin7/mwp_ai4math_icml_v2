"""Single source of truth for every baseline's LoRA rank schedule and replay policy.

The original repo hand-copied these into six near-identical notebooks; Baseline 4
silently drifted to r=256 at every stage instead of the paper's claimed fixed r=32
because nothing forced the six copies to agree. Every baseline is defined once here.
"""

from dataclasses import dataclass, field

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_DROPOUT = 0.2


@dataclass
class LoraStageConfig:
    r: int
    lora_alpha: int


@dataclass
class BaselineSpec:
    id: str
    name: str
    replay: bool  # True = cumulative D_i = {level <= i}; False = D_i = {level == i}
    num_stages: int
    ranks: list = field(default_factory=list)  # len == num_stages; ignored if num_stages == 1

    def stage_config(self, stage: int) -> LoraStageConfig:
        r = self.ranks[stage - 1]
        return LoraStageConfig(r=r, lora_alpha=2 * r)


BASELINES = {
    "b1": BaselineSpec(
        id="b1", name="Direct Baseline (single pass, r=32)",
        replay=False, num_stages=1, ranks=[32],
    ),
    "b2": BaselineSpec(
        id="b2", name="Sequential No-Replay, fixed r=32",
        replay=False, num_stages=5, ranks=[32, 32, 32, 32, 32],
    ),
    "b3": BaselineSpec(
        id="b3", name="SNR + PLRS (256->16)",
        replay=False, num_stages=5, ranks=[256, 128, 64, 32, 16],
    ),
    "b4": BaselineSpec(
        id="b4", name="Sequential Full Replay, fixed r=32",
        replay=True, num_stages=5, ranks=[32, 32, 32, 32, 32],
    ),
    "b5": BaselineSpec(
        id="b5", name="SFR + PLRS (256->16)",
        replay=True, num_stages=5, ranks=[256, 128, 64, 32, 16],
    ),
    "b6": BaselineSpec(
        id="b6", name="SFR + PLRS, no final shrink (256->32)",
        replay=True, num_stages=5, ranks=[256, 128, 64, 32, 32],
    ),
}


def get_baseline(baseline_id: str) -> BaselineSpec:
    if baseline_id not in BASELINES:
        raise ValueError(f"Unknown baseline id {baseline_id!r}; choose from {list(BASELINES)}")
    return BASELINES[baseline_id]
