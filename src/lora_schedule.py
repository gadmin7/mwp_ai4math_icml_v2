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
    use_rslora: bool = False


@dataclass
class BaselineSpec:
    id: str
    name: str
    replay: bool  # True = cumulative D_i = {level <= i}; False = D_i = {level == i}
    num_stages: int
    ranks: list = field(default_factory=list)  # len == num_stages; ignored if num_stages == 1
    # Rank-stabilized LoRA scales by alpha/sqrt(r) instead of alpha/r. Standard
    # alpha/r scaling is known to under-serve large ranks, which is a candidate
    # explanation for the r=256 single-pass baseline underperforming r=32 -- see b9/b10.
    use_rslora: bool = False

    def stage_config(self, stage: int) -> LoraStageConfig:
        r = self.ranks[stage - 1]
        # alpha = 2r keeps the effective scaling (alpha/r = 2) constant across stages,
        # so a stage's update gain does not change just because its rank did.
        return LoraStageConfig(r=r, lora_alpha=2 * r, use_rslora=self.use_rslora)

    def total_trainable(self, params_per_rank: int = 704_512) -> int:
        """Cumulative trainable parameters across all stages (Llama-3.2-1B by default).

        Used to build capacity-matched schedules: comparing b6 against a constant or
        expanding schedule only isolates the schedule's *shape* if total capacity is
        held roughly fixed.
        """
        return sum(self.ranks) * params_per_rank


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
    # --- Controls the original study lacks -------------------------------------
    # No baseline in b1-b6 expands, so "shrinking helps" is never tested against its
    # opposite; and b4 (constant r=32) vs b6 confounds schedule SHAPE with total
    # capacity (b6 has ~16x the cumulative trainable parameters). b7/b8 hold total
    # capacity roughly fixed against b6 (360.7M) so only the shape differs.
    #
    # Motivation: cumulative train data grows 13.3x across stages (536 -> 7124) while
    # b6's rank shrinks 8x, so capacity per example falls ~106x -- even as each stage
    # introduces harder, entirely unseen material. The paper's single largest ablation
    # gain also came from shrinking LESS at the final stage (+3.10 EM).
    "b7": BaselineSpec(
        id="b7", name="SFR + expanding rank (32->128), capacity-matched to b6",
        replay=True, num_stages=5, ranks=[32, 64, 96, 128, 128],   # 315.6M (~12% under b6)
    ),
    "b8": BaselineSpec(
        id="b8", name="SFR + constant rank r=102, capacity-matched to b6",
        replay=True, num_stages=5, ranks=[102, 102, 102, 102, 102],  # 360.7M (exact match)
    ),
    # --- Table 2 confound check -------------------------------------------------
    # The paper reads "single-pass r=256 underperforms r=32" as evidence that capacity
    # scheduling (not rank budget) drives the gains. But alpha/r scaling is known to
    # under-serve large ranks, so that result may be a scaling artifact rather than a
    # statement about capacity. b9 vs b10 isolates exactly that.
    "b9": BaselineSpec(
        id="b9", name="Direct baseline, single pass r=256 (Table 2 heavy baseline)",
        replay=False, num_stages=1, ranks=[256],
    ),
    "b10": BaselineSpec(
        id="b10", name="Direct baseline, single pass r=256 + rsLoRA scaling",
        replay=False, num_stages=1, ranks=[256], use_rslora=True,
    ),
}


def get_baseline(baseline_id: str) -> BaselineSpec:
    if baseline_id not in BASELINES:
        raise ValueError(f"Unknown baseline id {baseline_id!r}; choose from {list(BASELINES)}")
    return BASELINES[baseline_id]
