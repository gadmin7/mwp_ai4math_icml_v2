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
    # How examples are assigned to stages: "difficulty" (the paper's easy->hard
    # curriculum), "random", or "reverse". All three use identical stage sizes and
    # therefore identical exposure and step counts -- see data.assign_stages.
    partition: str = "difficulty"
    # Single-pass arm that reproduces cumulative replay's 5:4:3:2:1 per-level exposure
    # by duplication, to test whether staging adds anything over reweighting alone.
    exposure_weighted: bool = False
    # Evals without improvement before a stage stops. The paper's value is 2, but at
    # eval_steps=150 that kills a stage after only 300 flat steps -- and a staged arm
    # gets one such chance PER STAGE. Measured: b6/b7 stage 5 stopped at 1200/600 steps
    # against b1's 1950 on the same data, i.e. the staged arms were trained less at the
    # decisive stage. Since lr x steps is roughly constant for a target loss (Marek et
    # al. 2026), those arms are undertrained rather than converged. See b14.
    early_stopping_patience: int = 2
    # Epochs PER STAGE. Set explicitly on the staged/joint arms so all three see the
    # same number of example-passes -- otherwise the comparison measures compute, not
    # schedule. See CURRICULUM ARMS below for the arithmetic.
    num_epochs: int = 5
    # True  = PLRS-style: a NEW frozen adapter per stage (capacity grows with stages).
    # False = one adapter trained continuously as the data grows -- standard curriculum
    #         learning, and the only way a staged arm is capacity-matched to a joint one.
    # Stacking also fights transfer here: each new adapter is randomly initialised into
    # an independent subspace (measured: rank of union 510/510), so it cannot reuse what
    # the previous stage learned.
    stack_adapters: bool = True

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
    # --- Does easy->hard ORDERING actually matter? ------------------------------
    # b1-vs-b6 cannot answer this: it varies stage count, total steps, ordering,
    # per-level exposure, adapter structure and rank all at once. b8/b11/b12 fix the
    # rank (constant r=102), stage sizes, cumulative sizes and total step count, and
    # vary the difficulty composition of each stage:
    #   b8  difficulty (easy->hard)   b11 random (mixed)   b12 reverse (hard->easy)
    # Constant rank is deliberate -- a shrinking schedule would interact with the
    # ordering ("big rank for stage 1" is incoherent when stage 1 is random or hardest).
    #
    # IMPORTANT: under cumulative replay, ordering and per-level exposure are coupled by
    # construction -- whatever goes first is replayed most, so they cannot be varied
    # independently. Measured mean stages-trained-in, per level:
    #   b8  difficulty : L1=5.00 L2=4.00 L3=3.00 L4=2.00 L5=1.00   (the 5:4:3:2:1 gradient)
    #   b11 random     : L1=2.58 L2=2.39 L3=2.55 L4=2.50 L5=2.47   (flat)
    #   b12 reverse    : L1=1.00 L2=1.00 L3=1.75 L4=2.71 L5=4.07   (inverted)
    # So each arm tests an ordering *package* (order + the exposure it induces), not
    # ordering in isolation. b13 supplies the missing piece by isolating the exposure
    # weighting with no staging or ordering at all, giving a three-way decomposition:
    #   b13 = exposure only        b11 = staging, no difficulty signal, flat exposure
    #   b8  = staging + easy->hard ordering + 5:4:3:2:1 exposure
    # b13 ~ b8 would mean the effect is just reweighting; b11 ~ b8 would mean staging
    # helps but the difficulty ordering does not.
    #
    # b8 vs b11 per-level is also the test of the "positive backward transfer" claim:
    # level-1 data gets 5x the exposure of level-5 under the difficulty curriculum, so
    # the Level-1 gain (42.1 -> 53.3) may be exposure rather than transfer. b11 flattens
    # exposure, so a Level-1 gain that survives there is genuine transfer.
    "b11": BaselineSpec(
        id="b11", name="SFR + constant r=102, RANDOM stage partition",
        replay=True, num_stages=5, ranks=[102] * 5, partition="random",
    ),
    "b12": BaselineSpec(
        id="b12", name="SFR + constant r=102, REVERSE (hard->easy) partition",
        replay=True, num_stages=5, ranks=[102] * 5, partition="reverse",
    ),
    # Is sequential fine-tuning anything more than a data reweighting? Single pass over
    # an exposure-matched dataset (each example repeated 6-L times): same per-level
    # gradient budget and same total steps as the staged pipeline, no staging.
    # Caveat when reading b13 vs b8: b13 has one r=102 adapter (71.9M) against b8's five
    # (359.3M), so only a b13 >= b8 result is unambiguous -- it would show staging adds
    # nothing beyond reweighting despite 5x the capacity.
    "b13": BaselineSpec(
        id="b13", name="Single pass, exposure-weighted 5:4:3:2:1, r=102",
        replay=False, num_stages=1, ranks=[102], exposure_weighted=True,
    ),
    # --- Is the staged null result just truncation? -----------------------------
    # Identical to b6 except patience=5. b6 matched the single-pass baseline (13.54 vs
    # 13.96) while ALSO ending at a worse validation loss (0.7343 vs 0.7139), which is
    # consistent with two different stories: staging genuinely does not help, or the
    # staged arms were cut short by patience=2 firing five times. b14 separates them --
    # if it climbs materially over b6, every staged arm needs rerunning at higher
    # patience and the null result is an artifact.
    "b14": BaselineSpec(
        id="b14", name="b6 schedule with patience=5 (truncation control)",
        replay=True, num_stages=5, ranks=[256, 128, 64, 32, 32],
        early_stopping_patience=5,
    ),
    # --- Rank sweep: are ALL the ranks above the optimum? -----------------------
    # The training set is ~1.41M tokens (7124 examples x ~197 tokens). Trainable
    # parameters per rank are 704,512, so params-per-token runs:
    #   r=4 -> 2.0   r=8 -> 4.0   r=16 -> 8.0   r=32 -> 16.0   r=256 -> 128.3
    # Every rank run so far sits below ONE token per trainable parameter (compute-
    # optimal pretraining is ~20 tokens/param), i.e. deep in the over-parameterised
    # regime -- and the one clean comparison agrees: b1 (r=32) 13.96% beat b9 (r=256)
    # 10.58%. If the optimum is near r=8, then b6/b7/b8 compared schedule shapes
    # entirely above it, which would explain why they were indistinguishable.
    # These are single-pass, directly comparable to b1 (r=32) and b9 (r=256).
    "b15": BaselineSpec(
        id="b15", name="Direct baseline, single pass r=4",
        replay=False, num_stages=1, ranks=[4],
    ),
    "b16": BaselineSpec(
        id="b16", name="Direct baseline, single pass r=8",
        replay=False, num_stages=1, ranks=[8],
    ),
    "b17": BaselineSpec(
        id="b17", name="Direct baseline, single pass r=16",
        replay=False, num_stages=1, ranks=[16],
    ),
    # --- CURRICULUM ARMS: does moving through complexity IN STEPS help? -----------
    # The hypothesis, minimally: is staged training better than training on everything
    # at once? Two arms would conflate ORDER with EXPOSURE -- under cumulative replay
    # level 1 is seen 5x as often as level 5, so a staged win could be reweighting
    # rather than curriculum. Three arms separate them:
    #
    #   staged vs jointw  -> isolates ORDER    (exposure matched)
    #   jointw vs jointu  -> isolates EXPOSURE (no ordering in either)
    #
    # Compute is matched by construction. The exposure-weighted set has exactly
    # sum(cumulative sizes) = 17,741 examples, so:
    #   staged  5 stages x 2 epochs over cumulative  = 35,482 example-passes
    #   jointw  2 epochs over the 17,741 weighted set = 35,482   (exact)
    #   jointu  5 epochs over the 7,124 full set      = 35,620   (+0.4%)
    #
    # Rank is constant: PLRS's schedule is a separate question, and varying it here
    # would reintroduce the confound this design exists to remove.
    "staged": BaselineSpec(
        id="staged", name="Curriculum: L1 -> L1,2 -> ... -> L1..5 (constant rank)",
        replay=True, num_stages=5, ranks=[32] * 5, num_epochs=2,
        stack_adapters=False,   # capacity-matched to the joint arms: ONE adapter
    ),
    "jointw": BaselineSpec(
        id="jointw", name="Joint, exposure-weighted 5:4:3:2:1 (order removed, exposure kept)",
        replay=False, num_stages=1, ranks=[32], exposure_weighted=True, num_epochs=2,
    ),
    "jointu": BaselineSpec(
        id="jointu", name="Joint, uniform (order and exposure both removed)",
        replay=False, num_stages=1, ranks=[32], num_epochs=5,
    ),
}


def get_baseline(baseline_id: str) -> BaselineSpec:
    if baseline_id not in BASELINES:
        raise ValueError(f"Unknown baseline id {baseline_id!r}; choose from {list(BASELINES)}")
    return BASELINES[baseline_id]
