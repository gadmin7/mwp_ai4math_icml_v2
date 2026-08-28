#!/usr/bin/env bash
# Does a HIGHER-RANK first stage learn a better basis?
#
# All three arms trained stage 1 on the identical 536 level-1 examples and differ
# only in rank, so evaluating each stage-1 adapter ALONE on the level-1 test set
# isolates the effect of rank at the stage with the least data:
#
#   zero-shot (no adapter)   reference point b_j
#   b7 stage 1   r=32        215 params per training token
#   b8 stage 1   r=102       684 params per training token
#   b6 stage 1   r=256     1,700 params per training token
#
# Two questions at once:
#   1. Does rank help or hurt when stage 1 has only ~105k tokens?
#   2. Is stage 1 learning transferable skill, or mostly output FORMAT?
#      (compare against zero-shot; watch the box rate, which was 0% -> 100%)
#
# ~20 min, evaluation only, no training. Results in results/.
set -uo pipefail

cd "${REPO_DIR:-/home/mwp_ai4math_icml_v2}" || { echo "repo not found" >&2; exit 1; }
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN not set: export HF_TOKEN=\$(hf auth token)" >&2; exit 1; }

echo "=== zero-shot (no adapters) on level 1 ==="
python3 scripts/evaluate_stages.py --config configs/baseline6.yaml \
        --scope level1 --stages 0 2>&1 | tail -6

for spec in "7:32" "8:102" "6:256"; do
  arm="${spec%%:*}"; rank="${spec##*:}"
  echo
  echo "=== b${arm} stage 1 alone (r=${rank}) on level 1 ==="
  python3 scripts/evaluate_stages.py --config "configs/baseline${arm}.yaml" \
          --scope level1 --stages 1 2>&1 | tail -6
done

echo
echo "================= SUMMARY ================="
printf "%-22s %-8s %-10s %s\n" "model" "rank" "EM %" "box %"
for f in results/b6-stage0-level1-predictions.csv \
         results/b7-stage1-level1-predictions.csv \
         results/b8-stage1-level1-predictions.csv \
         results/b6-stage1-level1-predictions.csv; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import sys, pandas as pd, os
f = sys.argv[1]; d = pd.read_csv(f)
base = os.path.basename(f)
label = "zero-shot (no adapter)" if "stage0" in base else base.split("-")[0] + " stage 1"
rank = {"b6-stage0": "-", "b7-stage1": "32", "b8-stage1": "102", "b6-stage1": "256"}.get(
    "-".join(base.split("-")[:2]), "?")
box = d["predicted_solution"].astype(str).str.contains("boxed", regex=False).mean()
print("%-22s %-8s %-10.2f %.1f" % (label, rank, 100*d["correct"].mean(), 100*box))
PY
done
echo "==========================================="
echo
echo "Reading it: if r=256 is not clearly better than r=32, high-rank-first is"
echo "harmful (536 examples cannot support a 256-dim basis) and 'start big, shrink'"
echo "is wrong for the nested design too. If all three barely beat zero-shot on EM"
echo "while box% jumps to ~100, stage 1 taught FORMAT, not math -- and inheriting"
echo "its subspace buys little."
