#!/usr/bin/env bash
# Run every baseline sequentially on one GPU, in the recommended order.
#
#   tmux new -s mwp 'bash scripts/run_all.sh 2>&1 | tee runs/run_all.log'
#
# Order is deliberate: cheap/decisive arms first, so problems surface before the
# expensive ones and each block answers something before the next begins.
#   9 10  single-stage; settles the Table 2 scaling confound
#   1 2   confirms the pipeline on real data
#   6 7 8 capacity-matched shrink / expand / constant triad
#   11 12 13  curriculum-ordering controls
#   3 4 5 remaining paper reproductions
#
# Sequential on purpose: two training jobs on one card contend for VRAM. To
# parallelise, use a multi-GPU instance and pin one arm per GPU via CUDA_VISIBLE_DEVICES.
#
# A failing arm is recorded and the run continues -- losing 12 arms because arm 3
# hit a transient Hub error would be worse than finishing and retrying that one.
set -uo pipefail

ORDER="${ORDER:-9 10 1 2 6 7 8 11 12 13 3 4 5}"
REPO_DIR="${REPO_DIR:-/home/mwp_ai4math_icml_v2}"

# Fatal on purpose: `set -e` is deliberately off (so one failing arm doesn't abort the
# rest), which means an unchecked cd would silently run everything from the wrong place.
cd "$REPO_DIR" || { echo "REPO_DIR not found: $REPO_DIR" >&2; exit 1; }
mkdir -p runs

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Run:  export HF_TOKEN=\$(hf auth token)" >&2
  exit 1
fi

: > runs/failures.txt
started_all=$(date +%s)

for b in $ORDER; do
  cfg="configs/baseline${b}.yaml"
  [ -f "$cfg" ] || { echo "missing $cfg, skipping"; echo "b$b (missing config)" >> runs/failures.txt; continue; }

  echo ""
  echo "=================================================================="
  echo "  baseline b$b   ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "=================================================================="
  started=$(date +%s)

  if python3 scripts/run_baseline.py --config "$cfg" 2>&1 | tee "runs/b${b}.log"; then
    status="ok"
  else
    status="FAILED"
    echo "b$b" >> runs/failures.txt
  fi

  mins=$(( ($(date +%s) - started) / 60 ))
  echo "--- b$b $status in ${mins}m ---"
done

total=$(( ($(date +%s) - started_all) / 60 ))
echo ""
echo "=================================================================="
echo "  all arms done in $((total / 60))h $((total % 60))m"
if [ -s runs/failures.txt ]; then
  echo "  FAILED: $(tr '\n' ' ' < runs/failures.txt)"
  echo "  retry one with: python3 scripts/run_baseline.py --config configs/baseline<N>.yaml"
else
  echo "  no failures"
fi
echo "=================================================================="
