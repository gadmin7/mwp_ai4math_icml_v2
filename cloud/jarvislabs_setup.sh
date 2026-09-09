#!/usr/bin/env bash
# Bootstrap a JarvisLabs PyTorch TEMPLATE for a training or analysis run.
# Target: A100 80GB, 28 vCPU, 112GB RAM, 100GB storage.
#
# Launch a Template (pre-built PyTorch container), NOT a bare VM: requirements.txt
# leaves torch unpinned and the venv uses --system-site-packages, so we inherit the
# image's CUDA-matched torch instead of pulling a multi-GB wheel. A bare VM has none.
#
# Safe to re-run, and you SHOULD re-run after every pause/resume -- see the CUDA note.
# Full launch-to-teardown procedure: cloud/RUNBOOK.md
set -euo pipefail

REPO_URL="https://github.com/gadmin7/difficulty-geometry.git"

# PERSISTENCE: only /home survives a pause/resume on JarvisLabs. Anything installed
# globally is lost. These paths are /home rather than $HOME on purpose -- templates log
# you in as root, and /root is NOT persistent. Keeping the venv and HF cache under /home
# means a resumed instance keeps its dependencies and its multi-GB model downloads.
PERSIST="/home"
REPO_DIR="$PERSIST/difficulty-geometry"
VENV="$PERSIST/mwp-venv"
export HF_HOME="$PERSIST/.cache/huggingface"

echo "== cloning repo into $REPO_DIR =="
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" remote set-url origin "$REPO_URL"   # repo was renamed
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "== python venv at $VENV (under /home so it survives pause/resume) =="
[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "== installing requirements =="
pip install -q --upgrade pip
pip install -q -r requirements.txt

# ---------------------------------------------------------------------------
# CUDA / DRIVER CHECK.  Read this before "fixing" it a different way.
#
# A pause/resume can bring the volume up on DIFFERENT HARDWARE with an OLDER driver
# than the image's torch was built against. Observed in practice: resumed from driver
# 595.58 onto 570.86, and the inherited torch (cu130) reported
#   "The NVIDIA driver on your system is too old (found version 12080)"
# with torch.cuda.is_available() == False. Everything then silently runs on CPU.
#
# Two traps in the fix:
#   1. `pip install torch` is a SILENT NO-OP here. --system-site-packages means pip
#      sees the image's torch and considers the requirement satisfied. You need
#      --force-reinstall to actually get a wheel into the venv.
#   2. torch and torchvision must be a MATCHED PAIR. Installing torch alone against a
#      stale torchvision gives "operator torchvision::nms does not exist" at import,
#      which surfaces as an unrelated-looking transformers/peft import error.
# ---------------------------------------------------------------------------
echo "== CUDA check =="
if ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "  torch cannot see the GPU -- driver/toolkit mismatch after resume."
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  echo "  driver is $DRV; reinstalling a cu126 torch+torchvision pair"
  pip install -q --force-reinstall --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cu126
fi

echo "== environment =="
python3 - <<'PYEOF'
import torch
ok = torch.cuda.is_available()
print(f"  torch {torch.__version__}  built_for_cuda={torch.version.cuda}  available={ok}")
print(f"  gpu: {torch.cuda.get_device_name(0) if ok else 'NONE'}")
assert ok, "GPU still unavailable -- do NOT start a run, it will silently use CPU"
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
print(f"  bf16 matmul OK ({(x @ x).float().mean().item():.4f})")
import transformers, peft
print(f"  transformers {transformers.__version__}  peft {peft.__version__}")
PYEOF
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo "== huggingface auth =="
# Run login inline rather than instructing the user to -- they would be stuck at our
# prompt with no shell. Skipped if the token persisted in /home/.cache/huggingface.
if python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
  echo "  already logged in as $(python3 -c 'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])')"
elif [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
  cat <<'NOTE'
  Paste a token with read access to gated repos (meta-llama/Llama-3.2-1B) AND write
  access -- checkpoints push to GT1999/*. A read-only token fails at the first push,
  i.e. only after a stage has already finished training.
  Create one at: https://huggingface.co/settings/tokens
NOTE
  hf auth login
fi

echo "== verifying HF access before spending GPU time =="
python3 - <<'PYEOF'
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
print(f"  account: {api.whoami().get('name')}")
# The BASE model is what the arms train; Instruct is not used by the current configs.
hf_hub_download("meta-llama/Llama-3.2-1B", "config.json")
print("  gated model read: OK")
repo = f"{api.whoami().get('name')}/authcheck-tmp"
api.create_repo(repo, private=True, exist_ok=True)
api.delete_repo(repo)
print("  write access: OK")
PYEOF

echo "== verification: smoke test (adapter stacking + freezing semantics) =="
python3 scripts/smoke_test.py --skip-data

echo "== verification: end-to-end dry run of a real arm (~1 min) =="
# Catches API/version breakage on THIS box before a multi-hour run starts.
python3 scripts/dry_run.py --baseline staged --n-per-level 40

cat <<NOTE

== ready ==

Every new shell (including after a resume) needs:

  source $VENV/bin/activate
  export HF_HOME="$HF_HOME"
  export HF_TOKEN=\$(hf auth token)
  cd $REPO_DIR && mkdir -p runs results

Run an arm inside tmux so it survives a dropped SSH connection:

  tmux new -s a 'python3 scripts/run_baseline.py --config configs/jointu.yaml 2>&1 | tee runs/jointu.log'

The four curriculum arms (compute- and capacity-matched; ~45 min train + ~21 min eval each):

  jointu      no ordering, uniform exposure          <- the reference arm
  jointw      no ordering, 5:4:3:2:1 exposure
  staged      easy->hard, cumulative replay
  staged_nr   easy->hard, no replay

Analysis, no training required:

  python3 scripts/gradient_overlap.py --n-per-level 64 --k 32   # subspaces + shuffled floor
  python3 scripts/transfer_test.py    --n-train 400             # does L1 help L2..L5?
  python3 scripts/test_loss.py --configs configs/jointu.yaml configs/staged.yaml

WATCH GPU UTILISATION, NOT JUST THAT THE PROCESS IS ALIVE.
A stalled progress bar and a healthy job look identical from 'ps'. An unquantised model
that never reached the GPU ran evaluation at 1687 s/batch instead of 16 -- a 105x
slowdown that cost half an hour before anyone noticed. Sanity numbers on an A100 80GB:

  training    ~53 GB VRAM, 90-100% util, ~2.5 it/s
  evaluation  ~13 GB VRAM, 85-100% util, ~16 s/batch

  watch -n 30 nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

On the 40GB tier: use configs/staged_nr_40gb.yaml as the pattern -- batch_size 4 with
gradient_checkpointing, which holds the effective batch at 16 and stays comparable to
80GB runs. Training at batch 8 without checkpointing peaks at 53 GB and will OOM.

Archive before pausing -- /home persists, but the instance may not come back:

  bash scripts/collect_results.sh    # writes /home/results-<timestamp>.tar.gz
NOTE
