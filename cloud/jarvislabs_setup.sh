#!/usr/bin/env bash
# Bootstrap a JarvisLabs PyTorch TEMPLATE for a real training run.
# Target: A100 80GB, 28 vCPU, 112GB RAM, 100GB storage.
#
# Launch a Template (pre-built PyTorch container), NOT a VM: requirements.txt
# deliberately leaves torch unpinned and the venv below uses --system-site-packages,
# both so we inherit the image's CUDA-matched torch instead of pulling a multi-GB
# wheel that may not match the driver. A bare VM has no torch to inherit.
#
# Batch sizes / worker counts in configs/*.yaml and src/evaluate.py are tuned for this
# tier -- halve batch_size and drop workers to 2-4 on the 40GB/16-vCPU tier.
#
# See cloud/RUNBOOK.md for the full launch-to-teardown procedure.
# Safe to re-run; re-run after a pause/resume only if imports fail.
set -euo pipefail

REPO_URL="https://github.com/gadmin7/mwp_ai4math_icml_v2.git"

# PERSISTENCE: only /home survives a pause/resume on JarvisLabs. Anything installed
# globally (system pip, apt) is lost on resume. Note these paths are hardcoded to
# /home rather than $HOME on purpose -- templates log you in as root, where $HOME is
# /root, which is NOT persistent. Keeping the venv and the HF cache under /home means
# a resumed instance keeps both its dependencies and its multi-GB model downloads.
PERSIST="/home"
REPO_DIR="$PERSIST/mwp_ai4math_icml_v2"
VENV="$PERSIST/mwp-venv"
export HF_HOME="$PERSIST/.cache/huggingface"

echo "== cloning repo into $REPO_DIR =="
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "== python venv at $VENV (under /home so it survives pause/resume) =="
if [ ! -d "$VENV" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "== installing requirements =="
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== environment =="
python3 -c "import torch; print(f'  torch {torch.__version__}  cuda={torch.cuda.is_available()}  gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
python3 -c "import transformers, peft; print(f'  transformers {transformers.__version__}  peft {peft.__version__}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "== huggingface auth =="
# Run the login inline rather than telling the user to run it and blocking on a read --
# they'd be stuck at our prompt with no shell to run it in. Skip if already authed
# (e.g. re-running after a resume, where /home/.cache/huggingface persisted the token).
if python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
  echo "  already logged in as $(python3 -c 'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])')"
elif [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
  cat <<'NOTE'
  Paste a token with read access to gated repos (Llama-3.2-1B-Instruct) AND write
  access -- checkpoints are pushed to GT1999/*. A read-only token fails at the first
  push, i.e. only after a stage has already finished training.
  Create one at: https://huggingface.co/settings/tokens
NOTE
  hf auth login
fi

echo "== verifying HF access before spending GPU time =="
python3 - <<'PYEOF'
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
print(f"  account: {api.whoami().get('name')}")
hf_hub_download("meta-llama/Llama-3.2-1B-Instruct", "config.json")
print("  gated model read: OK")
# Round-trip a throwaway repo so a read-only token fails HERE, not after training.
repo = f"{api.whoami().get('name')}/mwp-v2-authcheck"
api.create_repo(repo, private=True, exist_ok=True)
api.delete_repo(repo)
print("  write access: OK")
PYEOF

echo "== verification: smoke test on real CUDA + bitsandbytes =="
python3 scripts/smoke_test.py --skip-data

echo "== verification: end-to-end dry run of the real pipeline (~1 min) =="
# Catches API/version breakage on THIS box before a multi-hour run starts.
python3 scripts/dry_run.py --baseline b6 --n-per-level 40

cat <<NOTE

== ready ==
Every new shell (including after a resume) needs:

  source $VENV/bin/activate
  export HF_HOME="$HF_HOME"
  export HF_TOKEN=\$(hf auth token)
  cd $REPO_DIR && mkdir -p runs

Then launch a baseline inside tmux so it survives a dropped SSH connection:

  tmux new -s b9 'python3 scripts/run_baseline.py --config configs/baseline9.yaml 2>&1 | tee runs/b9.log'

Recommended order (see cloud/RUNBOOK.md):  9 10 1 2 6 7 8 11 12 13 3 4 5
NOTE
