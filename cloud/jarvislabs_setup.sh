#!/usr/bin/env bash
# Bootstrap a JarvisLabs instance for a real training run.
# Provisioned as: A100 80GB, 28 vCPU, 112GB RAM, 100GB storage, dynamic IP.
# batch sizes and worker counts in configs/*.yaml and src/evaluate.py are tuned
# for this tier -- halve batch_size and drop workers to 2-4 if run on the 40GB/
# 16-vCPU tier instead.
# Run once per fresh instance, then use scripts/run_baseline.py.
set -euo pipefail

REPO_URL="https://github.com/gadmin7/mwp_ai4math_icml_v2.git"

echo "== cloning repo =="
git clone "$REPO_URL" ~/mwp_ai4math_icml_v2
cd ~/mwp_ai4math_icml_v2

echo "== installing pinned requirements =="
pip install -q -r requirements.txt

echo "== huggingface auth =="
echo "Run: hf auth login   (needs your token, can't be scripted)"
read -p "Press enter once logged in..."

echo "== re-running the smoke test on real CUDA + bitsandbytes before spending on full runs =="
python3 scripts/smoke_test.py --skip-data

echo "== ready. Launch a baseline with, e.g.: =="
echo "  export HF_TOKEN=\$(hf auth token)"
echo "  tmux new -s b1 'python3 scripts/run_baseline.py --config configs/baseline1.yaml 2>&1 | tee b1.log'"
echo "  tmux new -s b2 'python3 scripts/run_baseline.py --config configs/baseline2.yaml 2>&1 | tee b2.log'"
echo "  # ... then b3-b6 in order once the pipeline is confirmed working on b1/b2"
