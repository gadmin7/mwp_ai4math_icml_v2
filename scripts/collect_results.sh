#!/usr/bin/env bash
# Bundle everything worth keeping into one archive to scp home.
#
#   bash scripts/collect_results.sh          -> /home/mwp-results-<date>.tar.gz
#
# Includes: test predictions + EM summaries, training logs, per-stage loss/step
# histories, the resolved configs, and a manifest of where each checkpoint lives.
#
# EXCLUDES adapter weights (*.safetensors). They are already on the Hub, they are the
# bulk of the bytes (an r=256 adapter is ~720MB), and the manifest records the repo
# ids needed to pull any of them back. Anything not on the Hub is flagged below --
# check that before deleting the instance, since deleting takes /home with it.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/home/mwp_ai4math_icml_v2}"
cd "$REPO_DIR" || { echo "REPO_DIR not found: $REPO_DIR" >&2; exit 1; }

STAMP=$(date +%Y%m%d-%H%M)
OUT_DIR="${OUT_DIR:-/home}"
OUT="${OUT_DIR}/mwp-results-${STAMP}.tar.gz"
MANIFEST="checkpoint-manifest.txt"
mkdir -p "$OUT_DIR" || { echo "cannot create OUT_DIR: $OUT_DIR" >&2; exit 1; }

{
  echo "# collected $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo "# code commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo
  echo "## local checkpoints (adapter weights NOT in this archive)"
  if compgen -G "runs/*/*/adapter_model.safetensors" >/dev/null; then
    for f in runs/*/*/adapter_model.safetensors; do
      echo "  $(du -h "$f" | cut -f1)  $(dirname "$f")"
    done
  else
    echo "  (none found under runs/)"
  fi
  echo
  echo "## corresponding Hub repos"
  echo "  pattern: <hf_org>/mwp-v2-<model_tag>-<baseline_id>-stage<N>"
  for cfg in configs/*.yaml; do
    org=$(grep '^hf_org:' "$cfg" | awk '{print $2}')
    tag=$(grep '^model_tag:' "$cfg" | awk '{print $2}')
    bid=$(grep '^baseline_id:' "$cfg" | awk '{print $2}')
    echo "  $cfg -> $org/mwp-v2-$tag-$bid-stage*"
  done
} > "$MANIFEST"

echo "== bundling =="
TARGETS=("$MANIFEST" configs)
[ -d results ] && TARGETS+=(results)
[ -d runs ] && TARGETS+=(runs)
[ -d results ] || echo "  note: no results/ dir -- has scripts/evaluate.py been run yet?"

# Errors are NOT suppressed here on purpose: a silently failed tar would report an
# archive that does not exist, which is the worst possible outcome right before you
# delete the instance.
if ! tar -czf "$OUT" \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='__pycache__' \
  --exclude='checkpoint-*' \
  "${TARGETS[@]}"; then
  echo "ARCHIVE FAILED -- do not delete the instance" >&2
  exit 1
fi
[ -s "$OUT" ] || { echo "ARCHIVE IS EMPTY: $OUT" >&2; exit 1; }

echo
echo "  archive: $OUT  ($(du -h "$OUT" | cut -f1))"
echo "  contents:"
tar -tzf "$OUT" | sed 's/^/    /' | head -30
n=$(tar -tzf "$OUT" | wc -l)
[ "$n" -gt 30 ] && echo "    ... and $((n - 30)) more"

cat <<NOTE

== fetch it from your laptop ==
  scp -P <port> root@sshd.jarvislabs.ai:$OUT .

Confirm your checkpoints are on the Hub before deleting the instance -- deleting
takes /home with it, and only the Hub copies survive.
NOTE
