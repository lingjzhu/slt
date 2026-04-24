#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export SIGN_CLIP_LATENT_WEBDS_ROOT="${SIGN_CLIP_LATENT_WEBDS_ROOT:-$SCRIPT_DIR/islr/webdataset_224_leanvae_latents}"

CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "Set CHECKPOINT=/path/to/latent_sign_clip_checkpoint.pt" >&2
  exit 1
fi

exec python -m sign_clip.eval_latent_islr \
  --checkpoint "$CHECKPOINT" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-8}" \
  ${OUTPUT_JSON:+--output-json "$OUTPUT_JSON"}
