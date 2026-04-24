#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export SIGN_CLIP_WEBDS_ROOT="${SIGN_CLIP_WEBDS_ROOT:-$SCRIPT_DIR/islr/webdataset_224}"

OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/sign_clip_islr}"

if [[ -n "${CHECKPOINT:-}" ]]; then
  CKPT="$CHECKPOINT"
else
  CKPT="$(ls -1 "$OUTPUT_DIR"/step-*.pt 2>/dev/null | sort | tail -n1 || true)"
fi
if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
  echo "No checkpoint found. Set CHECKPOINT=/path/to/step-*.pt or OUTPUT_DIR=." >&2
  exit 1
fi
echo "Evaluating checkpoint: $CKPT"

RESULTS_JSON="${RESULTS_JSON:-$OUTPUT_DIR/eval_test_results.json}"

exec python -m sign_clip.eval_islr \
  --checkpoint "$CKPT" \
  --output-json "$RESULTS_JSON" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-4}" \
  ${DATASETS:+--datasets $DATASETS}
