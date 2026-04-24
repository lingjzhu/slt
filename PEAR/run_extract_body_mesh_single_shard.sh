#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_SHARD OUTPUT_SHARD [EXTRA_ARGS...]"
  echo "Example:"
  echo "  $0 /path/to/in.tar /path/to/out.tar --device cuda:0 --target-fps 12"
  exit 1
fi

INPUT_SHARD="$1"
OUTPUT_SHARD="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda run -n pear python "${SCRIPT_DIR}/extract_csl_daily_wds_features.py" \
  --input-shard "${INPUT_SHARD}" \
  --output-shard "${OUTPUT_SHARD}" \
  "$@"
