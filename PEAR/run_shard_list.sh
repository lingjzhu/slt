#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 LIST_FILE INPUT_ROOT OUTPUT_ROOT [EXTRA_ARGS...]"
  echo "Example:"
  echo "  $0 shard_list_0.txt /mnt/data2/sign_language_24fps/processed_24fps /mnt/data2/PEAR_outputs --device cuda:0"
  exit 1
fi

LIST_FILE="$1"
INPUT_ROOT="$2"
OUTPUT_ROOT="$3"
shift 3

SIF_PATH="/home/slimelab/Projects/slt/pear_v1.sif"
EXTRACT_SCRIPT="/home/slimelab/Projects/slt/PEAR/run_extract_body_mesh_single_shard.sh"

# Ensure input root is absolute
INPUT_ROOT=$(realpath "$INPUT_ROOT")
OUTPUT_ROOT=$(realpath "$OUTPUT_ROOT")

echo "Processing list: $LIST_FILE"
echo "Input root: $INPUT_ROOT"
echo "Output root: $OUTPUT_ROOT"

while IFS= read -r rel_path; do
  [[ -z "$rel_path" ]] && continue
  
  in_shard="${INPUT_ROOT}/${rel_path}"
  out_shard="${OUTPUT_ROOT}/${rel_path}"
  complete_file="${out_shard}.complete"
  
  if [[ -f "$complete_file" ]]; then
    echo "Skipping completed shard: $rel_path"
    continue
  fi

  # Ensure output directory exists
  mkdir -p "$(dirname "$out_shard")"
  
  echo "----------------------------------------------------------------"
  echo "Processing: $rel_path"
  
  # We need to bind the input root and output root to the container
  # We also bind the parent directory of the extraction script
  if apptainer exec --nv \
    -B "${INPUT_ROOT}:${INPUT_ROOT}:ro" \
    -B "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
    -B "/home/slimelab/Projects/slt:/home/slimelab/Projects/slt" \
    "$SIF_PATH" \
    bash "$EXTRACT_SCRIPT" "$in_shard" "$out_shard" --fp16 --minimal "$@"; then
    touch "$complete_file"
    echo "Successfully completed: $rel_path"
  else
    echo "Error processing: $rel_path"
  fi

done < "$LIST_FILE"

echo "Completed list: $LIST_FILE"
