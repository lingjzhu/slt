#!/usr/bin/env bash
# Launch extraction for webdataset_224 on GPUs 1, 2, 3

INPUT_ROOT="/home/slimelab/Projects/slt/islr/webdataset_224"
OUTPUT_ROOT="/mnt/data2/sign_language_24fps"

mkdir -p "$OUTPUT_ROOT"

cd /home/slimelab/Projects/slt/PEAR

for i in {0..2}; do
  GPU=$((i + 1))
  LIST="webdataset_224_list_${i}.txt"
  LOG="log_webdataset_224_${i}.txt"
  
  echo "Launching $LIST on GPU $GPU"
  nohup ./run_shard_list.sh "$LIST" "$INPUT_ROOT" "$OUTPUT_ROOT" --device "cuda:$GPU" --target-fps 24 --minimal > "$LOG" 2>&1 < /dev/null &
done

echo "3 jobs launched in background."
