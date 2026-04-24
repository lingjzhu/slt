#!/usr/bin/env bash
# Example script to launch all 8 lists across 8 GPUs on a single node.
# For HPC (e.g. SLURM), you would typically submit each list as a separate job.

INPUT_ROOT="/mnt/data2/sign_language_24fps/processed_24fps"
OUTPUT_ROOT="/mnt/data2/PEAR_outputs"

mkdir -p "$OUTPUT_ROOT"

for i in {0..7}; do
  echo "Launching list $i on GPU $i"
  ./run_shard_list.sh "shard_list_$i.txt" "$INPUT_ROOT" "$OUTPUT_ROOT" --device "cuda:$i" > "log_list_$i.txt" 2>&1 &
done

echo "All 8 lists launched in background."
echo "Use 'tail -f log_list_0.txt' to monitor progress."
