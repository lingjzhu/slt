#!/usr/bin/env bash

set -euo pipefail

OUT=/mnt/data4/all_train_plain_v3_mae_features_run_001_8fps
SRC=/mnt/data4/all_train_plain_v3
CKPT=/mnt/data4/mae_pretraining_runs/run_001/checkpoint-00004.pth
SCRIPT=/home/slimelab/Projects/slt/src/mae_pretraining/extract_features.py
MISSING_ROOT="$OUT/missing_rerun"

mkdir -p "$OUT/logs" "$OUT/manifests" "$OUT/shards" "$MISSING_ROOT/shards"

export PYTHONPATH=/home/slimelab/Projects/slt/src:${PYTHONPATH:-}

python - <<'PY'
from pathlib import Path

src_manifest = Path("/mnt/data4/all_train_plain_v3/manifests/paired_manifest.tsv")
feature_root = Path("/mnt/data4/all_train_plain_v3_mae_features_run_001_8fps/features")
missing_root = Path("/mnt/data4/all_train_plain_v3_mae_features_run_001_8fps/missing_rerun")
missing_manifest = missing_root / "missing_train.tsv"
shard_root = missing_root / "shards"

missing_root.mkdir(parents=True, exist_ok=True)
shard_root.mkdir(parents=True, exist_ok=True)

handles = []
for i in range(4):
    shard = shard_root / f"dataset_{i}"
    (shard / "manifests").mkdir(parents=True, exist_ok=True)
    videos_link = shard / "videos"
    if not videos_link.exists():
        videos_link.symlink_to(Path("/mnt/data4/all_train_plain_v3/videos"), target_is_directory=True)
    handles.append((shard / "manifests" / "train.tsv").open("w"))

kept = 0
missing = 0
skipped_duration = 0
with src_manifest.open() as fin, missing_manifest.open("w") as fout:
    for line in fin:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        try:
            duration = float(parts[1])
        except Exception:
            continue
        if duration > 30.0:
            skipped_duration += 1
            continue

        video_path = parts[0]
        stem = Path(video_path).stem
        prefix = stem[:5] if stem else "misc"
        feature_path = feature_root / prefix / f"{stem}.pt"
        if feature_path.exists():
            kept += 1
            continue

        fout.write(line)
        handles[missing % 4].write(line)
        missing += 1

for handle in handles:
    handle.close()

summary = missing_root / "summary.txt"
summary.write_text(
    f"existing_feature_rows_skipped={kept}\n"
    f"missing_rows_to_rerun={missing}\n"
    f"duration_gt_30_skipped={skipped_duration}\n"
)

print(f"Existing outputs skipped: {kept}")
print(f"Missing rows to rerun: {missing}")
print(f"Rows skipped for duration > 30s: {skipped_duration}")
print(f"Wrote missing manifest: {missing_manifest}")
PY

for gpu in 0 1 2 3; do
    csv_name="train_feature_missing_gpu${gpu}.csv"
    log_file="$OUT/logs/missing_gpu${gpu}.log"
    pid_file="$OUT/logs/missing_gpu${gpu}.pid"

    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" \
        --data-dir "$MISSING_ROOT/shards/dataset_${gpu}" \
        --output-dir "$OUT" \
        --pretrained-model-path "$CKPT" \
        --model-name hiera_base_128x224 \
        --split train \
        --target-fps 8 \
        --sampling-rate 1 \
        --window-stride 128 \
        --bf16 \
        --batch-size 8 \
        --gpu-batch-size 32 \
        --num-workers 1 \
        --video-backend decord \
        --fused \
        --csv-name "$csv_name" \
        > "$log_file" 2>&1 &

    echo $! > "$pid_file"
    echo "Launched missing-rerun GPU $gpu pid $(cat "$pid_file")"
done

status=0
for gpu in 0 1 2 3; do
    pid=$(cat "$OUT/logs/missing_gpu${gpu}.pid")
    if ! wait "$pid"; then
        echo "Missing-rerun GPU $gpu failed" >&2
        status=1
    fi
done

if [ "$status" -ne 0 ]; then
    echo "One or more missing-rerun workers failed" >&2
    exit "$status"
fi

echo "Missing-rerun extraction finished."
echo "Logs:"
echo "  $OUT/logs/missing_gpu0.log"
echo "  $OUT/logs/missing_gpu1.log"
echo "  $OUT/logs/missing_gpu2.log"
echo "  $OUT/logs/missing_gpu3.log"
