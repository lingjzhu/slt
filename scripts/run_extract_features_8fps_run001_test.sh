#!/usr/bin/env bash

set -euo pipefail

OUT=/mnt/data4/all_test_plain_v3_mae_features_run_001_8fps
SRC=/mnt/data4/all_test_plain_v3
CKPT=/mnt/data4/mae_pretraining_runs/run_001/checkpoint-00004.pth
SCRIPT=/home/slimelab/Projects/slt/src/mae_pretraining/extract_features.py

mkdir -p "$OUT/logs" "$OUT/manifests" "$OUT/shards"

export PYTHONPATH=/home/slimelab/Projects/slt/src:${PYTHONPATH:-}

python - <<'PY'
from pathlib import Path

src = Path("/mnt/data4/all_test_plain_v3/manifests/paired_manifest.tsv")
out = Path("/mnt/data4/all_test_plain_v3_mae_features_run_001_8fps/shards")
out.mkdir(parents=True, exist_ok=True)

handles = []
kept = 0
for i in range(4):
    shard = out / f"dataset_{i}"
    (shard / "manifests").mkdir(parents=True, exist_ok=True)
    videos_link = shard / "videos"
    if not videos_link.exists():
        videos_link.symlink_to(Path("/mnt/data4/all_test_plain_v3/videos"), target_is_directory=True)
    handles.append((shard / "manifests" / "test.tsv").open("w"))

with src.open() as fin:
    for line in fin:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        handles[kept % 4].write(line)
        kept += 1

for handle in handles:
    handle.close()

print(f"Created 4 test shard manifests with {kept} videos")
PY

for gpu in 0 1 2 3; do
    csv_name="test_feature_gpu${gpu}.csv"
    log_file="$OUT/logs/test_gpu${gpu}.log"
    pid_file="$OUT/logs/test_gpu${gpu}.pid"

    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" \
        --data-dir "$OUT/shards/dataset_${gpu}" \
        --output-dir "$OUT" \
        --pretrained-model-path "$CKPT" \
        --model-name hiera_base_128x224 \
        --split test \
        --target-fps 8 \
        --sampling-rate 1 \
        --window-stride 128 \
        --bf16 \
        --batch-size 32 \
        --gpu-batch-size 32 \
        --num-workers 4 \
        --video-backend decord \
        --fused \
        --csv-name "$csv_name" \
        > "$log_file" 2>&1 &

    echo $! > "$pid_file"
    echo "Launched test GPU $gpu pid $(cat "$pid_file")"
done

status=0
for gpu in 0 1 2 3; do
    pid=$(cat "$OUT/logs/test_gpu${gpu}.pid")
    if ! wait "$pid"; then
        echo "Test GPU $gpu failed" >&2
        status=1
    fi
done

if [ "$status" -ne 0 ]; then
    echo "One or more test GPU workers failed" >&2
    exit "$status"
fi

mkdir -p "$OUT/features"
ln -sfn . "$OUT/features/0"

python - <<'PY'
from pathlib import Path
import csv

out = Path("/mnt/data4/all_test_plain_v3_mae_features_run_001_8fps")
src_manifest = Path("/mnt/data4/all_test_plain_v3/manifests/paired_manifest.tsv")
shard_csvs = [out / f"test_feature_gpu{i}.csv" for i in range(4)]
merged_csv = out / "test_feature.csv"
test_feat_tsv = out / "manifests" / "test_features.tsv"
detail_tsv = out / "manifests" / "test_feature_detailed.tsv"

manifest_rows = []
with src_manifest.open() as fsrc:
    for line in fsrc:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 5:
            manifest_rows.append(parts)

rows_written = 0
with merged_csv.open("w", newline="") as fout:
    writer = None
    for path in shard_csvs:
        with path.open() as fin:
            reader = csv.DictReader(fin)
            if writer is None:
                writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                writer.writeheader()
            for row in reader:
                writer.writerow(row)
                rows_written += 1

print(f"Merged {rows_written} CSV rows")

with merged_csv.open() as fcsv, \
    test_feat_tsv.open("w", newline="") as ftest, \
    detail_tsv.open("w", newline="") as fdetail:
    reader = csv.DictReader(fcsv)
    test_writer = csv.writer(
        ftest, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
    )
    detail_fields = [
        "video_path",
        "feature_path",
        "duration",
        "dataset",
        "language",
        "caption",
        "num_feature_vectors",
        "feature_dim",
        "window_size",
        "window_stride",
        "num_clips",
        "padding_frames",
    ]
    detail_writer = csv.DictWriter(
        fdetail,
        fieldnames=detail_fields,
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )
    detail_writer.writeheader()

    for idx, (row, parts) in enumerate(zip(reader, manifest_rows), start=1):
        video_path, duration, dataset, language, caption = parts[:5]
        if row["video_path"] != video_path:
            raise RuntimeError(
                f"Manifest alignment mismatch at row {idx}: "
                f"{row['video_path']} != {video_path}"
            )

        video_name = Path(video_path).name
        test_writer.writerow([video_name, row["num_feature_vectors"], caption])
        detail_writer.writerow(
            {
                "video_path": video_path,
                "feature_path": row["feature_path"],
                "duration": duration,
                "dataset": dataset,
                "language": language,
                "caption": caption,
                "num_feature_vectors": row["num_feature_vectors"],
                "feature_dim": row["feature_dim"],
                "window_size": row["window_size"],
                "window_stride": row["window_stride"],
                "num_clips": row["num_clips"],
                "padding_frames": row["padding_frames"],
            }
        )

print("Wrote test_features.tsv and test_feature_detailed.tsv")
PY
