#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${DATA_ROOT:-/mnt/data2/youtube-sl-25}"
LIMIT="${LIMIT:-}"
START="${START:-}"
COOKIES="${COOKIES:-}"
DELAY_MIN="${DELAY_MIN:-60}"
DELAY_MAX="${DELAY_MAX:-180}"
SLEEP="${SLEEP:-0}"
KEEP_RAW="${KEEP_RAW:-0}"
OVERWRITE="${OVERWRITE:-0}"
NO_WRITE_SUBS="${NO_WRITE_SUBS:-0}"
WRITE_AUTO_SUBS="${WRITE_AUTO_SUBS:-0}"
SUB_LANGS="${SUB_LANGS:-}"
SUB_FORMAT="${SUB_FORMAT:-vtt/best}"
CRF="${CRF:-23}"
FFMPEG_PRESET="${FFMPEG_PRESET:-veryfast}"

mkdir -p "$DATA_ROOT"

ARGS=(
  scripts/download_youtube_sl25.py
  --output-dir "$DATA_ROOT/videos_512"
  --raw-dir "$DATA_ROOT/raw"
  --subtitles-dir "$DATA_ROOT/subtitles"
  --status-csv "$DATA_ROOT/download_status.csv"
  --archive "$DATA_ROOT/yt-dlp-archive.txt"
  --cookies "scripts/youtube_cookies_c.txt"
  --delay-min "$DELAY_MIN"
  --delay-max "$DELAY_MAX"
  --sleep "$SLEEP"
  --crf "$CRF"
  --ffmpeg-preset "$FFMPEG_PRESET"
  --sub-format "$SUB_FORMAT"
)

if [[ -n "$LIMIT" ]]; then ARGS+=(--limit "$LIMIT"); fi
if [[ -n "$START" ]]; then ARGS+=(--start "$START"); fi
if [[ -n "$COOKIES" ]]; then ARGS+=(--cookies "$COOKIES"); fi
if [[ -n "$SUB_LANGS" ]]; then ARGS+=(--sub-langs "$SUB_LANGS"); fi
if [[ "$KEEP_RAW" == "1" ]]; then ARGS+=(--keep-raw); fi
if [[ "$OVERWRITE" == "1" ]]; then ARGS+=(--overwrite); fi
if [[ "$NO_WRITE_SUBS" == "1" ]]; then ARGS+=(--no-write-subs); fi
if [[ "$WRITE_AUTO_SUBS" == "1" ]]; then ARGS+=(--write-auto-subs); fi

echo "Writing YouTube-SL-25 data to $DATA_ROOT"
echo "Log file: $DATA_ROOT/download.log"
exec "${ARGS[@]}"
