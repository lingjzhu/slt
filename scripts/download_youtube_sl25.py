#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v")
IGNORED_SIDECAR_SUFFIXES = (".part", ".ytdl", ".tmp")
SUBTITLE_ARCHIVE_SUFFIX = ".subtitles.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the YouTube-SL-25 videos with yt-dlp and transcode them "
            "to 512x512 mp4 files. Reruns skip completed files and resume partial downloads."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).with_name("youtube-sl-25_youtube-sl-25-metadata.csv"),
        help="CSV containing YouTube ids in the first column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/youtube-sl-25/videos_512"),
        help="Directory for final 512x512 mp4 videos.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/youtube-sl-25/raw"),
        help="Directory for yt-dlp partial/raw downloads.",
    )
    parser.add_argument(
        "--status-csv",
        type=Path,
        default=Path("data/youtube-sl-25/download_status.csv"),
        help="CSV log updated after each video.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/youtube-sl-25/yt-dlp-archive.txt"),
        help="yt-dlp download archive used to avoid redownloading completed raw videos.",
    )
    parser.add_argument(
        "--subtitles-dir",
        type=Path,
        default=Path("data/youtube-sl-25/subtitles"),
        help="Directory for per-video subtitle zip archives.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download at most this many rows after --start. Useful for smoke tests.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based row offset in the metadata CSV.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw yt-dlp downloads after successful 512x512 transcode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate final mp4 files even when they already exist.",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        default="veryfast",
        help="x264 preset for downsampled videos.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="x264 CRF for downsampled videos. Higher means smaller/lower quality.",
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Optional cookies.txt for yt-dlp if YouTube asks for auth.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds for yt-dlp to sleep inside each invocation.",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=60.0,
        help="Minimum seconds to wait between video attempts.",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=180.0,
        help="Maximum seconds to wait between video attempts.",
    )
    parser.add_argument(
        "--sub-langs",
        default="all",
        help="Uploaded subtitle languages to request from yt-dlp. Default: all.",
    )
    parser.add_argument(
        "--sub-format",
        default="vtt/best",
        help="Subtitle format preference for yt-dlp. Default: vtt/best.",
    )
    parser.add_argument(
        "--no-write-subs",
        action="store_true",
        help="Do not download subtitles.",
    )
    parser.add_argument(
        "--write-auto-subs",
        action="store_true",
        help="Also download YouTube auto-generated subtitles. Off by default to avoid huge auto-translation fanout.",
    )
    return parser.parse_args()


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required executable: {name}")


def ytdlp_cmd() -> list[str]:
    executable = shutil.which("yt-dlp")
    if executable is not None:
        return [executable]

    probe = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        return [sys.executable, "-m", "yt_dlp"]

    raise SystemExit(
        "Missing yt-dlp. Install it with `python -m pip install --user yt-dlp` "
        "and either add ~/.local/bin to PATH or run this script with the same Python."
    )


def read_video_ids(metadata: Path) -> list[str]:
    ids: list[str] = []
    with metadata.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            video_id = row[0].strip()
            if not video_id or video_id.lower() in {"youtube_id", "video_id", "id"}:
                continue
            ids.append(video_id)
    return ids


def youtube_url(video_id_or_url: str) -> str:
    if video_id_or_url.startswith(("http://", "https://")):
        return video_id_or_url
    return f"https://www.youtube.com/watch?v={video_id_or_url}"


def find_raw_video(raw_dir: Path, video_id: str) -> Path | None:
    candidates = [
        path
        for path in raw_dir.glob(f"{video_id}.*")
        if path.suffix.lower() in VIDEO_EXTENSIONS and not path.name.endswith(".part")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def subtitle_archive_path(subtitles_dir: Path, video_id: str) -> Path:
    return subtitles_dir / f"{video_id}{SUBTITLE_ARCHIVE_SUFFIX}"


def count_zip_members(path: Path) -> int:
    if not path.exists():
        return 0
    with zipfile.ZipFile(path) as zf:
        return len([name for name in zf.namelist() if not name.endswith("/")])


def loose_subtitle_paths(subtitles_dir: Path, video_id: str) -> list[Path]:
    archive_path = subtitle_archive_path(subtitles_dir, video_id)
    return [
        path
        for path in sorted(subtitles_dir.glob(f"{video_id}.*"))
        if path.is_file()
        and path != archive_path
        and not any(path.name.endswith(suffix) for suffix in IGNORED_SIDECAR_SUFFIXES)
    ]


def compress_subtitles(subtitles_dir: Path, video_id: str) -> tuple[Path | None, int]:
    subtitles_dir.mkdir(parents=True, exist_ok=True)
    archive_path = subtitle_archive_path(subtitles_dir, video_id)
    loose_paths = loose_subtitle_paths(subtitles_dir, video_id)
    if not loose_paths:
        if archive_path.exists():
            return archive_path, count_zip_members(archive_path)
        return None, 0

    existing_members: dict[str, bytes] = {}
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as zf:
            existing_members = {
                name: zf.read(name)
                for name in zf.namelist()
                if not name.endswith("/")
            }

    tmp_path = archive_path.with_name(archive_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in sorted(existing_members.items()):
            zf.writestr(name, data)
        for path in loose_paths:
            zf.write(path, arcname=path.name)
    tmp_path.replace(archive_path)

    for path in loose_paths:
        path.unlink(missing_ok=True)

    return archive_path, count_zip_members(archive_path)


def collect_subtitles(raw_dir: Path, subtitles_dir: Path, video_id: str) -> tuple[Path | None, int]:
    subtitles_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(raw_dir.glob(f"{video_id}.*")):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            continue
        if any(path.name.endswith(suffix) for suffix in IGNORED_SIDECAR_SUFFIXES):
            continue
        target = subtitles_dir / path.name
        if target.exists():
            path.unlink(missing_ok=True)
        else:
            path.replace(target)
    return compress_subtitles(subtitles_dir, video_id)


def run_yt_dlp(
    ytdlp: list[str],
    video_id: str,
    url: str,
    raw_dir: Path,
    archive: Path,
    cookies: Path | None,
    sleep: float,
    write_subs: bool,
    sub_langs: str,
    sub_format: str,
    write_auto_subs: bool,
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        *ytdlp,
        "--js-runtimes",
        "node",
        "--continue",
        "--no-overwrites",
        "--ignore-errors",
        "--download-archive",
        str(archive),
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*[height<=720]/b[height<=720]/bv*/b",
        "-o",
        str(raw_dir / "%(id)s.%(ext)s"),
    ]
    if sleep > 0:
        cmd.extend(["--sleep-interval", str(sleep), "--max-sleep-interval", str(sleep)])
    if cookies is not None:
        cmd.extend(["--cookies", str(cookies)])
    if write_subs:
        cmd.extend(
            [
                "--write-subs",
                "--sub-langs",
                sub_langs,
                "--sub-format",
                sub_format,
            ]
        )
        if write_auto_subs:
            cmd.append("--write-auto-subs")
    cmd.append(url)

    subprocess.run(cmd, check=True)
    raw_path = find_raw_video(raw_dir, video_id)
    if raw_path is None:
        raise RuntimeError("yt-dlp finished but no raw video file was found")
    return raw_path


def run_subtitle_download(
    ytdlp: list[str],
    video_id: str,
    url: str,
    raw_dir: Path,
    subtitles_dir: Path,
    cookies: Path | None,
    sub_langs: str,
    sub_format: str,
    write_auto_subs: bool,
) -> tuple[Path | None, int]:
    archive_path, subtitle_count = compress_subtitles(subtitles_dir, video_id)
    if archive_path is not None:
        return archive_path, subtitle_count

    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *ytdlp,
        "--js-runtimes",
        "node",
        "--skip-download",
        "--ignore-errors",
        "--write-subs",
        "--sub-langs",
        sub_langs,
        "--sub-format",
        sub_format,
        "-o",
        str(raw_dir / "%(id)s.%(ext)s"),
    ]
    if write_auto_subs:
        cmd.append("--write-auto-subs")
    if cookies is not None:
        cmd.extend(["--cookies", str(cookies)])
    cmd.append(url)
    subprocess.run(cmd, check=True)
    return collect_subtitles(raw_dir, subtitles_dir, video_id)


def transcode_512(raw_path: Path, output_path: Path, preset: str, crf: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    tmp_path.unlink(missing_ok=True)

    vf = (
        "scale=512:512:force_original_aspect_ratio=decrease,"
        "pad=512:512:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_path),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp_path),
    ]
    subprocess.run(cmd, check=True)
    tmp_path.replace(output_path)


def append_status(status_csv: Path, row: dict[str, str]) -> None:
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "updated_at",
        "video_id",
        "status",
        "url",
        "raw_path",
        "output_path",
        "message",
    ]
    write_header = not status_csv.exists()
    with status_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sleep_between_videos(delay_min: float, delay_max: float) -> None:
    if delay_min <= 0 and delay_max <= 0:
        return
    if delay_min < 0 or delay_max < 0:
        raise ValueError("--delay-min and --delay-max must be >= 0")
    low = min(delay_min, delay_max)
    high = max(delay_min, delay_max)
    delay = random.uniform(low, high)
    print(f"Sleeping {delay:.1f}s before next video.", flush=True)
    time.sleep(delay)


def main() -> int:
    args = parse_args()
    ytdlp = ytdlp_cmd()
    require_binary("ffmpeg")

    video_ids = read_video_ids(args.metadata)
    selected_ids = video_ids[args.start :]
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]

    print(f"Found {len(video_ids)} ids; processing {len(selected_ids)}.", flush=True)
    completed = 0
    failed = 0

    for idx, video_id in enumerate(selected_ids, start=args.start):
        did_attempt = False
        url = youtube_url(video_id)
        output_path = args.output_dir / f"{video_id}.mp4"
        if output_path.exists() and not args.overwrite:
            subtitle_archive: Path | None = None
            subtitle_count = 0
            if not args.no_write_subs:
                try:
                    did_attempt = True
                    print(f"[{idx}] fetch subtitles {video_id}", flush=True)
                    subtitle_archive, subtitle_count = run_subtitle_download(
                        ytdlp=ytdlp,
                        video_id=video_id,
                        url=url,
                        raw_dir=args.raw_dir,
                        subtitles_dir=args.subtitles_dir,
                        cookies=args.cookies,
                        sub_langs=args.sub_langs,
                        sub_format=args.sub_format,
                        write_auto_subs=args.write_auto_subs,
                    )
                except subprocess.CalledProcessError as exc:
                    failed += 1
                    message = f"subtitle command failed with exit code {exc.returncode}"
                    print(f"[{idx}] failed subtitles {video_id}: {message}", file=sys.stderr, flush=True)
                    append_status(
                        args.status_csv,
                        {
                            "updated_at": now_iso(),
                            "video_id": video_id,
                            "status": "failed_subtitles",
                            "url": url,
                            "raw_path": "",
                            "output_path": str(output_path),
                            "message": message,
                        },
                    )
                    print(f"[{idx}] Sleeping for 20 minutes due to potential soft ban...", flush=True)
                    time.sleep(1200)
                    continue
            print(f"[{idx}] skip existing {video_id}", flush=True)
            completed += 1
            append_status(
                args.status_csv,
                {
                    "updated_at": now_iso(),
                    "video_id": video_id,
                    "status": "skipped_existing",
                    "url": url,
                    "raw_path": "",
                    "output_path": str(output_path),
                    "message": f"subtitle_archive={subtitle_archive or ''}; subtitles={subtitle_count}",
                },
            )
            continue

        try:
            did_attempt = True
            print(f"[{idx}] download {video_id}", flush=True)
            raw_path = find_raw_video(args.raw_dir, video_id)
            if raw_path is None:
                raw_path = run_yt_dlp(
                    ytdlp=ytdlp,
                    video_id=video_id,
                    url=url,
                    raw_dir=args.raw_dir,
                    archive=args.archive,
                    cookies=args.cookies,
                    sleep=args.sleep,
                    write_subs=not args.no_write_subs,
                    sub_langs=args.sub_langs,
                    sub_format=args.sub_format,
                    write_auto_subs=args.write_auto_subs,
                )

            print(f"[{idx}] transcode {video_id} -> {output_path}", flush=True)
            transcode_512(
                raw_path=raw_path,
                output_path=output_path,
                preset=args.ffmpeg_preset,
                crf=args.crf,
            )
            subtitle_archive: Path | None = None
            subtitle_count = 0
            if not args.no_write_subs:
                subtitle_archive, subtitle_count = collect_subtitles(args.raw_dir, args.subtitles_dir, video_id)
            if not args.keep_raw:
                raw_path.unlink(missing_ok=True)

            completed += 1
            append_status(
                args.status_csv,
                {
                    "updated_at": now_iso(),
                    "video_id": video_id,
                    "status": "done",
                    "url": url,
                    "raw_path": str(raw_path),
                    "output_path": str(output_path),
                    "message": f"subtitle_archive={subtitle_archive or ''}; subtitles={subtitle_count}",
                },
            )
        except subprocess.CalledProcessError as exc:
            failed += 1
            message = f"command failed with exit code {exc.returncode}"
            print(f"[{idx}] failed {video_id}: {message}", file=sys.stderr, flush=True)
            append_status(
                args.status_csv,
                {
                    "updated_at": now_iso(),
                    "video_id": video_id,
                    "status": "failed",
                    "url": url,
                    "raw_path": "",
                    "output_path": str(output_path),
                    "message": message,
                },
            )
            print(f"[{idx}] Sleeping for 20 minutes due to potential soft ban...", flush=True)
            time.sleep(1200)
        except Exception as exc:
            failed += 1
            print(f"[{idx}] failed {video_id}: {exc}", file=sys.stderr, flush=True)
            append_status(
                args.status_csv,
                {
                    "updated_at": now_iso(),
                    "video_id": video_id,
                    "status": "failed",
                    "url": url,
                    "raw_path": "",
                    "output_path": str(output_path),
                    "message": str(exc),
                },
            )
        finally:
            is_last = idx == args.start + len(selected_ids) - 1
            if did_attempt and not is_last:
                sleep_between_videos(args.delay_min, args.delay_max)

    print(f"Finished. completed/skipped={completed}, failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
